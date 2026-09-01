#include "duckdb/execution/operator/helper/physical_limit.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_limit.hpp"
#include "duckdb/execution/operator/helper/physical_limit_percent.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"

namespace duckdb {

bool UseBatchLimit(ClientContext &context, PhysicalOperator &child_node, BoundLimitNode &limit_val,
                   BoundLimitNode &offset_val) {
#ifdef DUCKDB_ALTERNATIVE_VERIFY
	return true;
#else
	// we only want to use the batch limit when we are executing a complex query (e.g. involving a filter or join)
	// if we are doing a limit over a table scan we are otherwise scanning a lot of rows just to throw them away
	reference<PhysicalOperator> current_ref(child_node);
	bool finished = false;
	while (!finished) {
		auto &current_op = current_ref.get();
		switch (current_op.type) {
		case PhysicalOperatorType::TABLE_SCAN: {
			auto &table_scan = current_op.Cast<PhysicalTableScan>();
			if (table_scan.table_filters && !table_scan.table_filters->filters.empty()) {
				// A parallel table scan keeps roughly one block pinned per column, per thread, as long as it
				// is scanning a row group. For a wide projection this can use more memory even at LIMIT 1, where
				// the batch limit itself buffers only a single row per thread. Use non-parallel streaming
				// if the projection is wider than parallel_scan_columns_limit.
				auto max_scan_columns = Settings::Get<ParallelScanColumnsLimitSetting>(context);
				if (table_scan.column_ids.size() > max_scan_columns) {
					return false;
				}
				finished = true;
				break;
			}
			// limit on a table scan without filters - never use batch limit
			return false;
		}
		case PhysicalOperatorType::PROJECTION:
			current_ref = current_op.children[0];
			break;
		default:
			finished = true;
			break;
		}
	}
	// we only use batch limit when we are computing a small amount of values
	// as the batch limit materializes this many rows PER thread
	static constexpr const idx_t BATCH_LIMIT_THRESHOLD = 10000;

	if (limit_val.Type() != LimitNodeType::CONSTANT_VALUE) {
		return false;
	}
	if (offset_val.Type() == LimitNodeType::EXPRESSION_VALUE) {
		return false;
	}
	idx_t total_offset = limit_val.GetConstantValue();
	if (offset_val.Type() == LimitNodeType::CONSTANT_VALUE) {
		total_offset += offset_val.GetConstantValue();
	}
	return total_offset <= BATCH_LIMIT_THRESHOLD;
#endif
}

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalLimit &op) {
	D_ASSERT(op.children.size() == 1);
	auto &plan = CreatePlan(*op.children[0]);

	switch (op.limit_val.Type()) {
	case LimitNodeType::EXPRESSION_PERCENTAGE:
	case LimitNodeType::CONSTANT_PERCENTAGE: {
		auto &limit = Make<PhysicalLimitPercent>(op.types, std::move(op.limit_val), std::move(op.offset_val),
		                                         op.estimated_cardinality);
		limit.children.push_back(plan);
		return limit;
	}
	default: {
		if (!PreserveInsertionOrder(plan)) {
			// use parallel streaming limit if insertion order is not important
			auto &limit = Make<PhysicalStreamingLimit>(op.types, std::move(op.limit_val), std::move(op.offset_val),
			                                           op.estimated_cardinality, true);
			limit.children.push_back(plan);
			return limit;
		}

		// maintaining insertion order is important
		if (UseBatchIndex(plan) && UseBatchLimit(context, plan, op.limit_val, op.offset_val)) {
			// source supports batch index: use parallel batch limit
			auto &limit = Make<PhysicalLimit>(op.types, std::move(op.limit_val), std::move(op.offset_val),
			                                  op.estimated_cardinality);
			limit.children.push_back(plan);
			return limit;
		}

		// source does not support batch index: use a non-parallel streaming limit
		auto &limit = Make<PhysicalStreamingLimit>(op.types, std::move(op.limit_val), std::move(op.offset_val),
		                                           op.estimated_cardinality, false);
		limit.children.push_back(plan);
		return limit;
	}
	}
}

} // namespace duckdb
