//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/storage/table/table_statistics.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/mutex.hpp"

namespace duckdb {
class PersistentTableData;

class TableStatisticsLock {
public:
	TableStatisticsLock(mutex &l) : guard(l) {}

	lock_guard<mutex> guard;
};

class TableStatistics {
public:
	void Initialize(const vector<LogicalType> &types, PersistentTableData &data);
	void InitializeEmpty(const vector<LogicalType> &types);

	void InitializeAddColumn(TableStatistics &parent, const LogicalType &new_column_type);
	void InitializeRemoveColumn(TableStatistics &parent, idx_t removed_column);
	void InitializeAlterType(TableStatistics &parent, idx_t changed_idx, const LogicalType &new_type);

	void MergeStats(idx_t i, BaseStatistics &stats);
	void MergeStats(TableStatisticsLock &lock, idx_t i, BaseStatistics &stats);

	unique_ptr<BaseStatistics> CopyStats(idx_t i);
	BaseStatistics &GetStats(idx_t i);

	bool Empty();

	unique_ptr<TableStatisticsLock> GetLock();
private:
	//! The statistics lock
	mutex stats_lock;
	//! Column statistics
	vector<unique_ptr<BaseStatistics>> column_stats;
};

} // namespace duckdb
