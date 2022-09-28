//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/parser/parsed_data/create_index_info.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/parser/parsed_data/create_info.hpp"
#include "duckdb/common/enums/index_type.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/parser/parsed_expression.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {

struct CreateIndexInfo : public CreateInfo {
	CreateIndexInfo() : CreateInfo(CatalogType::INDEX_ENTRY) {
	}

	//! Index Type (e.g., B+-tree, Skip-List, ...)
	IndexType index_type;
	//! Name of the Index
	string index_name;
	//! Index Constraint Type
	IndexConstraintType constraint_type;
	//! The table to create the index on
	unique_ptr<BaseTableRef> table;
	//! Set of expressions to index by
	vector<unique_ptr<ParsedExpression>> expressions;
	vector<unique_ptr<ParsedExpression>> parsed_expressions;

	//! Types used for the CREATE INDEX scan
	vector<LogicalType> scan_types;
	//! The names of the columns, used for the CREATE INDEX scan
	vector<string> names;
	//! Column IDs needed for index creation
	vector<column_t> column_ids;

protected:
	void SerializeInternal(Serializer &serializer) const override;

public:
	unique_ptr<CreateInfo> Copy() const override;

	static unique_ptr<CreateIndexInfo> Deserialize(Deserializer &deserializer);
};

} // namespace duckdb
