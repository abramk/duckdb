//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/storage/statistics/validity_statistics.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/hyperloglog.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"

namespace duckdb {
class Serializer;
class Deserializer;
class Vector;

class DistinctStatistics : public BaseStatistics {
public:
	DistinctStatistics();
	explicit DistinctStatistics(unique_ptr<HyperLogLog> log);

	//! The HLL of the segment
	unique_ptr<HyperLogLog> log;

public:
	void Merge(const BaseStatistics &other) override;

	unique_ptr<BaseStatistics> Copy() const override;

	void Serialize(Serializer &serializer) const override;
	void Serialize(FieldWriter &writer) const override;

	static unique_ptr<DistinctStatistics> Deserialize(Deserializer &source);
	static unique_ptr<DistinctStatistics> Deserialize(FieldReader &reader);

	void Update(Vector &update, idx_t count);

	string ToString() const override;
};

} // namespace duckdb
