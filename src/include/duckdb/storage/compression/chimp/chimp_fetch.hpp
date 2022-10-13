//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/storage/compression/chimp/chimp_fetch.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/storage/compression/chimp/chimp.hpp"
#include "duckdb/storage/compression/chimp/chimp_scan.hpp"

#include "duckdb/common/limits.hpp"
#include "duckdb/common/types/null_value.hpp"
#include "duckdb/function/compression/compression.hpp"
#include "duckdb/function/compression_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/statistics/numeric_statistics.hpp"
#include "duckdb/storage/table/column_data_checkpointer.hpp"
#include "duckdb/storage/table/column_segment.hpp"
#include "duckdb/common/operator/subtract.hpp"

namespace duckdb {

template <class T>
void ChimpFetchRow(ColumnSegment &segment, ColumnFetchState &state, row_t row_id, Vector &result, idx_t result_idx) {
	using INTERNAL_TYPE = typename ChimpType<T>::type;

	ChimpScanState<T> scan_state(segment);
	scan_state.Skip(segment, row_id);
	auto result_data = FlatVector::GetData<INTERNAL_TYPE>(result);
	result_data[result_idx] = Chimp128Decompression<INTERNAL_TYPE>::Load(
	    scan_state.group_state.GetFlag(), scan_state.group_state.leading_zeros,
	    scan_state.group_state.leading_zero_index, scan_state.group_state.unpacked_data_blocks,
	    scan_state.group_state.unpacked_index, scan_state.chimp_state);
}

} // namespace duckdb
