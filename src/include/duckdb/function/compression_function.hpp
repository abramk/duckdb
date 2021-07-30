//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/compression_function.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/common/enums/compression_type.hpp"
#include "duckdb/common/map.hpp"
#include "duckdb/storage/storage_info.hpp"

namespace duckdb {
class DatabaseInstance;
class ColumnData;
class ColumnDataCheckpointer;
class CompressedSegment;
class SegmentStatistics;

struct ColumnFetchState;
struct ColumnScanState;
struct SegmentScanState;

struct AnalyzeState {
	virtual ~AnalyzeState(){}
};

struct CompressionState {
	virtual ~CompressionState(){}
};

//===--------------------------------------------------------------------===//
// Analyze
//===--------------------------------------------------------------------===//
//! The analyze functions are used to determine whether or not to use this compression method
//! The system first determines the potential compression methods to use based on the physical type of the column
//! After that the following steps are taken:
//! 1. The init_analyze is called to initialize the analyze state of every candidate compression method
//! 2. The analyze method is called with all of the input data in the order in which it must be stored.
//!    analyze can return "false". In that case, the compression method is taken out of consideration early.
//! 3. The final_analyze method is called, which should return a score for the compression method

//! The system then decides which compression function to use based on the analyzed score (returned from final_analyze)
typedef unique_ptr<AnalyzeState> (*compression_init_analyze_t)(ColumnData &col_data, PhysicalType type);
typedef bool (*compression_analyze_t)(AnalyzeState &state, Vector &input, idx_t count);
typedef idx_t (*compression_final_analyze_t)(AnalyzeState &state);

//===--------------------------------------------------------------------===//
// Compress
//===--------------------------------------------------------------------===//
typedef unique_ptr<CompressionState> (*compression_init_compression_t)(ColumnDataCheckpointer &checkpointer, unique_ptr<AnalyzeState> state);
typedef void (*compression_compress_data_t)(CompressionState& state, Vector &scan_vector, idx_t count);
typedef void (*compression_compress_finalize_t)(CompressionState& state);

//===--------------------------------------------------------------------===//
// Uncompress / Scan
//===--------------------------------------------------------------------===//
typedef unique_ptr<SegmentScanState> (*compression_init_segment_scan_t)(CompressedSegment &segment);
typedef void (*compression_scan_vector_t)(CompressedSegment &segment, ColumnScanState &state, idx_t start, idx_t scan_count, Vector &result);
typedef void (*compression_scan_partial_t)(CompressedSegment &segment, ColumnScanState &state, idx_t start, idx_t scan_count, Vector &result, idx_t result_offset);
typedef void (*compression_fetch_row_t)(CompressedSegment &segment, ColumnFetchState &state, row_t row_id, Vector &result, idx_t result_idx);

//===--------------------------------------------------------------------===//
// Append (optional)
//===--------------------------------------------------------------------===//
typedef void (*compression_init_segment_t)(CompressedSegment &segment, block_id_t block_id);
typedef idx_t (*compression_append_t)(CompressedSegment &segment, SegmentStatistics &stats, VectorData &data, idx_t offset, idx_t count);
typedef void (*compression_revert_append_t)(CompressedSegment &segment, idx_t start_row);

//! The type used for initializing hashed aggregate function states
class CompressionFunction {
public:
	CompressionFunction(CompressionType type, PhysicalType data_type, compression_init_analyze_t init_analyze,
	                    compression_analyze_t analyze, compression_final_analyze_t final_analyze,
	                    compression_init_compression_t init_compression, compression_compress_data_t compress,
	                    compression_compress_finalize_t compress_finalize, compression_init_segment_scan_t init_scan,
	                    compression_scan_vector_t scan_vector, compression_scan_partial_t scan_partial,
	                    compression_fetch_row_t fetch_row, compression_init_segment_t init_segment,
	                    compression_append_t append, compression_revert_append_t revert_append) :
	type(type), data_type(data_type), init_analyze(init_analyze), analyze(analyze), final_analyze(final_analyze),
	init_compression(init_compression), compress(compress), compress_finalize(compress_finalize), init_scan(init_scan),
	scan_vector(scan_vector), scan_partial(scan_partial), fetch_row(fetch_row), init_segment(init_segment), append(append), revert_append(revert_append) {}

	//! Compression type
	CompressionType type;
	//! The data type this function can compress
	PhysicalType data_type;

	//! Analyze step: determine which compression function is the most effective
	//! init_analyze is called once to set up the analyze state
	compression_init_analyze_t init_analyze;
	//! analyze is called several times (once per vector in the row group)
	//! analyze should return true, unless compression is no longer possible with this compression method
	//! in that case false should be returned
	compression_analyze_t analyze;
	//! final_analyze should return the score of the compression function
	//! ideally this is the exact number of bytes required to store the data
	//! this is not required/enforced: it can be an estimate as well
	compression_final_analyze_t final_analyze;

	//! Compression step: actually compress the data
	//! init_compression is called once to set up the comperssion state
	compression_init_compression_t init_compression;
	//! compress is called several times (once per vector in the row group)
	compression_compress_data_t compress;
	//! compress_finalize is called after
	compression_compress_finalize_t compress_finalize;

	//! init_scan is called to set up the scan state
	compression_init_segment_scan_t init_scan;
	//! scan_vector scans an entire vector using the scan state
	compression_scan_vector_t scan_vector;
	//! scan_partial scans a subset of a vector
	//! this can request > vector_size as well
	//! this is used if a vector crosses segment boundaries, or for child columns of lists
	compression_scan_partial_t scan_partial;
	//! fetch an individual row from the compressed vector
	//! used for index lookups
	compression_fetch_row_t fetch_row;

	// Append functions
	//! This only really needs to be defined for uncompressed segments

	//! Initialize a compressed segment (optional)
	compression_init_segment_t init_segment;
	//! Append to the compressed segment (optional)
	compression_append_t append;
	//! Revert append (optional)
	compression_revert_append_t revert_append;
};

//! The set of compression functions
struct CompressionFunctionSet {
	map<CompressionType, map<PhysicalType, CompressionFunction>> functions;
};

} // namespace duckdb
