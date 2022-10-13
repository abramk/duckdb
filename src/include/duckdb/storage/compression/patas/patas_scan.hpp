//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/storage/compression/chimp/chimp_scan.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/storage/compression/chimp/chimp.hpp"
#include "duckdb/storage/compression/chimp/algorithm/chimp_utils.hpp"

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

//! Do not change order of these variables
struct PatasUnpackedValueStats {
	uint8_t significant_bytes;
	uint8_t trailing_zeros;
	uint8_t index_diff;
};

template <class EXACT_TYPE>
struct PatasGroupState {
public:
	void Init(uint8_t *data) {
		byte_reader.SetStream(data);
	}

	void Reset() {
		index = 0;
	}

	void LoadPackedData(uint16_t *packed_data, idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			auto &unpacked = unpacked_data[i];
			PackedDataUtils<EXACT_TYPE>::Unpack(packed_data[i], (UnpackedData &)unpacked);
		}
	}

	void Scan(uint8_t *dest, idx_t count) {
		memcpy(dest, (void *)(values + index), sizeof(EXACT_TYPE) * count);
		index += count;
	}

	void LoadValues(EXACT_TYPE *value_buffer, idx_t count) {
		value_buffer[0] = (EXACT_TYPE)0;
		for (idx_t i = 0; i < count; i++) {
			value_buffer[i] = patas::PatasDecompression<EXACT_TYPE>::DecompressValue(
			    byte_reader, unpacked_data[i].significant_bytes, unpacked_data[i].trailing_zeros,
			    value_buffer[i - unpacked_data[i].index_diff]);
		}
	}

public:
	idx_t index;
	PatasUnpackedValueStats unpacked_data[PatasPrimitives::PATAS_GROUP_SIZE];
	EXACT_TYPE values[PatasPrimitives::PATAS_GROUP_SIZE];

private:
	ByteReader byte_reader;
};

template <class T>
struct PatasScanState : public SegmentScanState {
public:
	using EXACT_TYPE = typename FloatingToExact<T>::type;

	explicit PatasScanState(ColumnSegment &segment) : segment(segment), count(segment.count) {
		auto &buffer_manager = BufferManager::GetBufferManager(segment.db);

		handle = buffer_manager.Pin(segment.block);
		auto dataptr = handle.Ptr();
		// ScanStates never exceed the boundaries of a Segment,
		// but are not guaranteed to start at the beginning of the Block
		auto start_of_data_segment = dataptr + segment.GetBlockOffset() + PatasPrimitives::HEADER_SIZE;
		auto metadata_offset = Load<uint32_t>(dataptr + segment.GetBlockOffset());
		metadata_ptr = dataptr + segment.GetBlockOffset() + metadata_offset;
		group_state.Init(start_of_data_segment);
	}

	BufferHandle handle;
	data_ptr_t metadata_ptr;
	idx_t total_value_count = 0;
	PatasGroupState<EXACT_TYPE> group_state;

	ColumnSegment &segment;
	idx_t count;

	idx_t LeftInGroup() const {
		return PatasPrimitives::PATAS_GROUP_SIZE - (total_value_count % PatasPrimitives::PATAS_GROUP_SIZE);
	}

	inline bool GroupFinished() const {
		return (total_value_count % PatasPrimitives::PATAS_GROUP_SIZE) == 0;
	}

	// Scan up to a group boundary
	template <class EXACT_TYPE>
	void ScanGroup(EXACT_TYPE *values, idx_t group_size) {
		D_ASSERT(group_size <= PatasPrimitives::PATAS_GROUP_SIZE);
		D_ASSERT(group_size <= LeftInGroup());

		if (GroupFinished() && total_value_count < count) {
			if (group_size == PatasPrimitives::PATAS_GROUP_SIZE) {
				LoadGroup(values);
				total_value_count += group_size;
				return;
			} else {
				LoadGroup(group_state.values);
			}
		}
		group_state.Scan((uint8_t *)values, group_size);

		total_value_count += group_size;
	}

	void LoadGroup(EXACT_TYPE *value_buffer) {
		group_state.Reset();

		// Load the offset indicating where a groups data starts
		metadata_ptr -= sizeof(uint32_t);
		auto data_byte_offset = Load<uint32_t>(metadata_ptr);
		D_ASSERT(data_byte_offset < Storage::BLOCK_SIZE);
		//  Only used for point queries
		(void)data_byte_offset;

		idx_t group_size = MinValue((idx_t)PatasPrimitives::PATAS_GROUP_SIZE, (count - total_value_count));
		metadata_ptr -= sizeof(uint16_t) * group_size;
		group_state.LoadPackedData((uint16_t *)metadata_ptr, group_size);

		group_state.LoadValues(value_buffer, group_size);
	}

public:
	//! Skip the next 'skip_count' values, we don't store the values
	// TODO: use the metadata to determine if we can skip a group
	void Skip(ColumnSegment &segment, idx_t skip_count) {
		using EXACT_TYPE = typename FloatingToExact<T>::type;
		EXACT_TYPE buffer[PatasPrimitives::PATAS_GROUP_SIZE];

		while (skip_count) {
			auto skip_size = std::min(skip_count, LeftInGroup());
			ScanGroup(buffer, skip_size);
			skip_count -= skip_size;
		}
	}
};

template <class T>
unique_ptr<SegmentScanState> PatasInitScan(ColumnSegment &segment) {
	auto result = make_unique_base<SegmentScanState, PatasScanState<T>>(segment);
	return result;
}

//===--------------------------------------------------------------------===//
// Scan base data
//===--------------------------------------------------------------------===//
template <class T>
void PatasScanPartial(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result,
                      idx_t result_offset) {
	using EXACT_TYPE = typename FloatingToExact<T>::type;
	auto &scan_state = (PatasScanState<T> &)*state.scan_state;

	T *result_data = FlatVector::GetData<T>(result);
	result.SetVectorType(VectorType::FLAT_VECTOR);

	auto current_result_ptr = (EXACT_TYPE *)(result_data + result_offset);

	idx_t scanned = 0;

	while (scanned < scan_count) {
		const idx_t to_scan = MinValue(scan_count - scanned, scan_state.LeftInGroup());

		scan_state.template ScanGroup<EXACT_TYPE>(current_result_ptr + scanned, to_scan);
		scanned += to_scan;
	}
}

template <class T>
void PatasSkip(ColumnSegment &segment, ColumnScanState &state, idx_t skip_count) {
	auto &scan_state = (PatasScanState<T> &)*state.scan_state;
	scan_state.Skip(segment, skip_count);
}

template <class T>
void PatasScan(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result) {
	PatasScanPartial<T>(segment, state, scan_count, result, 0);
}

} // namespace duckdb
