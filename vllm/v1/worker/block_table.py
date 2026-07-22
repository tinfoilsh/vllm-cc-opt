# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from enum import Enum

import numpy as np
import torch

from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.ccbench_instrumentation import ccbench_instant, ccbench_span
from vllm.v1.utils import CpuGpuBuffer

logger = init_logger(__name__)


def get_block_table_width(
    max_num_blocks: int,
    block_size: int,
    kernel_block_size: int | None = None,
    *,
    token_alignment: int | None = 128,
) -> int:
    """Return the width after optional alignment and virtual block splitting."""
    if kernel_block_size is None:
        kernel_block_size = block_size
    if block_size % kernel_block_size != 0:
        raise ValueError(
            f"kernel_block_size {kernel_block_size} must divide "
            f"block_size {block_size} evenly"
        )
    if token_alignment is not None:
        if token_alignment <= 0:
            raise ValueError("token_alignment must be positive")
        block_alignment = token_alignment // math.gcd(token_alignment, block_size)
        max_num_blocks = cdiv(max_num_blocks, block_alignment) * block_alignment
    return max_num_blocks * block_size // kernel_block_size


class SlotMappingMode(Enum):
    TOKEN_TO_KV_SLOT = "token_to_kv_slot"
    NONE = "none"


class BlockTable:
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int,
        slot_mapping_mode: SlotMappingMode = SlotMappingMode.TOKEN_TO_KV_SLOT,
    ):
        """
        Args:
            block_size: Block size used for KV cache memory allocation
            max_num_reqs: Maximum number of concurrent requests supported.
            max_num_blocks_per_req: Maximum number of blocks per request.
            max_num_batched_tokens: Maximum number of tokens in a batch.
            pin_memory: Whether to pin memory for faster GPU transfers.
            device: Target device for the block table.
            kernel_block_size: The block_size of underlying attention kernel.
                Will be the same as `block_size` if `block_size` is supported
                by the attention kernel.
            slot_mapping_mode: How this cache group maps scheduled tokens to
                cache slots. Mamba-like state caches do not use token slot
                mappings and should use SlotMappingMode.NONE.
        """
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device
        self.kv_cache_block_size = block_size

        if kernel_block_size == block_size:
            # Standard case: allocation and computation use same block size
            # No block splitting needed, direct mapping
            self.block_size = block_size
            self.blocks_per_kv_block = 1
            self.use_hybrid_blocks = False
        else:
            # Hybrid case: allocation block size differs from kernel block size
            # Memory blocks are subdivided to match kernel requirements
            # Example: 32-token memory blocks with 16-token kernel blocks
            # → Each memory block corresponds to 2 kernel blocks
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )

            self.block_size = kernel_block_size
            self.blocks_per_kv_block = block_size // kernel_block_size
            self.use_hybrid_blocks = True

        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block

        self.block_table = self._make_buffer(
            self.max_num_reqs,
            self.max_num_blocks_per_req,
            dtype=torch.int32,
            name="block_table",
        )
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)
        block_table_dirty_update_requested = bool(
            int(os.environ.get("VLLM_CC_BLOCK_TABLE_DIRTY_UPDATE", "0") or "0")
        )
        block_table_dirty_update_cc_enabled = (
            current_platform.is_confidential_compute_enabled()
            if block_table_dirty_update_requested
            else False
        )
        if (
            block_table_dirty_update_requested
            and not block_table_dirty_update_cc_enabled
        ):
            logger.warning_once(
                "Ignoring VLLM_CC_BLOCK_TABLE_DIRTY_UPDATE because "
                "confidential compute is not enabled."
            )
        self._block_table_dirty_update_enabled = (
            block_table_dirty_update_requested
            and block_table_dirty_update_cc_enabled
        )
        self._block_table_dirty_commit_enabled = bool(
            int(os.environ.get("VLLM_CC_DECODE_METADATA_FASTPATH", "0") or "0")
            or self._block_table_dirty_update_enabled
        )
        self._block_table_dirty = False
        self._dirty_rows: list[int] = []
        self._dirty_starts: list[int] = []
        self._dirty_values: list[int] = []
        self._dirty_cu_lens: list[int] = []
        self._dirty_packed_cpu: torch.Tensor | None = None
        self._dirty_packed_gpu: torch.Tensor | None = None

        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens,
            dtype=torch.int64,
            name="slot_mapping",
        )

        if self.use_hybrid_blocks:
            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
                1, -1
            )
        else:
            self._kernel_block_arange = None

        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            # PCP might not be initialized in testing
            self.pcp_world_size = 1
            self.pcp_rank = 0
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size
        self.slot_mapping_mode = slot_mapping_mode

    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        if not block_ids:
            return

        if self.use_hybrid_blocks:
            block_ids = self.map_to_kernel_blocks(
                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange
            )

        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids
        self._block_table_dirty = True
        self._stage_dirty_update(row_idx, start, block_ids)

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)
        self._block_table_dirty = True

    def clear_row(self, row_idx: int) -> None:
        num_blocks = self.num_blocks_per_row[row_idx]
        if num_blocks > 0:
            self.block_table.np[row_idx, :num_blocks] = 0
            self._block_table_dirty = True
            self._stage_dirty_update(row_idx, 0, [0] * num_blocks)
        self.num_blocks_per_row[row_idx] = 0

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        block_table_np = self.block_table.np
        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks
        # Clear the vacated source row: dummy-run batches dereference stale
        # rows as mamba state slots and write state in place there, possibly
        # after the blocks have been freed and reallocated.
        block_table_np[src, :num_blocks] = 0
        self.num_blocks_per_row[src] = 0
        self._block_table_dirty = True
        self._stage_dirty_update(tgt, 0, block_table_np[tgt, :num_blocks])

    def swap_row(self, src: int, tgt: int) -> None:
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]
        self._block_table_dirty = True
        block_table_np = self.block_table.np
        src_blocks = self.num_blocks_per_row[src]
        tgt_blocks = self.num_blocks_per_row[tgt]
        self._stage_dirty_update(src, 0, block_table_np[src, :src_blocks])
        self._stage_dirty_update(tgt, 0, block_table_np[tgt, :tgt_blocks])

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        num_tokens = positions.shape[0]
        if self.slot_mapping_mode == SlotMappingMode.NONE:
            # Mamba/GDN groups consume the block table as recurrent state
            # indices and do not use per-token slot mappings.
            return
        assert self.slot_mapping_mode == SlotMappingMode.TOKEN_TO_KV_SLOT

        _compute_slot_mapping_kernel[(num_reqs + 1,)](
            num_tokens,
            self.max_num_batched_tokens,
            query_start_loc,
            positions,
            self.block_table.gpu,
            self.block_table.gpu.stride(0),
            self.block_size,
            self.slot_mapping.gpu,
            KV_CACHE_BLOCK_SIZE=self.kv_cache_block_size,
            BLOCKS_PER_KV_BLOCK=self.blocks_per_kv_block,
            TOTAL_CP_WORLD_SIZE=self.dcp_world_size,
            TOTAL_CP_RANK=self.dcp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=self.cp_kv_cache_interleave_size,
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )

    def commit_block_table(self, num_reqs: int) -> None:
        if not self._block_table_dirty_commit_enabled:
            self.block_table.copy_to_gpu(num_reqs)
            return

        if not self._block_table_dirty:
            return

        if (
            self._block_table_dirty_update_enabled
            and self._dirty_values
            and self._try_commit_dirty_updates(num_reqs)
        ):
            self._block_table_dirty = False
            return

        self.block_table.copy_to_gpu(num_reqs)
        self._clear_dirty_updates()
        self._block_table_dirty = False

    def clear(self) -> None:
        self.block_table.gpu.fill_(0)
        self.block_table.cpu.fill_(0)
        self._block_table_dirty = False
        self._clear_dirty_updates()

    def _stage_dirty_update(
        self,
        row_idx: int,
        start: int,
        block_ids: list[int] | np.ndarray,
    ) -> None:
        if not self._block_table_dirty_update_enabled:
            return
        if len(block_ids) == 0:
            return

        self._dirty_rows.append(row_idx)
        self._dirty_starts.append(start)
        if isinstance(block_ids, np.ndarray):
            self._dirty_values.extend(int(v) for v in block_ids.tolist())
        else:
            self._dirty_values.extend(int(v) for v in block_ids)
        self._dirty_cu_lens.append(len(self._dirty_values))

    def _clear_dirty_updates(self) -> None:
        self._dirty_rows.clear()
        self._dirty_starts.clear()
        self._dirty_values.clear()
        self._dirty_cu_lens.clear()

    def _ensure_dirty_update_buffer(self, packed_len: int) -> None:
        cur_len = (
            0
            if self._dirty_packed_cpu is None
            else int(self._dirty_packed_cpu.numel())
        )
        if cur_len >= packed_len:
            return

        new_len = 1 << (packed_len - 1).bit_length()
        self._dirty_packed_cpu = torch.empty(
            new_len,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self._dirty_packed_gpu = torch.empty(
            new_len,
            dtype=torch.int32,
            device=self.device,
        )

    def _coalesce_dirty_updates(
        self,
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Return disjoint dirty ranges populated from the final CPU table.

        A row can be cleared, moved, and reused before the next commit. Launching
        those staged writes as separate Triton programs gives overlapping stores
        no ordering guarantee. Merge overlapping or adjacent ranges so each GPU
        destination is written once with the authoritative final CPU value.
        """
        metadata_lengths = {
            len(self._dirty_rows),
            len(self._dirty_starts),
            len(self._dirty_cu_lens),
        }
        if len(metadata_lengths) != 1:
            raise ValueError("Inconsistent block-table dirty-update metadata")

        table_rows, table_cols = self.block_table.np.shape
        total_values = len(self._dirty_values)
        ranges_by_row: dict[int, list[tuple[int, int]]] = {}
        previous_cu_len = 0
        for row, start, cu_len in zip(
            self._dirty_rows,
            self._dirty_starts,
            self._dirty_cu_lens,
        ):
            if cu_len < previous_cu_len or cu_len > total_values:
                raise ValueError("Invalid block-table dirty-update cumulative length")
            update_len = cu_len - previous_cu_len
            previous_cu_len = cu_len
            if update_len <= 0:
                continue
            end = start + update_len
            if not 0 <= row < table_rows:
                raise ValueError(f"Block-table dirty-update row out of bounds: {row}")
            if not 0 <= start < end <= table_cols:
                raise ValueError(
                    "Block-table dirty-update range out of bounds: "
                    f"row={row}, start={start}, end={end}, columns={table_cols}"
                )
            ranges_by_row.setdefault(row, []).append((start, end))

        if previous_cu_len != total_values:
            raise ValueError("Unreferenced block-table dirty-update values")

        rows: list[int] = []
        starts: list[int] = []
        values: list[int] = []
        cu_lens: list[int] = []
        block_table_np = self.block_table.np

        for row in sorted(ranges_by_row):
            ranges = sorted(ranges_by_row[row])
            merged: list[tuple[int, int]] = []
            for start, end in ranges:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))

            for start, end in merged:
                rows.append(row)
                starts.append(start)
                values.extend(int(v) for v in block_table_np[row, start:end])
                cu_lens.append(len(values))

        return rows, starts, values, cu_lens

    def _try_commit_dirty_updates(self, num_reqs: int) -> bool:
        staged_updates = len(self._dirty_rows)
        dirty_rows, dirty_starts, dirty_values, dirty_cu_lens = (
            self._coalesce_dirty_updates()
        )
        n_updates = len(dirty_rows)
        n_values = len(dirty_values)
        if n_updates == 0 or n_values == 0:
            self._clear_dirty_updates()
            return True

        packed_meta_len = 3 * n_updates
        packed_len = packed_meta_len + n_values
        full_copy_elems = num_reqs * self.max_num_blocks_per_req
        if packed_len >= full_copy_elems:
            ccbench_instant(
                "ccbench.block_table.dirty_update.fallback",
                {
                    "reason": "packed_not_smaller",
                    "num_updates": n_updates,
                    "num_values": n_values,
                    "packed_bytes": packed_len * 4,
                    "full_copy_bytes": full_copy_elems * 4,
                },
            )
            return False

        self._ensure_dirty_update_buffer(packed_len)
        assert self._dirty_packed_cpu is not None
        assert self._dirty_packed_gpu is not None
        packed_cpu = self._dirty_packed_cpu[:packed_len]
        packed_gpu = self._dirty_packed_gpu[:packed_len]

        for i, (row, start, cu_len) in enumerate(
            zip(dirty_rows, dirty_starts, dirty_cu_lens)
        ):
            offset = 3 * i
            packed_cpu[offset] = row
            packed_cpu[offset + 1] = start
            packed_cpu[offset + 2] = cu_len
        packed_cpu[packed_meta_len:packed_len] = torch.tensor(
            dirty_values,
            dtype=torch.int32,
            device="cpu",
        )

        with ccbench_span(
            "ccbench.metadata.copy_to_gpu",
            {
                "object_name": "block_table_dirty_update_packed",
                "source": "packed_dirty_update",
                "phase": "prepare_inputs",
                "bytes": packed_len * 4,
                "dtype": "torch.int32",
                "pin_memory": self.pin_memory,
                "staged_updates": staged_updates,
                "num_updates": n_updates,
                "num_values": n_values,
                "full_copy_bytes_avoided": max(0, (full_copy_elems - packed_len) * 4),
            },
        ):
            packed_gpu.copy_(packed_cpu, non_blocking=True)

        with ccbench_span(
            "ccbench.block_table.dirty_update.apply",
            {
                "num_updates": n_updates,
                "staged_updates": staged_updates,
                "num_values": n_values,
                "packed_bytes": packed_len * 4,
                "full_copy_bytes_avoided": max(0, (full_copy_elems - packed_len) * 4),
            },
        ):
            _apply_block_table_dirty_update_kernel[(n_updates,)](
                self.block_table.gpu,
                self.block_table.gpu.stride(0),
                packed_gpu,
                packed_meta_len,
                BLOCK_SIZE=1024,
            )

        ccbench_instant(
            "ccbench.block_table.dirty_update.summary",
            {
                "num_updates": n_updates,
                "staged_updates": staged_updates,
                "num_values": n_values,
                "packed_bytes": packed_len * 4,
                "full_copy_bytes_avoided": max(0, (full_copy_elems - packed_len) * 4),
            },
        )
        self._clear_dirty_updates()
        return True

    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        """Convert kv_manager_block_id IDs to kernel block IDs.

        Example:
            # kv_manager_block_ids: 32 tokens,
            # Kernel block size: 16 tokens
            # blocks_per_kv_block = 2
            >>> kv_manager_block_ids = np.array([0, 1, 2])
            >>> Result: [0, 1, 2, 3, 4, 5]

            # Each kv_manager_block_id maps to 2 kernel block id:
            # kv_manager_block_id 0 → kernel block id [0, 1]
            # kv_manager_block_id 1 → kernel block id [2, 3]
            # kv_manager_block_id 2 → kernel block id [4, 5]
        """
        if blocks_per_kv_block == 1:
            return kv_manager_block_ids

        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )

        return kernel_block_ids.reshape(-1)

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        """Returns the device tensor of the block table."""
        return self.block_table.gpu[:num_reqs]

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table.np

    def _make_buffer(
        self,
        *size: int | torch.SymInt,
        dtype: torch.dtype,
        name: str | None = None,
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            buffer_name=name,
        )


class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks: list[int],
        cp_kv_cache_interleave_size: int = 1,
        slot_mapping_modes: list[SlotMappingMode] | None = None,
    ) -> None:
        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )
        if slot_mapping_modes is None:
            slot_mapping_modes = [SlotMappingMode.TOKEN_TO_KV_SLOT] * len(block_sizes)
        if len(slot_mapping_modes) != len(block_sizes):
            raise ValueError(
                f"slot_mapping_modes length ({len(slot_mapping_modes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        max_num_blocks = [
            (
                get_block_table_width(n, block_size, token_alignment=None)
                if slot_mapping_mode == SlotMappingMode.NONE
                else get_block_table_width(n, block_size)
            )
            for n, block_size, slot_mapping_mode in zip(
                max_num_blocks, block_sizes, slot_mapping_modes
            )
        ]

        self.block_tables = [
            BlockTable(
                block_size,
                max_num_reqs,
                max_num_blocks_per_req,
                max_num_batched_tokens,
                pin_memory,
                device,
                kernel_block_size,
                cp_kv_cache_interleave_size,
                slot_mapping_mode=slot_mapping_mode,
            )
            for (
                block_size,
                kernel_block_size,
                max_num_blocks_per_req,
                slot_mapping_mode,
            ) in zip(
                block_sizes, kernel_block_sizes, max_num_blocks, slot_mapping_modes
            )
        ]

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def clear_row(self, row_idx: int) -> None:
        for block_table in self.block_tables:
            block_table.clear_row(row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        for block_table in self.block_tables:
            block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]


@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def _compute_slot_mapping_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,  # [num_reqs + 1], int32
    positions_ptr,  # [num_tokens], int64
    block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32 (flat)
    block_table_stride,  # max_num_blocks_per_req
    block_size,
    slot_mapping_ptr,  # [max_num_tokens], int64
    KV_CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_KV_BLOCK: tl.constexpr,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    if req_idx == tl.num_programs(0) - 1:
        # Pad remaining slots for CUDA graph compatibility.
        for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE
    row_offset = req_idx * block_table_stride
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
        virtual_block_indices = pos // virtual_block_size
        virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
        is_local = (
            virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
        local_block_offsets = (
            virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
        )

        block_indices = (
            virtual_block_indices * BLOCKS_PER_KV_BLOCK
            + local_block_offsets // block_size
        )
        block_numbers = tl.load(
            block_table_ptr + row_offset + block_indices,
            mask=mask & is_local,
            other=0,
        ).to(tl.int64)
        slot_offsets = local_block_offsets % block_size
        slot_ids = block_numbers * block_size + slot_offsets
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)


@triton.jit
def _apply_block_table_dirty_update_kernel(
    block_table_ptr,
    block_table_stride,
    packed_updates_ptr,
    values_offset,
    BLOCK_SIZE: tl.constexpr,
):
    update_idx = tl.program_id(0)
    meta_offset = update_idx * 3
    row_idx = tl.load(packed_updates_ptr + meta_offset)
    start_idx = tl.load(packed_updates_ptr + meta_offset + 1)
    cu_end = tl.load(packed_updates_ptr + meta_offset + 2)
    cu_start = tl.load(packed_updates_ptr + meta_offset - 1) if update_idx > 0 else 0
    update_len = cu_end - cu_start

    row_ptr = block_table_ptr + row_idx * block_table_stride + start_idx
    values_ptr = packed_updates_ptr + values_offset + cu_start
    for i in range(0, update_len, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < update_len
        values = tl.load(values_ptr + offsets, mask=mask)
        tl.store(row_ptr + offsets, values, mask=mask)
