// Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0
//
//! PyO3 wrapper around Dynamo's `LocalKvIndexer`.
//!
//! This gives the TensorRT-LLM KV-event worker an in-process radix-tree +
//! ring-buffer of recent events, so a remote consumer (the standalone global
//! indexer) can recover missed events or pull a full snapshot directly from the
//! worker -- replacing the old fixed-size ZMQ replay deque.
//!
//! Design notes (see also trtllm-local-indexer-plan):
//! - The indexer is fed `event_id == seq`, the same ZMQ batch sequence the
//!   publisher already stamps. The consumer keeps detecting gaps on `seq`.
//! - `LocalKvIndexer`'s inner `KvIndexer` spawns its own OS thread + runtime for
//!   the radix-tree work, so the only thing this wrapper's runtime does is drive
//!   the brief async hand-offs (`apply_event_with_buffer`) and the dump build
//!   (`get_events_in_id_range`, which `tokio::spawn`s -> needs a multi-thread rt).
//! - Every method releases the GIL (`py.allow_threads`) so the Python asyncio
//!   loop keeps running while we block on the indexer.

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use tokio::runtime::Runtime;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::indexer::{KvIndexerMetrics, LocalKvIndexer};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    RouterEvent,
};
use dynamo_kv_router::zmq_wire::create_stored_block_from_parts;

/// In-process KV indexer for one worker.
///
/// Holds the authoritative `LocalKvIndexer` (radix tree + replay ring buffer +
/// snapshot cache) and a small tokio runtime used only to drive its async API
/// from synchronous Python.
#[pyclass]
struct LocalIndexer {
    inner: Arc<LocalKvIndexer>,
    rt: Runtime,
    token: CancellationToken,
    worker_id: u64,
    kv_block_size: u32,
}

#[pymethods]
impl LocalIndexer {
    /// Create a new local indexer.
    ///
    /// Args:
    ///     worker_id: stable id stamped on emitted `RouterEvent`s. Informational
    ///         for recovery -- the consumer applies events under the worker it
    ///         queried -- so any stable value (e.g. 0) is fine.
    ///     kv_block_size: KV-cache block size in tokens; must be > 0 and must
    ///         match the standalone indexer's block size (so `tokens_hash`
    ///         agrees across the live ZMQ path and the recovery path).
    ///     max_buffer_size: capacity of the replay ring buffer (events).
    #[new]
    #[pyo3(signature = (worker_id, kv_block_size, max_buffer_size=10000))]
    fn new(worker_id: u64, kv_block_size: u32, max_buffer_size: usize) -> PyResult<Self> {
        if kv_block_size == 0 {
            return Err(PyRuntimeError::new_err("kv_block_size must be > 0"));
        }

        // Multi-thread: get_events_in_id_range() spawns a dump-build task and
        // awaits it, so block_on needs worker threads to make progress. Two is
        // plenty -- the heavy radix-tree work runs on the inner indexer's own
        // thread, not here.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("kv-local-indexer-rt")
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to build tokio runtime: {e}")))?;

        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let inner = Arc::new(LocalKvIndexer::new(
            token.clone(),
            kv_block_size,
            metrics,
            max_buffer_size,
        ));

        Ok(Self {
            inner,
            rt,
            token,
            worker_id,
            kv_block_size,
        })
    }

    /// Feed a BlockStored event. `event_id` is set to `seq`.
    ///
    /// `token_ids` is the flat concatenation of the full blocks' tokens (length
    /// == len(block_hashes) * kv_block_size); it is split per block here.
    /// `block_hashes` / `parent_hash` are the signed-i64 external hashes the
    /// publisher already computes; `tokens_hash` is computed in Rust from the
    /// tokens (same hashing the standalone indexer uses on ingest).
    #[pyo3(signature = (seq, token_ids, block_hashes, parent_hash=None, dp_rank=0, lora_name=None, is_eagle=None))]
    fn apply_stored(
        &self,
        py: Python<'_>,
        seq: u64,
        token_ids: Vec<u32>,
        block_hashes: Vec<i64>,
        parent_hash: Option<i64>,
        dp_rank: u32,
        lora_name: Option<String>,
        is_eagle: Option<bool>,
    ) -> PyResult<()> {
        let inner = self.inner.clone();
        let rt = &self.rt;
        let kv_block_size = self.kv_block_size;
        let worker_id = self.worker_id;

        py.allow_threads(move || {
            let bs = kv_block_size as usize;
            let mut blocks = Vec::with_capacity(block_hashes.len());
            for (i, &bh) in block_hashes.iter().enumerate() {
                let start = i * bs;
                let end = start + bs;
                let block_tokens: &[u32] = token_ids.get(start..end).unwrap_or(&[]);
                blocks.push(create_stored_block_from_parts(
                    kv_block_size,
                    bh as u64,
                    block_tokens,
                    lora_name.as_deref(),
                    None, // mm_extra_info: TODO multimodal support
                    is_eagle,
                ));
            }

            let event = KvCacheEvent {
                event_id: seq,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent_hash.map(ExternalSequenceBlockHash::from),
                    start_position: None,
                    blocks,
                }),
                dp_rank,
            };
            let router_event = RouterEvent::new(worker_id, event);

            rt.block_on(inner.apply_event_with_buffer(router_event))
                .map_err(|e| PyRuntimeError::new_err(format!("apply_stored failed: {e}")))
        })
    }

    /// Feed a BlockRemoved event. `event_id` is set to `seq`.
    #[pyo3(signature = (seq, block_hashes, dp_rank=0))]
    fn apply_removed(
        &self,
        py: Python<'_>,
        seq: u64,
        block_hashes: Vec<i64>,
        dp_rank: u32,
    ) -> PyResult<()> {
        let inner = self.inner.clone();
        let rt = &self.rt;
        let worker_id = self.worker_id;

        py.allow_threads(move || {
            let hashes = block_hashes
                .into_iter()
                .map(ExternalSequenceBlockHash::from)
                .collect();
            let event = KvCacheEvent {
                event_id: seq,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes,
                }),
                dp_rank,
            };
            let router_event = RouterEvent::new(worker_id, event);

            rt.block_on(inner.apply_event_with_buffer(router_event))
                .map_err(|e| PyRuntimeError::new_err(format!("apply_removed failed: {e}")))
        })
    }

    /// Feed an AllBlocksCleared event. `event_id` is set to `seq`.
    #[pyo3(signature = (seq, dp_rank=0))]
    fn apply_cleared(&self, py: Python<'_>, seq: u64, dp_rank: u32) -> PyResult<()> {
        let inner = self.inner.clone();
        let rt = &self.rt;
        let worker_id = self.worker_id;

        py.allow_threads(move || {
            let event = KvCacheEvent {
                event_id: seq,
                data: KvCacheEventData::Cleared,
                dp_rank,
            };
            let router_event = RouterEvent::new(worker_id, event);

            rt.block_on(inner.apply_event_with_buffer(router_event))
                .map_err(|e| PyRuntimeError::new_err(format!("apply_cleared failed: {e}")))
        })
    }

    /// Recovery query: return events for `[start, end]` as a JSON string.
    ///
    /// `start=None` requests a full snapshot. The JSON is an externally-tagged
    /// `WorkerKvQueryResponse`: one of `Events` / `TreeDump` / `TooNew` /
    /// `InvalidRange` / `Error`. The HTTP recovery handler returns this body
    /// verbatim. May be heavy (a full tree dump), so callers should run it off
    /// the asyncio loop (executor thread).
    #[pyo3(signature = (start=None, end=None))]
    fn get_events_json(
        &self,
        py: Python<'_>,
        start: Option<u64>,
        end: Option<u64>,
    ) -> PyResult<String> {
        let inner = self.inner.clone();
        let rt = &self.rt;

        py.allow_threads(move || {
            let response = rt.block_on(inner.get_events_in_id_range(start, end));
            serde_json::to_string(&response)
                .map_err(|e| PyRuntimeError::new_err(format!("failed to serialize response: {e}")))
        })
    }

    /// Signal the inner indexer thread to stop. Safe to call multiple times.
    fn shutdown(&self) {
        self.token.cancel();
    }
}

#[pymodule]
fn kv_local_indexer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LocalIndexer>()?;
    Ok(())
}
