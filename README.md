# kv-local-indexer

A thin PyO3 wrapper around Dynamo's `LocalKvIndexer`. It gives a TRT-LLM
KV-event worker an in-process radix tree + ring buffer of recent KV events, so
the standalone global indexer can recover missed events (`Events`) or pull a
full snapshot (`TreeDump`) directly from the worker. Replaces the old
fixed-window ZMQ replay deque.

Consumed by `deepinfra/TensorRT-LLM` (`tensorrt_llm/serve/kv_zmq_publisher.py`),
which `pip install`s the prebuilt wheel from this repo's GitHub Releases. That
repo contains **no Rust** — all of it lives here.

## What it exposes

`LocalIndexer(worker_id, kv_block_size, max_buffer_size=10000)` with:

- `apply_stored(seq, token_ids, block_hashes, parent_hash=None, dp_rank=0, lora_name=None, is_eagle=None)`
- `apply_removed(seq, block_hashes, dp_rank=0)`
- `apply_cleared(seq, dp_rank=0)`
- `get_events_json(start=None, end=None) -> str`  (the recovery query)
- `shutdown()`

`seq` is used directly as the event's `event_id` (one event per ZMQ batch).

## Dependency on Dynamo

Depends on `dynamo-kv-router`, pinned in `Cargo.toml` to a **public upstream git
commit** of https://github.com/ai-dynamo/dynamo.git:

    rev = "b2b7868090a22e02a0cf5733eea7d53e8e0f0829"   # ai-dynamo/dynamo main

No local dynamo checkout — cargo fetches that commit during the build. Built with
`default-features = false`, which compiles only the in-process indexer +
protocols. The `standalone-indexer` feature is OFF, so the deepinfra-fork changes
(which live entirely under that feature) are not compiled here; upstream at this
rev is byte-identical to the fork for what this wrapper needs. To bump it, change
`rev`, rebuild (regenerates `Cargo.lock`), commit both.

## The wheel is portable

The compiled extension links only the standard C runtime
(`libc` / `libm` / `libgcc_s` + the loader) — no libpython, no libstdc++,
nothing engine-specific. So the wheel runs in any Linux with a glibc at least as
new as the build box's. `build_wheel.sh` builds in **manylinux_2_28** (glibc
floor 2.28); the engine image is glibc 2.39 / cp312, well above it.

Because it needs nothing from the private `localhost:30500` registry, the build
can run anywhere, including plain GitHub Actions CI.

## Build & release

```bash
./build_wheel.sh                 # -> dist/kv_local_indexer-*.whl (portable, cp312)
./release.sh v0.1.0              # build + publish as a GitHub Release on this repo
```

`release.sh` needs `gh` logged in (`gh auth login` once — a private repo's
release assets require a token; an SSH key alone can't fetch them).

For local dev in an active venv:

```bash
maturin develop --release        # build + install into the active venv
```

## How TensorRT-LLM consumes it

The engine never builds Rust. Its `build_python_changes.sh` downloads the pinned
wheel from this repo's release (`gh release download <KV_VER> --repo
deepinfra/kv-local-indexer`) into `wheels/`, and `Dockerfile.python` does a plain
`pip install` of it.

- **Engine Python-only change:** nothing here changes; the pinned wheel is reused.
- **Rust change here:** PR + `./release.sh vX` → bump `KV_VER` in the engine's
  `build_python_changes.sh` → rebuild the engine image.

## What's committed

- Source (`src/lib.rs`, `Cargo.toml`, build glue).
- `Cargo.lock` — pins every transitive dep for reproducible builds.

The wheel is **not** committed — it ships via GitHub Releases (see `dist/` is
gitignored).
