# Latency Benchmarking Design Notes

## Three Loading Modes

### Mode 1: Batch GET (`AISBatchLoader`)
- **Branch**: `gb-latency`
- **How it works**: Aggregates all object URLs from a CutSet, sends one batch GET request to AIStore. All audio bytes are returned together and injected into the cut manifests (recording sources set to `type="memory"`).
- **Where I/O happens**: Inside `AISBatchLoader.__call__()`, which runs in `AudioSamples.__call__()` before `collate_audio`.
- **Latency measurement**: Straightforward. Time the batch GET call for per-batch latency. Divide by object count for per-object estimate. Implemented with `collect_metrics=True`, p50/p95/p99/avg stats.

### Mode 2: Random Access GET (`AISObjectLoader`)
- **Branch**: `g-latency`
- **How it works**: Makes individual GET requests per object (optionally concurrent via ThreadPoolExecutor). Each object's bytes replace the recording source in the cut manifest.
- **Where I/O happens**: Inside `AISObjectLoader.__call__()`, which runs in `AudioSamples.__call__()` before `collate_audio`.
- **Latency measurement**: Straightforward. Each `_fetch_object()` call is individually timed for per-object latency. Wall-clock time of the entire `__call__()` gives per-batch latency. Same metrics infrastructure as Mode 1.

### Mode 3: Sequential Tarred (`LazyNeMoTarredIterator._iter_sequential`)
- **Branch**: `seq-g-latency`
- **How it works**: Opens tar file in streaming mode (`mode="r|*"`), reads members sequentially. For each tar member, reads raw audio bytes and creates a Recording with `type="memory"`. Cuts are yielded one at a time.
- **Where I/O happens**: Inside `_iter_sequential()` during the DataLoader worker's iteration loop — BEFORE the sampler batches cuts and BEFORE `AudioSamples.__call__()` runs.
- **Latency measurement**: PROBLEMATIC (see below).

## Why Sequential Mode Latency Is Hard

### The data flow
```
_iter_sequential() opens tar once, yields cuts one-by-one
    → each yield does tar.extractfile(tar_info).read() (actual I/O)
    → cut gets Recording(type="memory", source=raw_audio_bytes)
    → sampler accumulates cuts until batch is full (metadata only, no I/O)
    → dataset.__getitem__(batch) receives cuts with audio already in memory
    → AudioSamples.__call__() only decodes in-memory bytes via soundfile (no I/O)
```

### The problems
1. **No batch boundary at I/O level**: The tar reader yields cuts one at a time. The "batch" is defined by the sampler, not the tar reader. I/O for a batch is spread across multiple sampler pulls.
2. **Connection already established**: The tar stream is opened once. Each `.read()` pulls the next N bytes from an already-open stream. Per-object read time is mostly `object_size / stream_throughput`.
3. **Timing at `AudioSamples.__call__()` only measures decode**: By the time `collate_audio` runs, audio is already in memory. The `_timed_read_audio` wrapper only captures `BytesIO` wrapping + soundfile decode (CPU work, not storage I/O).

### Possible alternatives
- **Shard-level throughput**: Time the entire `_iter_sequential()` per shard, divide by total bytes.
- **End-to-end batch latency**: Time from "sampler starts filling batch" to "batch tensor ready" — captures I/O + sampling + decode together.
- **Per-object tar read time**: Instrument `tar.extractfile(tar_info).read()` — but this mostly measures `object_size / throughput` and will be fairly uniform.

## AudioSamples Decode Metrics (implemented on seq-g-latency)

Even though it doesn't capture tar I/O, we added metrics to `AudioSamples` that measure per-object and per-batch **decode latency** (time to decode in-memory bytes to numpy arrays). This is still useful for understanding if CPU decoding is a bottleneck.

### Implementation
- `collation.py`: Added `_timed_read_audio()` wrapper and `object_latencies` parameter threaded through `collate_audio` → `read_audio_from_cuts`
- `input_strategies.py`: Added `collect_metrics`, `metrics_save_interval` to `AudioSamples.__init__()`. Added `get_metrics()`, `save_metrics()`, `reset_metrics()`, `_auto_save_metrics()`.
- Same metrics structure as AISObjectLoader: per_object (p50/p95/p99/avg), batch (p50/p95/p99/avg + per_object_avg), num_batches, total_objects.

## Bug Fix: `rich_exception_info` (on g-latency)

`lhotse/utils.py:rich_exception_info` tried to re-raise exceptions with `raise type(e)(message)`. This fails for exceptions like `BotoCoreError` whose `__init__` doesn't accept a string arg. Fixed to fall back to `RuntimeError` when re-construction fails.
