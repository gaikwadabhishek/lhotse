Subject: Latency measurement challenge for sequential tar loading mode

Hi Pewter,

I wanted to get your advice on a problem I ran into while implementing latency benchmarking across our three AIS loading modes.

Quick recap of the three modes:

1. Batch GET (AISBatchLoader) — one batch request for all objects in a CutSet. Audio bytes replace the cut's recording sources before collate_audio runs.
2. Random access GET (AISObjectLoader) — individual GET requests per object. Same as above — we fetch bytes, replace recording sources, then collate_audio decodes from memory.
3. Sequential tarred (LazyNeMoTarredIterator._iter_sequential) — opens tar in streaming mode, reads objects sequentially, attaches bytes to cuts as they're yielded.

Latency measurement for modes 1 and 2 is straightforward. The I/O happens at a clean boundary inside AudioSamples.__call__() — we make network calls, get the bytes, replace the content in the CutSet, and then decode. We can time per-object fetches and the overall batch easily because the I/O and the batch boundary are at the same level.

Mode 3 (sequential) is the problem. The I/O doesn't happen at batch time — it happens lazily during iteration, driven by the sampler pulling cuts one at a time:

```
_iter_sequential() opens tar once, yields cuts one-by-one
    → each yield reads the next object from the tar stream
    → sampler accumulates cuts until batch is full
    → dataset.__getitem__(batch) gets cuts with audio already in memory
    → AudioSamples.__call__() only decodes from memory (no I/O)
```

The tar stream is a single open connection. Each .read() just pulls the next N bytes from an already-established stream. The "batch" boundary is defined by the sampler, not by the tar reader — so I/O for a single batch is spread across multiple sampler pulls interleaved with sampling logic. There's no clean place to measure "how long did it take to load this batch from storage."

If I instrument _iter_sequential() per-object, I'm mostly measuring object_size / stream_throughput which will be fairly uniform and not that useful. And I can't measure "per-batch I/O" because the batch doesn't exist at that level.

What I'm looking for advice on:
- Is there a meaningful latency metric we should capture for the sequential mode, or should we focus benchmarking on modes 1 and 2 where the measurement is clean?
- Would shard-level throughput (total time to read an entire tar shard / total bytes) be more useful than per-object timing?
- Should we instead measure end-to-end batch latency (from sampler start to tensor ready) which captures I/O + sampling + decode together?

Let me know what you think.
