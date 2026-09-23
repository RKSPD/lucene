# IVFasterEvo vector-width benchmarks

Benchmarked on September 21, 2026 using the Cohere Wikipedia corpus with
1024-dimensional `float32` vectors and dot-product similarity.

## Search settings

- `topK=100`, `fanout=100`, 1,000 queries
- `nlist=sqrt(vector count)`: 1,000 / 3,162 / 6,306
- `spill=1`, `spillMargin=1.05`
- `nprobe=32`, `nprobeMargin=0.75`
- The refreshed 1M 256-bit Evo result uses `nprobe=45` to match IVFaster's
  `0.949` recall
- `bruteN=700`, `verifyMultiplier=2`
- Nitrox2 coarse tier and INT8 fine tier
- One force-merged segment
- 128-bit runs use `-XX:MaxVectorSize=16`
- 256-bit runs use the platform-preferred 256-bit species

## Search results

Latency is milliseconds per query. Each cell is `recall / latency`.

| Vectors | IVFasterEvo 128-bit | IVFaster 128-bit | IVFasterEvo 256-bit | IVFaster 256-bit |
|---:|---:|---:|---:|---:|
| 1M | 0.940 / **1.125 ms** | 0.949 / 1.790 ms | 0.949 / **0.874 ms** | 0.949 / 0.888 ms |
| 10M | 0.947 / **3.061 ms** | 0.948 / 4.771 ms | 0.947 / **1.997 ms** | 0.948 / 2.221 ms |
| 39.77M | 0.942 / **6.001 ms** | 0.939 / 8.953 ms | 0.942 / **4.082 ms** | 0.939 / 4.251 ms |

## Relative latency

| Vectors | Evo versus IVFaster, 128-bit | Evo versus IVFaster, 256-bit | Evo 128-bit slowdown versus 256-bit |
|---:|---:|---:|---:|
| 1M | 37.2% faster | 1.6% faster | N/A (different `nprobe`) |
| 10M | 35.8% faster | 10.1% faster | 1.53x |
| 39.77M | 33.0% faster | 4.0% faster | 1.47x |

Before the width fix, IVFasterEvo measured 12.792 ms at 10M and 15.728 ms at
39.77M with 128-bit species. The width-aware fine-rerank fallback reduced those
latencies to 3.061 ms and 6.001 ms respectively.

## Indexing results

The 10M and 39.77M indexes were built with 8 indexing threads, 24 merge workers,
8 merge threads, and force merge enabled.

| Vectors | Codec | Ingest | Merge wait | Force merge | Total |
|---:|---|---:|---:|---:|---:|
| 1M | IVFasterEvo | 20.94 s | 0.00 s | 15.01 s | 35.95 s |
| 1M | IVFaster | 18.70 s | 0.00 s | 15.05 s | 33.75 s |
| 10M | IVFasterEvo | 231.76 s | 27.80 s | 169.53 s | 429.09 s |
| 10M | IVFaster | 250.20 s | 0.00 s | 188.72 s | 438.92 s |
| 39.77M | IVFasterEvo | 2,054.73 s | 16.19 s | 1,227.50 s | 3,298.42 s |
| 39.77M | IVFaster | 2,244.08 s | 0.00 s | 1,291.74 s | 3,535.82 s |

## HNSW SQ7 baseline

This box also has a 1M-vector Lucene HNSW SQ7 run from August 12, 2026.
Luceneutil reports the quantized representation as `8 bits` because each SQ7
value occupies one byte. The index used `maxConn=16`, `beamWidthIndex=100`,
`topK=100`, and 1,000 queries.

| Vectors | Fanout | Recall | Latency |
|---:|---:|---:|---:|
| 1M | 100 | 0.933 | 1.426 ms |
| 1M | 150 | 0.946 | 1.703 ms |
| 1M | 200 | 0.954 | 1.987 ms |

The SQ7 HNSW index took 131.40 s to ingest, 95.89 s waiting for merges, and
555.44 s to force merge, for 782.73 s total. Its final index size was
4,957.28 MB.

Only the 1M SQ7 HNSW scale was available in the benchmark logs on this box; no
comparable 10M or 39.77M SQ7 HNSW run was recorded.

## Implementation change

The INT8 rerank kernel remains specialized for the tuned 256-bit shape. At
other preferred widths, Evo now follows IVFaster's strategy: copy each strided
fine code into contiguous scratch space and call Lucene core's signed-byte dot
product, allowing core to select its native-width implementation.

The coarse Hamming scan also includes IVFaster's four-vector shape for a
256-byte Nitrox2 row under 512-bit species.

## Dimension scaling

Benchmarked on September 23, 2026 with 1M vectors, 200 queries, 256-bit
preferred species, `nlist=1000`, `nprobe=45`, `topK=100`, `fanout=100`,
`bruteN=700`, and INT8 fine codes. The 256- and 768-dimensional corpora are
prefix truncations of the 1024-dimensional Cohere corpus, so recall values are
only comparable within each dimension; latency is the cross-dimension result
of interest.

| Dimensions | Recall | Latency | Indexing | Force merge |
|---:|---:|---:|---:|---:|
| 256 | 0.802 | 0.835 ms | 10.39 s | 6.11 s |
| 768 | 0.935 | 1.035 ms | 14.93 s | 11.52 s |
| 1024 | 0.945 | 1.075 ms | 20.51 s | 14.78 s |

The coarse scan uses a four-row, fixed-register-pressure loop for arbitrary
multiples of 128 dimensions. Exact four- and eight-preferred-vector shapes
retain their fully unrolled query-register paths. This removes the generic
per-row fallback for dimensions such as 768 without regressing the tuned
1024-dimensional shape.
