# IVFasterEvo vector-width benchmarks

Benchmarked on September 21, 2026 using the Cohere Wikipedia corpus with
1024-dimensional `float32` vectors and dot-product similarity.

## Search settings

- `topK=100`, `fanout=100`, 1,000 queries
- `nlist=sqrt(vector count)`: 1,000 / 3,162 / 6,306
- `spill=1`, `spillMargin=1.05`
- `nprobe=32`, `nprobeMargin=0.75`
- `bruteN=700`, `verifyMultiplier=2`
- Nitrox2 coarse tier and INT8 fine tier
- One force-merged segment
- 128-bit runs use `-XX:MaxVectorSize=16`
- 256-bit runs use the platform-preferred 256-bit species

## Search results

Latency is milliseconds per query. Each cell is `recall / latency`.

| Vectors | IVFasterEvo 128-bit | IVFaster 128-bit | IVFasterEvo 256-bit | IVFaster 256-bit |
|---:|---:|---:|---:|---:|
| 1M | 0.940 / **1.125 ms** | 0.948 / 1.809 ms | 0.940 / **0.761 ms** | 0.948 / 0.924 ms |
| 10M | 0.947 / **3.061 ms** | 0.948 / 4.771 ms | 0.947 / **1.997 ms** | 0.948 / 2.221 ms |
| 39.77M | 0.942 / **6.001 ms** | 0.939 / 8.953 ms | 0.942 / **4.082 ms** | 0.939 / 4.251 ms |

## Relative latency

| Vectors | Evo versus IVFaster, 128-bit | Evo versus IVFaster, 256-bit | Evo 128-bit slowdown versus 256-bit |
|---:|---:|---:|---:|
| 1M | 37.8% faster | 17.6% faster | 1.48x |
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
| 1M | IVFasterEvo | 18.12 s | 5.91 s | 14.17 s | 38.20 s |
| 1M | IVFaster | 80.70 s | 22.28 s | 42.27 s | 145.25 s |
| 10M | IVFasterEvo | 231.76 s | 27.80 s | 169.53 s | 429.09 s |
| 10M | IVFaster | 250.20 s | 0.00 s | 188.72 s | 438.92 s |
| 39.77M | IVFasterEvo | 2,054.73 s | 16.19 s | 1,227.50 s | 3,298.42 s |
| 39.77M | IVFaster | 2,244.08 s | 0.00 s | 1,291.74 s | 3,535.82 s |

## Implementation change

The INT8 rerank kernel remains specialized for the tuned 256-bit shape. At
other preferred widths, Evo now follows IVFaster's strategy: copy each strided
fine code into contiguous scratch space and call Lucene core's signed-byte dot
product, allowing core to select its native-width implementation.

The coarse Hamming scan also includes IVFaster's four-vector shape for a
256-byte Nitrox2 row under 512-bit species.

