/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * IVFasterEvo, a two-tier inverted-file (IVF) vector format for low-latency approximate
 * nearest-neighbor search.
 *
 * <p>Vectors are normalized and randomly rotated with a Hadamard transform, clustered into cells,
 * and written to their primary cell plus up to {@code spillBits} spill cells near cell boundaries.
 * Each stored slot carries two encodings: a Nitrox2 coarse code (two bits per dimension, a sign bit
 * and a magnitude bit, compared by XOR and popcount Hamming distance) and a fine INT8 or FP32
 * vector. A persisted slot-to-document section maps scan results back to doc IDs.
 *
 * <p>At query time the target is rotated and encoded the same way. A navigation graph over the
 * centroid codes picks candidate cells, which are re-ranked by exact centroid distance and trimmed
 * to the probe count. The coarse codes of the probed cells are scanned with SIMD Hamming kernels,
 * and each segment keeps a deduplicated shortlist of its nearest survivors. Search through {@link
 * org.apache.lucene.sandbox.codecs.ivfaster_evo.IVFasterEvoKnnQuery}: it merges every segment's
 * shortlist into one index-wide shortlist of roughly 700 candidates and reranks only those with the
 * fine tier, so fine reads per query do not grow with the segment count. A plain {@code
 * KnnFloatVectorQuery} still works but reranks each segment's shortlist separately. Filters are
 * applied before any scoring: each segment either scans the probed cells under the filter or ranks
 * the accepted documents directly, whichever is cheaper, and the result joins the same global
 * shortlist.
 *
 * <p>Coarse codes are read through the mapped index and are meant to stay in the page cache. For
 * indexes whose fine tier is larger than RAM, {@code -Divfaster.evo.uringFine=true} reads fine
 * records with batched io_uring {@code O_DIRECT} reads instead (Linux, v2 indexes, non-compound
 * segments on a file-system directory), so reranks never evict the cached coarse codes. It is off
 * by default: when the index fits in RAM, mapped reads are much faster.
 *
 * <p>To limit clustering cost, flushed segments seed clustering from the centroids of earlier
 * segments ("hot start"). Compatible merges copy encoded rows and warm-start from a donor segment,
 * with centroids averaged across same-lineage segments weighted by their live cell populations.
 *
 * <p>Main components:
 *
 * <ul>
 *   <li>{@link org.apache.lucene.sandbox.codecs.ivfaster_evo.IVFasterEvoVectorsFormat}: format
 *       configuration and per-query {@code SearchStrategy}.
 *   <li>{@code IVFasterEvoVectorsWriter}: stages, clusters, encodes, and persists vectors.
 *   <li>{@code IVFasterEvoVectorsReader}: cell selection, coarse scan, and fine rerank.
 *   <li>{@code Clustering}: iterative spherical clustering with spill assignment.
 *   <li>{@code Centroids}: centroid codes, routing, and the centroid navigation graph.
 *   <li>{@code Tiers}: the Nitrox2 and fine encodings and the Hadamard rotation.
 *   <li>{@code Kernels}: scalar kernels with Panama Vector API replacements.
 *   <li>{@code HotStart}: centroid state retained across flushes and merges.
 * </ul>
 *
 * @lucene.experimental
 */
package org.apache.lucene.sandbox.codecs.ivfaster_evo;
