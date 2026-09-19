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
 * An inverted-file vector format optimized for low-latency approximate nearest-neighbor search.
 *
 * <p>At index time, IVFasterEvo rotates normalized vectors, clusters them around centroids, and
 * stores each vector in its primary cell plus a small number of spill cells. At query time it
 * follows the same rotation, chooses nearby cells, scans their compact coarse codes, and reranks a
 * bounded shortlist with the fine representation.
 *
 * <p>The coarse tier is Nitrox2, an extended Hamming code with a sign bit and a magnitude bit. Its
 * layout is designed around a high-CPU-memory-bandwidth XOR and popcount scan, allowing SIMD to
 * reject most candidates with very low latency. The remaining candidates use either INT8 or FP32
 * fine vectors for reranking.
 *
 * <p>The main implementation pieces are:
 *
 * <ul>
 *   <li>{@link org.apache.lucene.sandbox.codecs.ivfaster_evo.IVFasterEvoVectorsFormat} configures
 *       clustering, probing, spilling, and the fine tier.
 *   <li>{@code Clustering} builds cells and secondary spill assignments.
 *   <li>{@code Centroids} encodes centroids and builds the graph used to select probe cells.
 *   <li>{@code IVFasterEvoVectorsWriter} stages, clusters, and persists vectors.
 *   <li>{@code IVFasterEvoVectorsReader} performs coarse scans and fine reranking.
 *   <li>{@code HotStart} reuses centroid state across flushes and merges so clustering work is not
 *       discarded when the index is split into segments.
 * </ul>
 *
 * @lucene.experimental
 */
package org.apache.lucene.sandbox.codecs.ivfaster_evo;
