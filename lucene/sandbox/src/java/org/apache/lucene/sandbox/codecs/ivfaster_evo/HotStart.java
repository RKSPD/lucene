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

package org.apache.lucene.sandbox.codecs.ivfaster_evo;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.store.Directory;

/**
 * Retains recent centroid state so later segments can warm-start clustering.
 *
 * <p>Flushes and merges create different physical parts of the same index. Reusing compatible
 * centroids, assignments, and distance bounds keeps that earlier clustering work from being
 * discarded and reduces repeated indexing cost.
 */
final class HotStart {
  private static final Map<Directory, Map<String, List<Seed>>> INDEXES = new WeakHashMap<>();

  /** Reusable clustering state associated with one segment lineage. */
  record Seed(
      String segment,
      String lineage,
      int vectors,
      float[][] centroids,
      int[] members,
      int[] assignment,
      int[] cell2,
      float[] d1,
      float[] d2) {}

  /** Builds the cache key for a vector field configuration. */
  private static String key(FieldInfo info) {
    return info.name + "/" + info.getVectorDimension() + "/" + info.getVectorSimilarityFunction();
  }

  /** Clears all cached hot-start state. */
  static synchronized void clear() {
    INDEXES.clear();
  }

  /** Publishes clustering state for reuse by later flushes or merges. */
  static synchronized void publish(
      Directory dir,
      String segment,
      String lineage,
      FieldInfo info,
      Clustering.Result result,
      int vectors) {
    var fields = INDEXES.computeIfAbsent(dir, _ -> new HashMap<>());
    List<Seed> seeds = fields.computeIfAbsent(key(info), _ -> new ArrayList<>());
    seeds.removeIf(seed -> seed.segment.equals(segment));
    seeds.add(
        new Seed(
            segment,
            lineage,
            vectors,
            result.centroids(),
            primaryMembers(result.assignment(), result.centroids().length),
            result.assignment(),
            result.cell2(),
            result.d1(),
            result.d2()));
    seeds.sort(Comparator.comparingInt(Seed::vectors).reversed());
  }

  /** Returns the cached state for a specific segment. */
  static synchronized Seed snapshot(Directory dir, String segment, FieldInfo info) {
    Map<String, List<Seed>> fields = INDEXES.get(dir);
    if (fields == null) return null;
    List<Seed> seeds = fields.get(key(info));
    if (seeds == null) return null;
    for (Seed seed : seeds) if (seed.segment.equals(segment)) return seed;
    return null;
  }

  /** Selects a compatible live segment as a clustering seed. */
  static synchronized Seed seed(Directory dir, String writing, FieldInfo info, int nlist, int count)
      throws IOException {
    if (INDEXES.containsKey(dir) == false) readCommit(dir);
    List<Seed> seeds = INDEXES.get(dir).get(key(info));
    if (seeds == null) return null;
    Set<String> live = new HashSet<>();
    for (String file : dir.listAll()) live.add(IndexFileNames.parseSegmentName(file));
    seeds.removeIf(s -> s.segment.equals(writing) || live.contains(s.segment) == false);
    Seed donor = null;
    for (Seed candidate : seeds) {
      int cells = candidate.centroids.length;
      if (cells <= nlist && cells >= Math.max(1, nlist / 2) && cells <= count) {
        donor = candidate;
        break;
      }
    }
    if (donor == null) return null;
    float[][][] centroids = new float[seeds.size()][][];
    int[][] members = new int[seeds.size()][];
    String[] lineages = new String[seeds.size()];
    int donorIndex = -1;
    for (int i = 0; i < seeds.size(); i++) {
      Seed candidate = seeds.get(i);
      centroids[i] = candidate.centroids;
      members[i] = candidate.members;
      lineages[i] = candidate.lineage;
      if (candidate == donor) donorIndex = i;
    }
    return new Seed(
        donor.segment,
        donor.lineage,
        donor.vectors,
        weightedCentroids(centroids, members, lineages, donorIndex),
        null,
        null,
        null,
        null,
        null);
  }

  /** Counts primary assignments for weighting corresponding centroids. */
  private static int[] primaryMembers(int[] assignment, int nlist) {
    int[] members = new int[nlist];
    if (assignment != null) {
      for (int cell : assignment) if (cell >= 0 && cell < nlist) members[cell]++;
    }
    return members;
  }

  /** Averages corresponding same-lineage centroids by their live primary-cell populations. */
  static float[][] weightedCentroids(
      float[][][] centroidSets, int[][] populations, String[] lineages, int donor) {
    float[][] from = centroidSets[donor];
    int nlist = from.length, dim = from[0].length;
    float[][] seed = new float[nlist][dim];
    long[] weights = new long[nlist];
    for (int r = 0; r < centroidSets.length; r++) {
      float[][] centroids = centroidSets[r];
      int[] members = populations[r];
      if (centroids == null
          || members == null
          || centroids.length != nlist
          || lineages[r] == null
          || lineages[r].equals(lineages[donor]) == false) continue;
      for (int c = 0; c < nlist; c++) {
        int weight = members[c];
        if (weight == 0) continue;
        weights[c] += weight;
        float[] source = centroids[c], target = seed[c];
        for (int d = 0; d < dim; d++) target[d] += weight * source[d];
      }
    }
    for (int c = 0; c < nlist; c++) {
      if (weights[c] == 0) {
        System.arraycopy(from[c], 0, seed[c], 0, dim);
      } else {
        normalize(seed[c]);
      }
    }
    return seed;
  }

  /** Normalizes one centroid in place. */
  private static void normalize(float[] vector) {
    double norm = 0;
    for (float value : vector) norm += (double) value * value;
    if (norm == 0) return;
    float scale = (float) (1.0 / Math.sqrt(norm));
    for (int d = 0; d < vector.length; d++) vector[d] *= scale;
  }

  /** Rebuilds the hot-start cache from the latest commit. */
  private static void readCommit(Directory dir) throws IOException {
    INDEXES.put(dir, new HashMap<>());
    if (DirectoryReader.indexExists(dir) == false) return;
    try (DirectoryReader reader = DirectoryReader.open(dir)) {
      for (LeafReaderContext leaf : reader.leaves()) {
        if (leaf.reader() instanceof SegmentReader sr && sr.getVectorReader() != null) {
          for (FieldInfo info : sr.getFieldInfos()) {
            if (sr.getVectorReader().unwrapReaderForField(info.name)
                    instanceof IVFasterEvoVectorsReader evo
                && evo.field(info.name) instanceof IVFasterEvoVectorsReader.Field view
                && view.rotationSeed
                    == IVFasterEvoVectorsWriter.rotationSeed(info.getVectorDimension())) {
              int[] assignment = new int[view.count];
              for (int ord = 0; ord < view.count; ord++) assignment[ord] = view.cellOf(ord);
              var empty =
                  new Clustering.Result(
                      view.centroids, new int[0], 0, assignment, null, null, null);
              String segment = sr.getSegmentInfo().info.name;
              publish(dir, segment, segment, info, empty, view.count);
            }
          }
        }
      }
    }
  }
}
