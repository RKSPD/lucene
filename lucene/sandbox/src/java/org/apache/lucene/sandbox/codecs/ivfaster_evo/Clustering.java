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
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.Centroids.CentroidCodes;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.IVFasterEvoVectorsWriter.StagedVectors;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.util.ArrayUtil;

/**
 * Builds IVF cells and spill assignments with iterative spherical clustering.
 *
 * <p>Movement bounds avoid rerouting vectors whose nearest-cell decision cannot have changed, and
 * spill cells preserve recall for vectors near cell boundaries.
 */
final class Clustering {
  static final long SEED = 42L;

  static final int OVERSAMPLE = 3, MIN_SHORTLIST = 32;

  static final float MARGIN = Float.parseFloat(System.getProperty("ivfaster.spillMargin", "1.40"));

  static final float REAP_MARGIN = 1.02f;

  static final float SOAR_LAMBDA = 1f;

  static final float CONVERGE_FRACTION = 0.003f;

  static final long FIX = 1L << 30;

  /** Prevents construction of the clustering utility class. */
  private Clustering() {}

  /** Prior assignments and distances that let a merge resume rather than restart clustering. */
  record WarmState(int[] assignment, int[] cell2, float[] d1, float[] d2) {}

  /** Final centroids and primary or spill cell assignments. */
  record Result(
      float[][] centroids,
      int[] cells,
      int stride,
      int[] assignment,
      int[] cell2,
      float[] d1,
      float[] d2) {
    /** Counts the cells assigned to one vector. */
    int cellCount(int i) {
      int n = 0;
      while (n < stride && cells[i * stride + n] >= 0) n++;
      return n;
    }

    /** Returns one cell assigned to a vector. */
    int cell(int i, int k) {
      return cells[i * stride + k];
    }
  }

  /** Clusters staged vectors and computes their primary and spill cells. */
  static Result cluster(StagedVectors src, int nlist, float[][] seed, WarmState warm, int spillBits)
      throws IOException {
    Run run = new Run(src, nlist, seed, spillBits);
    run.routeAll(warm);
    int count = src.count, convergeAt = (int) (CONVERGE_FRACTION * count);
    do {
      run.updateCentroids();
    } while (run.reap(false) > convergeAt);
    run.reap(true);
    for (int i = 0; i < count; i++) {
      if (run.cells[i * run.stride] < 0) run.cells[i * run.stride] = run.assignment[i];
    }
    return new Result(
        run.centroids, run.cells, run.stride, run.assignment, run.cell2, run.d1, run.d2);
  }

  private static final class Run {
    final StagedVectors src;
    final int count, dim, nlist, spillBits, stride;
    final float[][] centroids;
    final CentroidCodes codes;
    final int[] assignment, cell2, cells, members;
    final float[] d1, d2, movement;
    final float[] docSlack, docMaxSlack;
    final long[] sums;
    final Object[] locks;
    final ThreadLocal<float[]> spillResidual, spillLoss;

    /** Initializes mutable state for one clustering run. */
    Run(StagedVectors src, int nlist, float[][] seed, int spillBits) throws IOException {
      this.src = src;
      this.nlist = nlist;
      this.spillBits = spillBits;
      count = src.count;
      dim = src.dim;
      stride = 1 + spillBits;
      centroids = seed != null ? copyOf(seed) : sampleCentroids();
      if (centroids.length != nlist) {
        throw new IllegalArgumentException(
            "seed has " + centroids.length + " centroids but nlist is " + nlist);
      }
      for (float[] c : centroids) normalize(c);
      codes = new CentroidCodes(centroids, dim, null);
      assignment = new int[count];
      cell2 = new int[count];
      d1 = new float[count];
      d2 = new float[count];
      docSlack = new float[count];
      docMaxSlack = new float[count];
      movement = new float[nlist];
      members = new int[nlist];
      sums = new long[Math.multiplyExact(nlist, dim)];
      cells = new int[Math.multiplyExact(count, stride)];
      Arrays.fill(cells, -1);
      spillResidual = ThreadLocal.withInitial(() -> new float[dim]);
      spillLoss = ThreadLocal.withInitial(() -> new float[spillBits]);
      locks = new Object[Math.min(1024, Integer.highestOneBit(Math.max(1, nlist)) << 1)];
      for (int i = 0; i < locks.length; i++) locks[i] = new Object();
    }

    /** Copies seed centroids to the active dimension. */
    float[][] copyOf(float[][] seed) {
      float[][] out = new float[seed.length][];
      for (int i = 0; i < seed.length; i++) out[i] = ArrayUtil.copyOfSubArray(seed[i], 0, dim);
      return out;
    }

    /** Selects deterministic reservoir samples as initial centroids. */
    float[][] sampleCentroids() throws IOException {
      int k = Math.min(nlist, count);
      Random random = new Random(SEED);
      int[] pick = new int[k];
      for (int i = 0; i < k; i++) pick[i] = i;
      for (int i = k; i < count; i++) {
        int j = random.nextInt(i + 1);
        if (j < k) pick[j] = i;
      }
      float[][] out = new float[nlist][dim];
      StagedVectors.Cursor cur = src.cursor();
      for (int c = 0; c < nlist && k > 0; c++) {
        cur.load(pick[c % k]);
        System.arraycopy(cur.vector(), 0, out[c], 0, dim);
      }
      return out;
    }

    /** Adds or removes a vector from a centroid's fixed-point sum. */
    void apply(int cell, float[] v, int sign) {
      int base = cell * dim;
      synchronized (locks[cell & (locks.length - 1)]) {
        for (int d = 0; d < dim; d++) sums[base + d] += sign * Math.round((double) v[d] * FIX);
        members[cell] += sign;
      }
    }

    /** Chooses the coarse shortlist size for a requested result count. */
    int shortlistFor(int keep) {
      return Math.min(nlist, Math.max(OVERSAMPLE * keep, MIN_SHORTLIST));
    }

    /** Assigns every vector to initial or warm-start centroids. */
    void routeAll(WarmState warm) throws IOException {
      int shortlist = shortlistFor(2);
      Parallel.overRange(
          count,
          (lo, hi) -> {
            CentroidCodes.Scratch scratch = new CentroidCodes.Scratch(dim, nlist, shortlist);
            CentroidCodes.Routing routing = new CentroidCodes.Routing(2);
            StagedVectors.Cursor cur = src.cursor();
            for (int i = lo; i < hi; i++) {
              cur.load(i);
              float[] vector = cur.vector();
              int carried = warm == null ? -1 : warm.assignment[i];
              if (carried >= 0 && carried < nlist) {
                assignment[i] = carried;
                cell2[i] = warm.cell2[i];
                if (Float.isNaN(warm.d1[i])) {
                  d1[i] = codes.exactDistance(vector, carried);
                  d2[i] =
                      cell2[i] >= 0 && cell2[i] < nlist
                          ? codes.exactDistance(vector, cell2[i])
                          : Float.MAX_VALUE;
                } else {
                  d1[i] = warm.d1[i];
                  d2[i] = warm.d2[i];
                }
              } else {
                cur.coarseInto(scratch.qCode);
                codes.routePacked(vector, shortlist, 2, routing, scratch);
                assignment[i] = routing.count > 0 ? routing.cells[0] : 0;
                d1[i] = routing.d1;
                d2[i] = routing.d2;
                cell2[i] = routing.cell2;
              }
              apply(assignment[i], vector, 1);
            }
          });
    }

    /** Recomputes normalized centroids and movement bounds. */
    void updateCentroids() throws IOException {
      Parallel.overRange(
          nlist,
          (from, to) -> {
            float[] prev = new float[dim];
            for (int c = from; c < to; c++) {
              movement[c] = 0f;
              if (members[c] == 0) continue;
              float[] cent = centroids[c];
              System.arraycopy(cent, 0, prev, 0, dim);
              double inv = 1.0 / ((double) members[c] * FIX), moved = 0;
              for (int d = 0; d < dim; d++) cent[d] = (float) (sums[c * dim + d] * inv);
              normalize(cent);
              for (int d = 0; d < dim; d++) {
                double delta = (double) cent[d] - prev[d];
                moved += delta * delta;
              }
              movement[c] = (float) Math.sqrt(moved);
            }
          });
      codes.encodeAll();
      float maxMove = 0f;
      for (float m : movement) if (m > maxMove) maxMove = m;
      for (int i = 0; i < count; i++) {
        docSlack[i] += cell2[i] >= 0 ? movement[assignment[i]] + movement[cell2[i]] : 2f * maxMove;
        docMaxSlack[i] += maxMove;
      }
    }

    /** Rechecks vectors whose assignment may have changed. */
    int reap(boolean last) throws IOException {
      boolean withSpill = last && spillBits > 0;
      int keep = withSpill ? 1 + spillBits : 2, shortlist = shortlistFor(keep);
      AtomicInteger changed = new AtomicInteger();
      Parallel.overRange(
          count,
          (lo, hi) -> {
            CentroidCodes.Scratch scratch = new CentroidCodes.Scratch(dim, nlist, shortlist);
            CentroidCodes.Routing routing = new CentroidCodes.Routing(keep);
            int[] cands = new int[1 + keep];
            int localChanged = 0;
            StagedVectors.Cursor cur = src.cursor();
            for (int i = lo; i < hi; i++) {
              float gap = d2[i] - d1[i], wide = docMaxSlack[i] * (2f + MARGIN);
              if (gap > docSlack[i]
                  && CentroidCodes.withinMargin(d1[i], d2[i], REAP_MARGIN) == false
                  && (withSpill == false || gap > (MARGIN - 1f) * Math.abs(d1[i]) + wide)) {
                continue;
              }
              cur.load(i);
              float[] vector = cur.vector();
              cur.coarseInto(scratch.qCode);
              codes.routePacked(vector, shortlist, keep, routing, scratch);
              int incumbent = assignment[i], n = routing.count;
              float incumbentDist = codes.exactDistance(vector, incumbent);
              int[] spillCands = routing.cells;
              if (n > 0 && routing.d1 <= incumbentDist) {
                if (routing.cells[0] != incumbent) {
                  localChanged++;
                  apply(incumbent, vector, -1);
                  apply(routing.cells[0], vector, 1);
                }
                assignment[i] = routing.cells[0];
                d1[i] = routing.d1;
                d2[i] = routing.d2;
                cell2[i] = routing.cell2;
              } else {
                d1[i] = incumbentDist;
                d2[i] = n > 0 ? routing.d1 : Float.MAX_VALUE;
                cell2[i] = n > 0 ? routing.cells[0] : -1;
                cands[0] = incumbent;
                System.arraycopy(routing.cells, 0, cands, 1, n);
                spillCands = cands;
                n++;
              }
              docSlack[i] = docMaxSlack[i] = 0f;
              if (withSpill) spill(vector, spillCands, n, d1[i], d2[i], i * stride);
              else if (last) cells[i * stride] = assignment[i];
            }
            changed.addAndGet(localChanged);
          });
      return changed.get();
    }

    /** Selects secondary spill cells using residual loss. */
    void spill(float[] vector, int[] cands, int nCand, float dist1, float dist2, int base) {
      Arrays.fill(cells, base, base + stride, -1);
      int primary = cells[base] = nCand > 0 ? cands[0] : 0;
      if (nCand <= 1 || CentroidCodes.withinMargin(dist1, dist2, MARGIN) == false) return;
      float[] r1 = spillResidual.get(), pc = centroids[primary];
      double r1NormSq = 0;
      for (int d = 0; d < dim; d++) {
        r1[d] = vector[d] - pc[d];
        r1NormSq += (double) r1[d] * r1[d];
      }
      if (r1NormSq == 0) {
        for (int i = 1; i < nCand && i <= spillBits; i++) cells[base + i] = cands[i];
        return;
      }
      double invR1NormSq = 1.0 / r1NormSq;
      float[] bestLoss = spillLoss.get();
      int filled = 0;
      for (int ci = 1; ci < nCand; ci++) {
        int c = cands[ci];
        if (c == primary || c < 0) continue;
        float[] cc = centroids[c];
        double resNormSq = 0, dotR1 = 0;
        for (int d = 0; d < dim; d++) {
          double e = vector[d] - cc[d];
          resNormSq += e * e;
          dotR1 += e * r1[d];
        }
        float loss = (float) (resNormSq + SOAR_LAMBDA * dotR1 * dotR1 * invR1NormSq);
        if (filled == spillBits && loss >= bestLoss[spillBits - 1]) continue;
        int pos = filled < spillBits ? filled++ : spillBits - 1;
        for (; pos > 0 && bestLoss[pos - 1] > loss; pos--) {
          bestLoss[pos] = bestLoss[pos - 1];
          cells[base + 1 + pos] = cells[base + pos];
        }
        bestLoss[pos] = loss;
        cells[base + 1 + pos] = c;
      }
    }
  }

  /** Normalizes a vector in place when it has nonzero length. */
  private static void normalize(float[] v) {
    double norm = 0;
    for (float x : v) norm += (double) x * x;
    if (norm == 0) return;
    float inv = (float) (1.0 / Math.sqrt(norm));
    for (int d = 0; d < v.length; d++) v[d] *= inv;
  }

  static final class Parallel {
    private static final int MIN_PER_THREAD = 4096;

    static final int WORKERS =
        Math.max(
            1,
            Integer.getInteger(
                "ivfaster.buildThreads", Runtime.getRuntime().availableProcessors()));

    private static final AtomicInteger THREAD_ID = new AtomicInteger();

    private static final TaskExecutor EXEC =
        new TaskExecutor(
            Executors.newFixedThreadPool(
                WORKERS,
                r -> {
                  Thread t = new Thread(r, "ivfaster-build-" + THREAD_ID.getAndIncrement());
                  t.setDaemon(true);
                  return t;
                }));

    /** Prevents construction of the parallel execution utility. */
    private Parallel() {}

    interface RangeTask {
      /** Processes the half-open range from {@code from} to {@code to}. */
      void run(int from, int to) throws IOException;
    }

    /** Processes a range using the default parallel grain size. */
    static void overRange(int count, RangeTask body) throws IOException {
      overRange(count, MIN_PER_THREAD, body);
    }

    /** Splits a range across the configured build workers. */
    static void overRange(int count, int minPerThread, RangeTask body) throws IOException {
      int tasks = Math.min(WORKERS, Math.max(1, count / minPerThread));
      if (tasks <= 1) {
        body.run(0, count);
        return;
      }
      int chunk = (count + tasks - 1) / tasks;
      List<Callable<Void>> work = new ArrayList<>(tasks);
      for (int t = 0; t < tasks; t++) {
        int from = t * chunk, to = Math.min(count, from + chunk);
        if (from >= to) continue;
        work.add(
            () -> {
              body.run(from, to);
              return null;
            });
      }
      EXEC.invokeAll(work);
    }
  }
}
