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
package org.apache.lucene.benchmark.jmh;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.apache.lucene.internal.vectorization.BitSetConjunctionSupport;
import org.apache.lucene.internal.vectorization.VectorizationProvider;
import org.apache.lucene.search.BulkScorer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.DocIdStream;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.TwoPhaseIterator;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Experiments with layouts and execution strategies for conjunctions of many materialized filters.
 *
 * <p>The sparse lead remains in a contiguous clause-major array so that empty vector blocks can be
 * skipped without touching the much larger filter matrix. Trailing filters are additionally
 * transposed into block-major order:
 *
 * <pre>
 * [doc vector block][filter][vector lane]
 * </pre>
 *
 * <p>Grouped methods use four independent accumulators and check for an empty intersection after
 * every {@link #groupSize} filters. The precomputed method models a cache of common filter groups;
 * its setup cost is deliberately excluded from query evaluation.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(
    value = 1,
    jvmArgsPrepend = {
      "--add-modules=jdk.incubator.vector",
      "--add-opens=org.apache.lucene.core/org.apache.lucene.search=org.apache.lucene.benchmark.jmh"
    },
    jvmArgsAppend = {"-Xmx3g", "-Xms3g", "-XX:+AlwaysPreTouch"})
public class BlockMajorFilterConjunctionBenchmark {

  private static final VectorSpecies<Long> LONG_SPECIES = LongVector.SPECIES_PREFERRED;
  private static final BitSetConjunctionSupport BIT_SET_CONJUNCTION_SUPPORT =
      VectorizationProvider.getInstance().getBitSetConjunctionSupport();

  @Param({"4194304"})
  public int maxDoc;

  @Param({"100", "500", "1000"})
  public int filterCount;

  @Param({"0.001", "0.005"})
  public double leadDensity;

  @Param({"independent", "nested"})
  public String pattern;

  @Param({"4", "8", "16"})
  public int groupSize;

  private long[][] clauseWords;
  private FixedBitSet[] orderedFilters;
  private long[] leadWords;
  private long[] blockMajorTrailingWords;
  private long[] precomputedGroupWords;
  private int wordCount;
  private int laneCount;
  private int vectorBlockCount;
  private int trailingFilterCount;
  private int groupCount;
  private Constructor<? extends BulkScorer> denseConjunctionConstructor;
  private BulkScorer denseConjunctionBulkScorer;
  private final CountingLeafCollector countingCollector = new CountingLeafCollector();

  @Setup(Level.Trial)
  public void setup() {
    if (filterCount < 2) {
      throw new IllegalArgumentException("filterCount must be at least 2");
    }
    if (groupSize < 4 || groupSize > 16 || (groupSize & 3) != 0) {
      throw new IllegalArgumentException("groupSize must be one of 4, 8, 12 or 16");
    }

    FixedBitSet[] filters = new FixedBitSet[filterCount];
    filters[0] = uniformlyDistributedSet(maxDoc, leadDensity);
    for (int i = 1; i < filterCount; ++i) {
      filters[i] = randomHalfDenseSet(maxDoc, 0x9E3779B97F4A7C15L * i);
      if ("nested".equals(pattern)) {
        filters[i].or(filters[0]);
      } else if ("independent".equals(pattern) == false) {
        throw new IllegalArgumentException("Unknown pattern: " + pattern);
      }
    }

    Integer[] order = new Integer[filterCount];
    Arrays.setAll(order, i -> i);
    Arrays.sort(order, Comparator.comparingInt(i -> filters[i].cardinality()));

    clauseWords = new long[filterCount][];
    orderedFilters = new FixedBitSet[filterCount];
    for (int i = 0; i < filterCount; ++i) {
      orderedFilters[i] = filters[order[i]];
      clauseWords[i] = orderedFilters[i].getBits();
    }
    wordCount = clauseWords[0].length;
    laneCount = BIT_SET_CONJUNCTION_SUPPORT.longLaneCount();
    if (laneCount != LONG_SPECIES.length()) {
      throw new IllegalStateException(
          "The benchmark requires the Panama provider: provider lanes="
              + laneCount
              + " preferred species lanes="
              + LONG_SPECIES.length());
    }
    if (wordCount % laneCount != 0) {
      throw new IllegalArgumentException(
          "maxDoc must produce a whole number of preferred-species vectors: wordCount="
              + wordCount
              + " lanes="
              + laneCount);
    }

    BlockMajorFilterMatrix matrix =
        BlockMajorFilterMatrix.fromBitSets(orderedFilters, maxDoc, laneCount, groupSize);
    leadWords = matrix.leadWords();
    blockMajorTrailingWords = matrix.trailingWords();
    precomputedGroupWords = matrix.groupWords();
    vectorBlockCount = matrix.vectorBlockCount();
    trailingFilterCount = matrix.trailingFilterCount();
    groupCount = matrix.groupCount();

    long expected = countLeadDriven();
    assertSame("clauseMajorShortCircuit", expected, countClauseMajorShortCircuit());
    assertSame("clauseMajorGrouped", expected, countClauseMajorGrouped());
    assertSame("blockMajorSerial", expected, countBlockMajorSerial());
    assertSame("blockMajorGrouped", expected, countBlockMajorGrouped());
    assertSame("blockMajorStreaming4", expected, countBlockMajorStreaming4());
    assertSame("blockMajorStreaming8", expected, countBlockMajorStreaming8());
    assertSame("blockMajorAdaptiveStreaming8", expected, countBlockMajorAdaptiveStreaming8());
    assertSame("precomputedGroups", expected, countPrecomputedGroups());

    try {
      Class<?> scorerClass = Class.forName("org.apache.lucene.search.DenseConjunctionBulkScorer");
      @SuppressWarnings("unchecked")
      Constructor<? extends BulkScorer> constructor =
          (Constructor<? extends BulkScorer>)
              scorerClass.getDeclaredConstructor(List.class, List.class, int.class, float.class);
      constructor.setAccessible(true);
      denseConjunctionConstructor = constructor;
      BulkScorer scorer = newDenseConjunctionBulkScorer();
      countingCollector.count = 0;
      scorer.score(countingCollector, null, 0, maxDoc);
      assertSame("denseConjunctionBulkScorer", expected, countingCollector.count);
    } catch (ReflectiveOperationException e) {
      throw new AssertionError("Unable to construct DenseConjunctionBulkScorer", e);
    } catch (IOException e) {
      throw new AssertionError("Unable to verify DenseConjunctionBulkScorer", e);
    }
  }

  @Setup(Level.Invocation)
  public void setupInvocation() {
    try {
      denseConjunctionBulkScorer = newDenseConjunctionBulkScorer();
      countingCollector.count = 0;
    } catch (ReflectiveOperationException e) {
      throw new AssertionError("Unable to construct DenseConjunctionBulkScorer", e);
    }
  }

  private BulkScorer newDenseConjunctionBulkScorer()
      throws InstantiationException, IllegalAccessException, InvocationTargetException {
    List<DocIdSetIterator> iterators = new ArrayList<>(filterCount);
    for (FixedBitSet filter : orderedFilters) {
      iterators.add(new BitSetIterator(filter, filter.cardinality()));
    }
    return denseConjunctionConstructor.newInstance(
        iterators, List.<TwoPhaseIterator>of(), maxDoc, 0f);
  }

  /** Document-at-a-time evaluation driven by the sparsest filter. */
  @Benchmark
  public long leadDriven() {
    return countLeadDriven();
  }

  private long countLeadDriven() {
    long count = 0;
    for (int wordIndex = 0; wordIndex < wordCount; ++wordIndex) {
      long candidates = leadWords[wordIndex];
      advanceCandidate:
      while (candidates != 0) {
        int bit = Long.numberOfTrailingZeros(candidates);
        long mask = 1L << bit;
        for (int filter = 1; filter < filterCount; ++filter) {
          if ((clauseWords[filter][wordIndex] & mask) == 0) {
            candidates &= candidates - 1;
            continue advanceCandidate;
          }
        }
        ++count;
        candidates &= candidates - 1;
      }
    }
    return count;
  }

  private static void assertSame(String method, long expected, long actual) {
    if (expected != actual) {
      throw new AssertionError(method + " counted " + actual + " matches, expected " + expected);
    }
  }

  private static FixedBitSet uniformlyDistributedSet(int maxDoc, double density) {
    if (density <= 0 || density > 1) {
      throw new IllegalArgumentException("density must be in (0, 1], got " + density);
    }
    FixedBitSet set = new FixedBitSet(maxDoc);
    int cardinality = Math.max(1, (int) Math.round(maxDoc * density));
    for (long i = 0; i < cardinality; ++i) {
      set.set((int) (i * maxDoc / cardinality));
    }
    return set;
  }

  private static FixedBitSet randomHalfDenseSet(int maxDoc, long seed) {
    FixedBitSet set = new FixedBitSet(maxDoc);
    SplittableRandom random = new SplittableRandom(seed);
    long[] bits = set.getBits();
    for (int i = 0; i < bits.length; ++i) {
      bits[i] = random.nextLong();
    }
    int ghostBits = (bits.length << 6) - maxDoc;
    if (ghostBits > 0) {
      bits[bits.length - 1] &= -1L >>> ghostBits;
    }
    return set;
  }

  /** Lucene's existing 4096-doc-window dense conjunction implementation. */
  @Benchmark
  public long denseConjunctionBulkScorer() throws IOException {
    denseConjunctionBulkScorer.score(countingCollector, null, 0, maxDoc);
    return countingCollector.count;
  }

  /** Existing clause-major shape with a zero check after every filter. */
  @Benchmark
  public long clauseMajorShortCircuit() {
    return countClauseMajorShortCircuit();
  }

  private long countClauseMajorShortCircuit() {
    LongVector sum = LongVector.zero(LONG_SPECIES);
    for (int block = 0; block < vectorBlockCount; ++block) {
      int wordOffset = block * laneCount;
      LongVector intersection = LongVector.fromArray(LONG_SPECIES, leadWords, wordOffset);
      if (intersection.eq(0L).allTrue()) {
        continue;
      }
      for (int filter = 1; filter < filterCount; ++filter) {
        intersection =
            intersection.and(LongVector.fromArray(LONG_SPECIES, clauseWords[filter], wordOffset));
        if (intersection.eq(0L).allTrue()) {
          break;
        }
      }
      sum = sum.add(intersection.lanewise(VectorOperators.BIT_COUNT));
    }
    return sum.reduceLanes(VectorOperators.ADD);
  }

  /** Clause-major loads with four independent accumulators and one zero check per group. */
  @Benchmark
  public long clauseMajorGrouped() {
    return countClauseMajorGrouped();
  }

  private long countClauseMajorGrouped() {
    LongVector sum = LongVector.zero(LONG_SPECIES);
    for (int block = 0; block < vectorBlockCount; ++block) {
      int wordOffset = block * laneCount;
      LongVector intersection = LongVector.fromArray(LONG_SPECIES, leadWords, wordOffset);
      if (intersection.eq(0L).allTrue()) {
        continue;
      }
      for (int groupStart = 1; groupStart < filterCount; groupStart += groupSize) {
        int groupEnd = Math.min(groupStart + groupSize, filterCount);
        LongVector groupIntersection = clauseMajorGroup(wordOffset, groupStart, groupEnd);
        intersection = intersection.and(groupIntersection);
        if (intersection.eq(0L).allTrue()) {
          break;
        }
      }
      sum = sum.add(intersection.lanewise(VectorOperators.BIT_COUNT));
    }
    return sum.reduceLanes(VectorOperators.ADD);
  }

  private LongVector clauseMajorGroup(int wordOffset, int filterStart, int filterEnd) {
    if (filterEnd - filterStart < 4) {
      LongVector intersection =
          LongVector.fromArray(LONG_SPECIES, clauseWords[filterStart], wordOffset);
      for (int filter = filterStart + 1; filter < filterEnd; ++filter) {
        intersection =
            intersection.and(LongVector.fromArray(LONG_SPECIES, clauseWords[filter], wordOffset));
      }
      return intersection;
    }
    int filter = filterStart + 4;
    LongVector acc0 = LongVector.fromArray(LONG_SPECIES, clauseWords[filterStart], wordOffset);
    LongVector acc1 = LongVector.fromArray(LONG_SPECIES, clauseWords[filterStart + 1], wordOffset);
    LongVector acc2 = LongVector.fromArray(LONG_SPECIES, clauseWords[filterStart + 2], wordOffset);
    LongVector acc3 = LongVector.fromArray(LONG_SPECIES, clauseWords[filterStart + 3], wordOffset);
    for (; filter + 3 < filterEnd; filter += 4) {
      acc0 = acc0.and(LongVector.fromArray(LONG_SPECIES, clauseWords[filter], wordOffset));
      acc1 = acc1.and(LongVector.fromArray(LONG_SPECIES, clauseWords[filter + 1], wordOffset));
      acc2 = acc2.and(LongVector.fromArray(LONG_SPECIES, clauseWords[filter + 2], wordOffset));
      acc3 = acc3.and(LongVector.fromArray(LONG_SPECIES, clauseWords[filter + 3], wordOffset));
    }
    LongVector intersection = acc0.and(acc1).and(acc2.and(acc3));
    for (; filter < filterEnd; ++filter) {
      intersection =
          intersection.and(LongVector.fromArray(LONG_SPECIES, clauseWords[filter], wordOffset));
    }
    return intersection;
  }

  /** Block-major loads with a zero check after every filter. */
  @Benchmark
  public long blockMajorSerial() {
    return countBlockMajorSerial();
  }

  private long countBlockMajorSerial() {
    LongVector sum = LongVector.zero(LONG_SPECIES);
    for (int block = 0; block < vectorBlockCount; ++block) {
      int wordOffset = block * laneCount;
      LongVector intersection = LongVector.fromArray(LONG_SPECIES, leadWords, wordOffset);
      if (intersection.eq(0L).allTrue()) {
        continue;
      }
      int blockBase = block * trailingFilterCount * laneCount;
      for (int filter = 0; filter < trailingFilterCount; ++filter) {
        intersection =
            intersection.and(
                LongVector.fromArray(
                    LONG_SPECIES, blockMajorTrailingWords, blockBase + filter * laneCount));
        if (intersection.eq(0L).allTrue()) {
          break;
        }
      }
      sum = sum.add(intersection.lanewise(VectorOperators.BIT_COUNT));
    }
    return sum.reduceLanes(VectorOperators.ADD);
  }

  /** Block-major loads with four independent accumulators and one zero check per group. */
  @Benchmark
  public long blockMajorGrouped() {
    return countBlockMajorGrouped();
  }

  private long countBlockMajorGrouped() {
    return BIT_SET_CONJUNCTION_SUPPORT.countBlockMajor(
        leadWords, blockMajorTrailingWords, vectorBlockCount, trailingFilterCount, groupSize);
  }

  /** Branch-free full scan with eight independent AND accumulators and one final reduction. */
  @Benchmark
  public long blockMajorStreaming8() {
    return countBlockMajorStreaming8();
  }

  private long countBlockMajorStreaming8() {
    LongVector sum = LongVector.zero(LONG_SPECIES);
    for (int block = 0; block < vectorBlockCount; ++block) {
      int wordOffset = block * laneCount;
      int blockBase = block * trailingFilterCount * laneCount;
      LongVector intersection =
          LongVector.fromArray(LONG_SPECIES, leadWords, wordOffset)
              .and(intersectBlockMajor8(blockBase, 0, trailingFilterCount));
      sum = sum.add(intersection.lanewise(VectorOperators.BIT_COUNT));
    }
    return sum.reduceLanes(VectorOperators.ADD);
  }

  /** Branch-free full scan with four independent AND accumulators and one final reduction. */
  @Benchmark
  public long blockMajorStreaming4() {
    return countBlockMajorStreaming4();
  }

  private long countBlockMajorStreaming4() {
    LongVector sum = LongVector.zero(LONG_SPECIES);
    for (int block = 0; block < vectorBlockCount; ++block) {
      int wordOffset = block * laneCount;
      int blockBase = block * trailingFilterCount * laneCount;
      LongVector intersection =
          LongVector.fromArray(LONG_SPECIES, leadWords, wordOffset)
              .and(intersectBlockMajor4(blockBase, 0, trailingFilterCount));
      sum = sum.add(intersection.lanewise(VectorOperators.BIT_COUNT));
    }
    return sum.reduceLanes(VectorOperators.ADD);
  }

  /** Maximum sequential read throughput over the block-major matrix with four load streams. */
  @Benchmark
  public long rawSequentialRead4() {
    LongVector acc0 = LongVector.zero(LONG_SPECIES);
    LongVector acc1 = LongVector.zero(LONG_SPECIES);
    LongVector acc2 = LongVector.zero(LONG_SPECIES);
    LongVector acc3 = LongVector.zero(LONG_SPECIES);
    int stride = 4 * laneCount;
    int offset = 0;
    int limit = blockMajorTrailingWords.length - stride + 1;
    for (; offset < limit; offset += stride) {
      acc0 = xor(acc0, LongVector.fromArray(LONG_SPECIES, blockMajorTrailingWords, offset));
      acc1 =
          xor(
              acc1,
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + laneCount));
      acc2 =
          xor(
              acc2,
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + 2 * laneCount));
      acc3 =
          xor(
              acc3,
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + 3 * laneCount));
    }
    LongVector checksum = xor(xor(acc0, acc1), xor(acc2, acc3));
    for (; offset < blockMajorTrailingWords.length; offset += laneCount) {
      checksum =
          xor(
              checksum,
              LongVector.fromArray(LONG_SPECIES, blockMajorTrailingWords, offset));
    }
    return checksum.reduceLanes(VectorOperators.XOR);
  }

  /** Maximum sequential read throughput over the block-major matrix with eight load streams. */
  @Benchmark
  public long rawSequentialRead8() {
    LongVector acc0 = LongVector.zero(LONG_SPECIES);
    LongVector acc1 = LongVector.zero(LONG_SPECIES);
    LongVector acc2 = LongVector.zero(LONG_SPECIES);
    LongVector acc3 = LongVector.zero(LONG_SPECIES);
    LongVector acc4 = LongVector.zero(LONG_SPECIES);
    LongVector acc5 = LongVector.zero(LONG_SPECIES);
    LongVector acc6 = LongVector.zero(LONG_SPECIES);
    LongVector acc7 = LongVector.zero(LONG_SPECIES);
    int stride = 8 * laneCount;
    int offset = 0;
    int limit = blockMajorTrailingWords.length - stride + 1;
    for (; offset < limit; offset += stride) {
      acc0 = xor(acc0, LongVector.fromArray(LONG_SPECIES, blockMajorTrailingWords, offset));
      acc1 =
          xor(
              acc1,
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + laneCount));
      acc2 =
          xor(
              acc2,
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + 2 * laneCount));
      acc3 =
          xor(
              acc3,
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + 3 * laneCount));
      acc4 =
          xor(
              acc4,
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + 4 * laneCount));
      acc5 =
          xor(
              acc5,
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + 5 * laneCount));
      acc6 =
          xor(
              acc6,
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + 6 * laneCount));
      acc7 =
          xor(
              acc7,
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + 7 * laneCount));
    }
    LongVector checksum =
        xor(xor(acc0, acc1), xor(acc2, acc3));
    checksum = xor(checksum, xor(xor(acc4, acc5), xor(acc6, acc7)));
    for (; offset < blockMajorTrailingWords.length; offset += laneCount) {
      checksum =
          xor(
              checksum,
              LongVector.fromArray(LONG_SPECIES, blockMajorTrailingWords, offset));
    }
    return checksum.reduceLanes(VectorOperators.XOR);
  }

  private static LongVector xor(LongVector left, LongVector right) {
    return left.lanewise(VectorOperators.XOR, right);
  }

  /**
   * Checks the first 16 selectivity-ordered filters, then switches surviving blocks to a
   * branch-free eight-accumulator scan.
   */
  @Benchmark
  public long blockMajorAdaptiveStreaming8() {
    return countBlockMajorAdaptiveStreaming8();
  }

  private long countBlockMajorAdaptiveStreaming8() {
    LongVector sum = LongVector.zero(LONG_SPECIES);
    int firstGroupEnd = Math.min(16, trailingFilterCount);
    for (int block = 0; block < vectorBlockCount; ++block) {
      int wordOffset = block * laneCount;
      LongVector intersection = LongVector.fromArray(LONG_SPECIES, leadWords, wordOffset);
      if (intersection.eq(0L).allTrue()) {
        continue;
      }

      int blockBase = block * trailingFilterCount * laneCount;
      intersection = intersection.and(intersectBlockMajor8(blockBase, 0, firstGroupEnd));
      if (intersection.eq(0L).allTrue()) {
        continue;
      }

      if (firstGroupEnd < trailingFilterCount) {
        intersection =
            intersection.and(
                intersectBlockMajor8(blockBase, firstGroupEnd, trailingFilterCount));
      }
      sum = sum.add(intersection.lanewise(VectorOperators.BIT_COUNT));
    }
    return sum.reduceLanes(VectorOperators.ADD);
  }

  private LongVector intersectBlockMajor8(int blockBase, int filterStart, int filterEnd) {
    LongVector acc0 = LongVector.broadcast(LONG_SPECIES, -1L);
    LongVector acc1 = LongVector.broadcast(LONG_SPECIES, -1L);
    LongVector acc2 = LongVector.broadcast(LONG_SPECIES, -1L);
    LongVector acc3 = LongVector.broadcast(LONG_SPECIES, -1L);
    LongVector acc4 = LongVector.broadcast(LONG_SPECIES, -1L);
    LongVector acc5 = LongVector.broadcast(LONG_SPECIES, -1L);
    LongVector acc6 = LongVector.broadcast(LONG_SPECIES, -1L);
    LongVector acc7 = LongVector.broadcast(LONG_SPECIES, -1L);

    int filter = filterStart;
    for (; filter + 7 < filterEnd; filter += 8) {
      int offset = blockBase + filter * laneCount;
      acc0 =
          acc0.and(LongVector.fromArray(LONG_SPECIES, blockMajorTrailingWords, offset));
      acc1 =
          acc1.and(
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + laneCount));
      acc2 =
          acc2.and(
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + 2 * laneCount));
      acc3 =
          acc3.and(
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + 3 * laneCount));
      acc4 =
          acc4.and(
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + 4 * laneCount));
      acc5 =
          acc5.and(
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + 5 * laneCount));
      acc6 =
          acc6.and(
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + 6 * laneCount));
      acc7 =
          acc7.and(
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + 7 * laneCount));
    }

    LongVector intersection =
        acc0.and(acc1)
            .and(acc2.and(acc3))
            .and(acc4.and(acc5).and(acc6.and(acc7)));
    for (; filter < filterEnd; ++filter) {
      intersection =
          intersection.and(
              LongVector.fromArray(
                  LONG_SPECIES,
                  blockMajorTrailingWords,
                  blockBase + filter * laneCount));
    }
    return intersection;
  }

  private LongVector intersectBlockMajor4(int blockBase, int filterStart, int filterEnd) {
    LongVector acc0 = LongVector.broadcast(LONG_SPECIES, -1L);
    LongVector acc1 = LongVector.broadcast(LONG_SPECIES, -1L);
    LongVector acc2 = LongVector.broadcast(LONG_SPECIES, -1L);
    LongVector acc3 = LongVector.broadcast(LONG_SPECIES, -1L);

    int filter = filterStart;
    for (; filter + 3 < filterEnd; filter += 4) {
      int offset = blockBase + filter * laneCount;
      acc0 =
          acc0.and(LongVector.fromArray(LONG_SPECIES, blockMajorTrailingWords, offset));
      acc1 =
          acc1.and(
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + laneCount));
      acc2 =
          acc2.and(
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + 2 * laneCount));
      acc3 =
          acc3.and(
              LongVector.fromArray(
                  LONG_SPECIES, blockMajorTrailingWords, offset + 3 * laneCount));
    }

    LongVector intersection = acc0.and(acc1).and(acc2.and(acc3));
    for (; filter < filterEnd; ++filter) {
      intersection =
          intersection.and(
              LongVector.fromArray(
                  LONG_SPECIES,
                  blockMajorTrailingWords,
                  blockBase + filter * laneCount));
    }
    return intersection;
  }

  /**
   * Query-time evaluation when common groups have already been intersected and cached as bit sets.
   */
  @Benchmark
  public long precomputedGroups() {
    return countPrecomputedGroups();
  }

  private long countPrecomputedGroups() {
    return BIT_SET_CONJUNCTION_SUPPORT.countBlockMajorGroups(
        leadWords, precomputedGroupWords, vectorBlockCount, groupCount);
  }

  private static final class CountingLeafCollector implements LeafCollector {
    long count;

    @Override
    public void setScorer(Scorable scorer) {}

    @Override
    public void collect(int doc) {
      ++count;
    }

    @Override
    public void collect(DocIdStream stream) throws IOException {
      count += stream.count();
    }

    @Override
    public void collectRange(int min, int max) {
      count += max - min;
    }
  }
}
