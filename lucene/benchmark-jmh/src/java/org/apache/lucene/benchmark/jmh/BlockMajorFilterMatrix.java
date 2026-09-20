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

import java.util.Arrays;
import org.apache.lucene.internal.vectorization.BitSetConjunctionSupport;
import org.apache.lucene.util.FixedBitSet;

/**
 * Block-major representation of materialized filters.
 *
 * <p>The sparsest filter is kept as a contiguous lead bit set. Remaining filters use the layout
 * {@code [doc vector block][filter][vector lane]}. Intersections of fixed-size filter groups are
 * materialized at construction time.
 */
final class BlockMajorFilterMatrix {

  private final long[] leadWords;
  private final long[] trailingWords;
  private final long[] groupWords;
  private final int vectorBlockCount;
  private final int trailingFilterCount;
  private final int groupCount;

  private BlockMajorFilterMatrix(
      long[] leadWords,
      long[] trailingWords,
      long[] groupWords,
      int vectorBlockCount,
      int trailingFilterCount,
      int groupCount) {
    this.leadWords = leadWords;
    this.trailingWords = trailingWords;
    this.groupWords = groupWords;
    this.vectorBlockCount = vectorBlockCount;
    this.trailingFilterCount = trailingFilterCount;
    this.groupCount = groupCount;
  }

  /**
   * Fused transpose from selectivity-ordered bit sets. Each source word is read once, copied to the
   * block-major matrix, and included in its precomputed group intersection while it is hot.
   */
  static BlockMajorFilterMatrix fromBitSets(
      FixedBitSet[] filters, int maxDoc, int laneCount, int groupSize) {
    return fromBitSets(filters, maxDoc, laneCount, groupSize, true);
  }

  /** Materializes only the lead filter and precomputed group intersections. */
  static BlockMajorFilterMatrix groupsOnlyFromBitSets(
      FixedBitSet[] filters,
      int maxDoc,
      int groupSize,
      BitSetConjunctionSupport conjunctionSupport) {
    int laneCount = conjunctionSupport.longLaneCount();
    validate(filters.length, laneCount, groupSize);
    int wordCount = (maxDoc + Long.SIZE - 1) >>> 6;
    int vectorBlockCount = (wordCount + laneCount - 1) / laneCount;
    int paddedWordCount = vectorBlockCount * laneCount;
    int trailingFilterCount = filters.length - 1;
    int groupCount = (trailingFilterCount + groupSize - 1) / groupSize;

    long[] leadWords = Arrays.copyOf(filters[0].getBits(), paddedWordCount);
    long[] groupWords = newLongArray((long) vectorBlockCount * groupCount * laneCount);
    long[][] filterWords = new long[filters.length][];
    Arrays.setAll(filterWords, filter -> filters[filter].getBits());
    conjunctionSupport.buildBlockMajorGroups(filterWords, groupWords, wordCount, groupSize);

    return new BlockMajorFilterMatrix(
        leadWords, new long[0], groupWords, vectorBlockCount, trailingFilterCount, groupCount);
  }

  private static BlockMajorFilterMatrix fromBitSets(
      FixedBitSet[] filters,
      int maxDoc,
      int laneCount,
      int groupSize,
      boolean retainTrailingWords) {
    validate(filters.length, laneCount, groupSize);
    int wordCount = (maxDoc + Long.SIZE - 1) >>> 6;
    int vectorBlockCount = (wordCount + laneCount - 1) / laneCount;
    int paddedWordCount = vectorBlockCount * laneCount;
    int trailingFilterCount = filters.length - 1;
    int groupCount = (trailingFilterCount + groupSize - 1) / groupSize;

    long[] leadWords = Arrays.copyOf(filters[0].getBits(), paddedWordCount);
    long[] trailingWords =
        retainTrailingWords
            ? newLongArray((long) vectorBlockCount * trailingFilterCount * laneCount)
            : new long[0];
    long[] groupWords = newLongArray((long) vectorBlockCount * groupCount * laneCount);
    long[] groupAccumulator = new long[laneCount];

    for (int block = 0; block < vectorBlockCount; ++block) {
      int wordOffset = block * laneCount;
      int trailingBlockBase = block * trailingFilterCount * laneCount;
      int groupBlockBase = block * groupCount * laneCount;
      for (int group = 0; group < groupCount; ++group) {
        Arrays.fill(groupAccumulator, -1L);
        int filterStart = 1 + group * groupSize;
        int filterEnd = Math.min(filterStart + groupSize, filters.length);
        for (int filter = filterStart; filter < filterEnd; ++filter) {
          long[] source = filters[filter].getBits();
          int destination = trailingBlockBase + (filter - 1) * laneCount;
          for (int lane = 0; lane < laneCount; ++lane) {
            int sourceIndex = wordOffset + lane;
            long word = sourceIndex < source.length ? source[sourceIndex] : 0L;
            if (retainTrailingWords) {
              trailingWords[destination + lane] = word;
            }
            groupAccumulator[lane] &= word;
          }
        }
        System.arraycopy(
            groupAccumulator, 0, groupWords, groupBlockBase + group * laneCount, laneCount);
      }
    }

    return new BlockMajorFilterMatrix(
        leadWords, trailingWords, groupWords, vectorBlockCount, trailingFilterCount, groupCount);
  }

  /**
   * Direct materialization from selectivity-ordered, sorted doc-ID lists. Construction walks output
   * blocks in order, consumes every input list monotonically, and never builds clause-major bit
   * sets.
   */
  static BlockMajorFilterMatrix fromSortedDocIds(
      int[][] filterDocIds, int maxDoc, int laneCount, int groupSize) {
    return fromSortedDocIds(filterDocIds, maxDoc, laneCount, groupSize, true);
  }

  /** Builds only the lead filter and precomputed groups directly from sorted doc-ID lists. */
  static BlockMajorFilterMatrix groupsOnlyFromSortedDocIds(
      int[][] filterDocIds, int maxDoc, int laneCount, int groupSize) {
    return fromSortedDocIds(filterDocIds, maxDoc, laneCount, groupSize, false);
  }

  private static BlockMajorFilterMatrix fromSortedDocIds(
      int[][] filterDocIds, int maxDoc, int laneCount, int groupSize, boolean retainTrailingWords) {
    validate(filterDocIds.length, laneCount, groupSize);
    int wordCount = (maxDoc + Long.SIZE - 1) >>> 6;
    int vectorBlockCount = (wordCount + laneCount - 1) / laneCount;
    int paddedWordCount = vectorBlockCount * laneCount;
    int trailingFilterCount = filterDocIds.length - 1;
    int groupCount = (trailingFilterCount + groupSize - 1) / groupSize;

    long[] leadWords = new long[paddedWordCount];
    long[] trailingWords =
        retainTrailingWords
            ? newLongArray((long) vectorBlockCount * trailingFilterCount * laneCount)
            : new long[0];
    long[] groupWords = newLongArray((long) vectorBlockCount * groupCount * laneCount);
    int[] cursors = new int[filterDocIds.length];
    long[] blockWords = new long[filterDocIds.length * laneCount];

    int docsPerBlock = laneCount * Long.SIZE;
    for (int block = 0; block < vectorBlockCount; ++block) {
      Arrays.fill(blockWords, 0L);
      int blockMin = block * docsPerBlock;
      int blockMax = Math.min(blockMin + docsPerBlock, maxDoc);

      for (int filter = 0; filter < filterDocIds.length; ++filter) {
        int[] docIds = filterDocIds[filter];
        int cursor = cursors[filter];
        int filterBase = filter * laneCount;
        while (cursor < docIds.length && docIds[cursor] < blockMax) {
          int relativeDoc = docIds[cursor] - blockMin;
          blockWords[filterBase + (relativeDoc >>> 6)] |= 1L << (relativeDoc & 63);
          ++cursor;
        }
        cursors[filter] = cursor;
      }

      System.arraycopy(blockWords, 0, leadWords, block * laneCount, laneCount);
      if (retainTrailingWords) {
        int trailingBlockBase = block * trailingFilterCount * laneCount;
        System.arraycopy(
            blockWords,
            laneCount,
            trailingWords,
            trailingBlockBase,
            trailingFilterCount * laneCount);
      }

      int groupBlockBase = block * groupCount * laneCount;
      for (int group = 0; group < groupCount; ++group) {
        int groupBase = groupBlockBase + group * laneCount;
        Arrays.fill(groupWords, groupBase, groupBase + laneCount, -1L);
        int filterStart = 1 + group * groupSize;
        int filterEnd = Math.min(filterStart + groupSize, filterDocIds.length);
        for (int filter = filterStart; filter < filterEnd; ++filter) {
          int filterBase = filter * laneCount;
          for (int lane = 0; lane < laneCount; ++lane) {
            groupWords[groupBase + lane] &= blockWords[filterBase + lane];
          }
        }
      }
    }

    return new BlockMajorFilterMatrix(
        leadWords, trailingWords, groupWords, vectorBlockCount, trailingFilterCount, groupCount);
  }

  private static void validate(int filterCount, int laneCount, int groupSize) {
    if (filterCount < 2) {
      throw new IllegalArgumentException("filterCount must be at least 2");
    }
    if (laneCount <= 0) {
      throw new IllegalArgumentException("laneCount must be positive");
    }
    if (groupSize < 4 || groupSize > 16 || (groupSize & 3) != 0) {
      throw new IllegalArgumentException("groupSize must be one of 4, 8, 12 or 16");
    }
  }

  private static long[] newLongArray(long length) {
    if (length > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("Array is too large: " + length + " longs");
    }
    return new long[(int) length];
  }

  long[] leadWords() {
    return leadWords;
  }

  long[] trailingWords() {
    return trailingWords;
  }

  long[] groupWords() {
    return groupWords;
  }

  int vectorBlockCount() {
    return vectorBlockCount;
  }

  int trailingFilterCount() {
    return trailingFilterCount;
  }

  int groupCount() {
    return groupCount;
  }
}
