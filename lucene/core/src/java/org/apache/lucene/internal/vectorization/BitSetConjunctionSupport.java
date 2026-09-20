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
package org.apache.lucene.internal.vectorization;

import org.apache.lucene.util.FixedBitSet;

/** SIMD-accelerated conjunction operations over materialized filter bit sets. */
public interface BitSetConjunctionSupport {

  /** Number of 64-bit words processed together. */
  int longLaneCount();

  /**
   * Intersects selectivity-ordered clause-major filters into block-major groups. Filter zero is the
   * lead and is not included in the output.
   */
  void buildBlockMajorGroups(long[][] filterWords, long[] groupWords, int wordCount, int groupSize);

  /**
   * Intersects {@code sourceCount} bit sets into the first {@code length} bits of {@code dest},
   * reading source bits starting at {@code sourceFrom}.
   */
  void andBitSets(
      FixedBitSet[] sources, int sourceCount, int sourceFrom, FixedBitSet dest, int length);

  /** Intersects one source bit set into a destination range. */
  default void andBitSet(
      FixedBitSet source, int sourceFrom, FixedBitSet dest, int destFrom, int length) {
    int coveredLength = Math.max(0, Math.min(length, source.length() - sourceFrom));
    if (coveredLength > 0) {
      FixedBitSet.andRange(source, sourceFrom, dest, destFrom, coveredLength);
    }
    if (coveredLength < length) {
      dest.clear(destFrom + coveredLength, destFrom + length);
    }
  }

  /**
   * Counts the intersection of a lead filter and block-major trailing filters, checking for an
   * empty result after every group.
   */
  long countBlockMajor(
      long[] leadWords,
      long[] trailingWords,
      int blockCount,
      int trailingFilterCount,
      int groupSize);

  /** Counts the intersection of a lead filter and precomputed block-major filter groups. */
  long countBlockMajorGroups(long[] leadWords, long[] groupWords, int blockCount, int groupCount);

  /**
   * Counts matches in a filter-word-major document matrix.
   *
   * <p>{@code documentFilterWords} uses the layout {@code [query filter word][document]}. Each
   * query word represents up to 64 filters, and a document matches when every bit in every {@code
   * queryMasks} word is present in its corresponding document word.
   */
  long countBooleanMatrixVector(
      long[] documentFilterWords, int docCount, int paddedDocCount, long[] queryMasks);
}
