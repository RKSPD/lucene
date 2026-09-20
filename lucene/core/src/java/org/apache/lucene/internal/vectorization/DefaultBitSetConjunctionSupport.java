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

/** Scalar implementation of {@link BitSetConjunctionSupport}. */
final class DefaultBitSetConjunctionSupport implements BitSetConjunctionSupport {

  static final DefaultBitSetConjunctionSupport INSTANCE = new DefaultBitSetConjunctionSupport();

  private DefaultBitSetConjunctionSupport() {}

  @Override
  public int longLaneCount() {
    return 1;
  }

  @Override
  public void buildBlockMajorGroups(
      long[][] filterWords, long[] groupWords, int wordCount, int groupSize) {
    int trailingFilterCount = filterWords.length - 1;
    int groupCount = (trailingFilterCount + groupSize - 1) / groupSize;
    for (int word = 0; word < wordCount; ++word) {
      int groupBlockBase = word * groupCount;
      for (int group = 0; group < groupCount; ++group) {
        long intersection = -1L;
        int filterStart = 1 + group * groupSize;
        int filterEnd = Math.min(filterStart + groupSize, filterWords.length);
        for (int filter = filterStart; filter < filterEnd; ++filter) {
          intersection &= filterWords[filter][word];
        }
        groupWords[groupBlockBase + group] = intersection;
      }
    }
  }

  @Override
  public void andBitSets(
      FixedBitSet[] sources, int sourceCount, int sourceFrom, FixedBitSet dest, int length) {
    for (int i = 0; i < sourceCount; ++i) {
      andBitSet(sources[i], sourceFrom, dest, 0, length);
    }
  }

  @Override
  public long countBlockMajor(
      long[] leadWords,
      long[] trailingWords,
      int blockCount,
      int trailingFilterCount,
      int groupSize) {
    long count = 0;
    for (int block = 0; block < blockCount; ++block) {
      long intersection = leadWords[block];
      if (intersection == 0) {
        continue;
      }
      int blockBase = block * trailingFilterCount;
      for (int groupStart = 0;
          groupStart < trailingFilterCount && intersection != 0;
          groupStart += groupSize) {
        int groupEnd = Math.min(groupStart + groupSize, trailingFilterCount);
        long groupIntersection = -1L;
        for (int filter = groupStart; filter < groupEnd; ++filter) {
          groupIntersection &= trailingWords[blockBase + filter];
        }
        intersection &= groupIntersection;
      }
      count += Long.bitCount(intersection);
    }
    return count;
  }

  @Override
  public long countBlockMajorGroups(
      long[] leadWords, long[] groupWords, int blockCount, int groupCount) {
    long count = 0;
    for (int block = 0; block < blockCount; ++block) {
      long intersection = leadWords[block];
      int blockBase = block * groupCount;
      for (int group = 0; group < groupCount && intersection != 0; ++group) {
        intersection &= groupWords[blockBase + group];
      }
      count += Long.bitCount(intersection);
    }
    return count;
  }

  @Override
  public long countBooleanMatrixVector(
      long[] documentFilterWords, int docCount, int paddedDocCount, long[] queryMasks) {
    long count = 0;
    for (int doc = 0; doc < docCount; ++doc) {
      boolean matches = true;
      for (int word = 0; word < queryMasks.length; ++word) {
        long documentWord = documentFilterWords[word * paddedDocCount + doc];
        long queryWord = queryMasks[word];
        if ((documentWord & queryWord) != queryWord) {
          matches = false;
          break;
        }
      }
      if (matches) {
        ++count;
      }
    }
    return count;
  }
}
