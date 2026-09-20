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

import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.apache.lucene.util.FixedBitSet;

/** Panama Vector API implementation of {@link BitSetConjunctionSupport}. */
final class PanamaBitSetConjunctionSupport implements BitSetConjunctionSupport {

  static final PanamaBitSetConjunctionSupport INSTANCE = new PanamaBitSetConjunctionSupport();

  private static final VectorSpecies<Long> LONG_SPECIES = LongVector.SPECIES_PREFERRED;

  private PanamaBitSetConjunctionSupport() {}

  @Override
  public int longLaneCount() {
    return LONG_SPECIES.length();
  }

  @Override
  public void buildBlockMajorGroups(
      long[][] filterWords, long[] groupWords, int wordCount, int groupSize) {
    int laneCount = LONG_SPECIES.length();
    int blockCount = (wordCount + laneCount - 1) / laneCount;
    int trailingFilterCount = filterWords.length - 1;
    int groupCount = (trailingFilterCount + groupSize - 1) / groupSize;
    for (int block = 0; block < blockCount; ++block) {
      int wordOffset = block * laneCount;
      VectorMask<Long> mask = LONG_SPECIES.indexInRange(wordOffset, wordCount);
      int groupBlockBase = block * groupCount * laneCount;
      for (int group = 0; group < groupCount; ++group) {
        LongVector intersection = LongVector.broadcast(LONG_SPECIES, -1L);
        int filterStart = 1 + group * groupSize;
        int filterEnd = Math.min(filterStart + groupSize, filterWords.length);
        for (int filter = filterStart; filter < filterEnd; ++filter) {
          intersection =
              intersection.and(
                  LongVector.fromArray(LONG_SPECIES, filterWords[filter], wordOffset, mask));
        }
        intersection.intoArray(groupWords, groupBlockBase + group * laneCount);
      }
    }
  }

  @Override
  public void andBitSets(
      FixedBitSet[] sources, int sourceCount, int sourceFrom, FixedBitSet dest, int length) {
    int sourceShift = sourceFrom & 63;
    int sourceWord = sourceFrom >>> 6;
    int fullWordCount = length >>> 6;
    int vectorizedWordCount = LONG_SPECIES.loopBound(fullWordCount);
    int laneCount = LONG_SPECIES.length();
    long[] destWords = dest.getBits();

    for (int destWord = 0; destWord < vectorizedWordCount; destWord += laneCount) {
      LongVector intersection = LongVector.fromArray(LONG_SPECIES, destWords, destWord);
      for (int i = 0; i < sourceCount; ++i) {
        long[] sourceWords = sources[i].getBits();
        int lowOffset = sourceWord + destWord;
        VectorMask<Long> lowMask = LONG_SPECIES.indexInRange(lowOffset, sourceWords.length);
        LongVector source = LongVector.fromArray(LONG_SPECIES, sourceWords, lowOffset, lowMask);
        if (sourceShift != 0) {
          int highOffset = lowOffset + 1;
          VectorMask<Long> highMask = LONG_SPECIES.indexInRange(highOffset, sourceWords.length);
          LongVector high = LongVector.fromArray(LONG_SPECIES, sourceWords, highOffset, highMask);
          source =
              source
                  .lanewise(VectorOperators.LSHR, sourceShift)
                  .or(high.lanewise(VectorOperators.LSHL, Long.SIZE - sourceShift));
        }
        intersection = intersection.and(source);
      }
      intersection.intoArray(destWords, destWord);
    }

    int processedBits = vectorizedWordCount << 6;
    int remainingBits = length - processedBits;
    for (int i = 0; i < sourceCount; ++i) {
      andBitSet(sources[i], sourceFrom + processedBits, dest, processedBits, remainingBits);
    }
  }

  @Override
  public long countBlockMajor(
      long[] leadWords,
      long[] trailingWords,
      int blockCount,
      int trailingFilterCount,
      int groupSize) {
    LongVector sum = LongVector.zero(LONG_SPECIES);
    int laneCount = LONG_SPECIES.length();
    for (int block = 0; block < blockCount; ++block) {
      int wordOffset = block * laneCount;
      LongVector intersection = LongVector.fromArray(LONG_SPECIES, leadWords, wordOffset);
      if (intersection.eq(0L).allTrue()) {
        continue;
      }
      int blockBase = block * trailingFilterCount * laneCount;
      for (int groupStart = 0; groupStart < trailingFilterCount; groupStart += groupSize) {
        int groupEnd = Math.min(groupStart + groupSize, trailingFilterCount);
        LongVector groupIntersection =
            intersectGroup(trailingWords, blockBase, laneCount, groupStart, groupEnd);
        intersection = intersection.and(groupIntersection);
        if (intersection.eq(0L).allTrue()) {
          break;
        }
      }
      sum = sum.add(intersection.lanewise(VectorOperators.BIT_COUNT));
    }
    return sum.reduceLanes(VectorOperators.ADD);
  }

  @Override
  public long countBlockMajorGroups(
      long[] leadWords, long[] groupWords, int blockCount, int groupCount) {
    LongVector sum = LongVector.zero(LONG_SPECIES);
    int laneCount = LONG_SPECIES.length();
    for (int block = 0; block < blockCount; ++block) {
      int wordOffset = block * laneCount;
      LongVector intersection = LongVector.fromArray(LONG_SPECIES, leadWords, wordOffset);
      if (intersection.eq(0L).allTrue()) {
        continue;
      }
      int blockBase = block * groupCount * laneCount;
      for (int group = 0; group < groupCount; ++group) {
        intersection =
            intersection.and(
                LongVector.fromArray(LONG_SPECIES, groupWords, blockBase + group * laneCount));
        if (intersection.eq(0L).allTrue()) {
          break;
        }
      }
      sum = sum.add(intersection.lanewise(VectorOperators.BIT_COUNT));
    }
    return sum.reduceLanes(VectorOperators.ADD);
  }

  @Override
  public long countBooleanMatrixVector(
      long[] documentFilterWords, int docCount, int paddedDocCount, long[] queryMasks) {
    long count = 0;
    int laneCount = LONG_SPECIES.length();
    int vectorizedDocCount = LONG_SPECIES.loopBound(docCount);
    for (int doc = 0; doc < vectorizedDocCount; doc += laneCount) {
      VectorMask<Long> matches = LONG_SPECIES.maskAll(true);
      for (int word = 0; word < queryMasks.length; ++word) {
        long queryWord = queryMasks[word];
        LongVector documentWords =
            LongVector.fromArray(LONG_SPECIES, documentFilterWords, word * paddedDocCount + doc);
        matches = matches.and(documentWords.and(queryWord).eq(queryWord));
        if (matches.anyTrue() == false) {
          break;
        }
      }
      count += matches.trueCount();
    }
    for (int doc = vectorizedDocCount; doc < docCount; ++doc) {
      boolean matches = true;
      for (int word = 0; word < queryMasks.length; ++word) {
        long queryWord = queryMasks[word];
        long documentWord = documentFilterWords[word * paddedDocCount + doc];
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

  private static LongVector intersectGroup(
      long[] words, int blockBase, int laneCount, int filterStart, int filterEnd) {
    if (filterEnd - filterStart < 4) {
      LongVector intersection =
          LongVector.fromArray(LONG_SPECIES, words, blockBase + filterStart * laneCount);
      for (int filter = filterStart + 1; filter < filterEnd; ++filter) {
        intersection =
            intersection.and(
                LongVector.fromArray(LONG_SPECIES, words, blockBase + filter * laneCount));
      }
      return intersection;
    }

    int filter = filterStart + 4;
    LongVector acc0 =
        LongVector.fromArray(LONG_SPECIES, words, blockBase + filterStart * laneCount);
    LongVector acc1 =
        LongVector.fromArray(LONG_SPECIES, words, blockBase + (filterStart + 1) * laneCount);
    LongVector acc2 =
        LongVector.fromArray(LONG_SPECIES, words, blockBase + (filterStart + 2) * laneCount);
    LongVector acc3 =
        LongVector.fromArray(LONG_SPECIES, words, blockBase + (filterStart + 3) * laneCount);
    for (; filter + 3 < filterEnd; filter += 4) {
      acc0 = acc0.and(LongVector.fromArray(LONG_SPECIES, words, blockBase + filter * laneCount));
      acc1 =
          acc1.and(LongVector.fromArray(LONG_SPECIES, words, blockBase + (filter + 1) * laneCount));
      acc2 =
          acc2.and(LongVector.fromArray(LONG_SPECIES, words, blockBase + (filter + 2) * laneCount));
      acc3 =
          acc3.and(LongVector.fromArray(LONG_SPECIES, words, blockBase + (filter + 3) * laneCount));
    }
    LongVector intersection = acc0.and(acc1).and(acc2.and(acc3));
    for (; filter < filterEnd; ++filter) {
      intersection =
          intersection.and(
              LongVector.fromArray(LONG_SPECIES, words, blockBase + filter * laneCount));
    }
    return intersection;
  }
}
