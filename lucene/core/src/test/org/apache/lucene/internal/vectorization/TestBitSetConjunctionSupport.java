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

import java.util.Arrays;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.FixedBitSet;

public class TestBitSetConjunctionSupport extends LuceneTestCase {

  public void testScalar() {
    assertCounts(DefaultBitSetConjunctionSupport.INSTANCE);
  }

  public void testSelectedProvider() {
    assertCounts(VectorizationProvider.lookup(true).getBitSetConjunctionSupport());
  }

  private void assertCounts(BitSetConjunctionSupport support) {
    int laneCount = support.longLaneCount();
    int blockCount = atLeast(1);
    int trailingFilterCount = atLeast(1);
    int groupSize = 4 * random().nextInt(1, 5);
    int groupCount = (trailingFilterCount + groupSize - 1) / groupSize;

    long[] leadWords = new long[blockCount * laneCount];
    long[] trailingWords = new long[blockCount * trailingFilterCount * laneCount];
    for (int i = 0; i < leadWords.length; ++i) {
      leadWords[i] = random().nextLong();
    }
    for (int i = 0; i < trailingWords.length; ++i) {
      trailingWords[i] = random().nextLong();
    }

    long expected =
        referenceCount(leadWords, trailingWords, blockCount, trailingFilterCount, laneCount);
    assertEquals(
        expected,
        support.countBlockMajor(
            leadWords, trailingWords, blockCount, trailingFilterCount, groupSize));

    long[][] filterWords = new long[trailingFilterCount + 1][blockCount * laneCount];
    System.arraycopy(leadWords, 0, filterWords[0], 0, leadWords.length);
    for (int block = 0; block < blockCount; ++block) {
      int trailingBlockBase = block * trailingFilterCount * laneCount;
      int wordBase = block * laneCount;
      for (int filter = 0; filter < trailingFilterCount; ++filter) {
        System.arraycopy(
            trailingWords,
            trailingBlockBase + filter * laneCount,
            filterWords[filter + 1],
            wordBase,
            laneCount);
      }
    }

    long[] expectedGroupWords =
        buildReferenceGroups(
            trailingWords, blockCount, trailingFilterCount, laneCount, groupSize, groupCount);
    long[] groupWords = new long[expectedGroupWords.length];
    support.buildBlockMajorGroups(filterWords, groupWords, blockCount * laneCount, groupSize);
    assertArrayEquals(expectedGroupWords, groupWords);
    assertEquals(
        expected, support.countBlockMajorGroups(leadWords, groupWords, blockCount, groupCount));

    assertBooleanMatrixVector(support);
    assertAndBitSets(support);
  }

  private void assertAndBitSets(BitSetConjunctionSupport support) {
    int sourceCount = random().nextInt(1, 9);
    int sourceFrom = random().nextInt(128);
    int length = random().nextInt(1, 5001);
    FixedBitSet[] sources = new FixedBitSet[sourceCount];
    for (int i = 0; i < sourceCount; ++i) {
      int sourceLength = sourceFrom + random().nextInt(1, length + 1);
      sources[i] = randomBitSet(sourceLength);
    }
    FixedBitSet expected = randomBitSet(length);
    FixedBitSet actual = expected.clone();
    for (FixedBitSet source : sources) {
      DefaultBitSetConjunctionSupport.INSTANCE.andBitSet(source, sourceFrom, expected, 0, length);
    }
    support.andBitSets(sources, sourceCount, sourceFrom, actual, length);
    assertEquals(expected, actual);
  }

  private FixedBitSet randomBitSet(int length) {
    FixedBitSet bitSet = new FixedBitSet(length);
    long[] words = bitSet.getBits();
    for (int i = 0; i < words.length; ++i) {
      words[i] = random().nextLong();
    }
    int ghostBits = (words.length << 6) - length;
    if (ghostBits > 0) {
      words[words.length - 1] &= -1L >>> ghostBits;
    }
    return bitSet;
  }

  private void assertBooleanMatrixVector(BitSetConjunctionSupport support) {
    int laneCount = support.longLaneCount();
    int docCount = atLeast(1);
    int paddedDocCount = (docCount + laneCount - 1) / laneCount * laneCount;
    int queryWordCount = random().nextInt(1, 6);
    long[] queryMasks = new long[queryWordCount];
    long[] documentFilterWords = new long[queryWordCount * paddedDocCount];
    for (int word = 0; word < queryWordCount; ++word) {
      queryMasks[word] = random().nextLong() | 1L;
      for (int doc = 0; doc < docCount; ++doc) {
        documentFilterWords[word * paddedDocCount + doc] = random().nextLong();
      }
    }

    long expected = 0;
    for (int doc = 0; doc < docCount; ++doc) {
      boolean matches = true;
      for (int word = 0; word < queryWordCount; ++word) {
        long queryWord = queryMasks[word];
        long documentWord = documentFilterWords[word * paddedDocCount + doc];
        if ((documentWord & queryWord) != queryWord) {
          matches = false;
          break;
        }
      }
      if (matches) {
        ++expected;
      }
    }
    assertEquals(
        expected,
        support.countBooleanMatrixVector(
            documentFilterWords, docCount, paddedDocCount, queryMasks));
  }

  private static long[] buildReferenceGroups(
      long[] trailingWords,
      int blockCount,
      int trailingFilterCount,
      int laneCount,
      int groupSize,
      int groupCount) {
    long[] groupWords = new long[blockCount * groupCount * laneCount];
    for (int block = 0; block < blockCount; ++block) {
      int trailingBlockBase = block * trailingFilterCount * laneCount;
      int groupBlockBase = block * groupCount * laneCount;
      for (int group = 0; group < groupCount; ++group) {
        int groupBase = groupBlockBase + group * laneCount;
        Arrays.fill(groupWords, groupBase, groupBase + laneCount, -1L);
        int filterStart = group * groupSize;
        int filterEnd = Math.min(filterStart + groupSize, trailingFilterCount);
        for (int filter = filterStart; filter < filterEnd; ++filter) {
          int filterBase = trailingBlockBase + filter * laneCount;
          for (int lane = 0; lane < laneCount; ++lane) {
            groupWords[groupBase + lane] &= trailingWords[filterBase + lane];
          }
        }
      }
    }
    return groupWords;
  }

  private static long referenceCount(
      long[] leadWords,
      long[] trailingWords,
      int blockCount,
      int trailingFilterCount,
      int laneCount) {
    long count = 0;
    for (int block = 0; block < blockCount; ++block) {
      int wordBase = block * laneCount;
      int trailingBlockBase = block * trailingFilterCount * laneCount;
      for (int lane = 0; lane < laneCount; ++lane) {
        long intersection = leadWords[wordBase + lane];
        for (int filter = 0; filter < trailingFilterCount; ++filter) {
          intersection &= trailingWords[trailingBlockBase + filter * laneCount + lane];
        }
        count += Long.bitCount(intersection);
      }
    }
    return count;
  }
}
