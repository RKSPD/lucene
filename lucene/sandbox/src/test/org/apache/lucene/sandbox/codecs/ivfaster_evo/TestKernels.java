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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Tests the dimension-independent SIMD kernels. */
public class TestKernels extends LuceneTestCase {
  /** Compares batch Hamming scans with the scalar implementation at every supported shape. */
  public void testHammingMultiplesOf128() {
    Kernels scalar = new Kernels();
    Kernels vector = Kernels.INSTANCE;
    for (int dim = 128; dim <= 2048; dim += 128) {
      int codeBytes = Tiers.Nitrox2.bytesPerVector(dim);
      byte[] query = new byte[codeBytes];
      random().nextBytes(query);
      for (int rows = 1; rows <= 9; rows++) {
        int offset = 13;
        byte[] codes = new byte[offset + rows * codeBytes];
        random().nextBytes(codes);
        int[] expected = new int[rows];
        int[] actual = new int[rows];
        scalar.hamming(query, codes, offset, rows, expected);
        vector.hamming(query, codes, offset, rows, actual);
        assertArrayEquals("dim=" + dim + " rows=" + rows, expected, actual);

        try (Arena arena = Arena.ofConfined()) {
          MemorySegment nativeCodes = arena.allocate(codes.length);
          for (int i = 0; i < codes.length; i++) {
            nativeCodes.set(ValueLayout.JAVA_BYTE, i, codes[i]);
          }
          Arrays.fill(actual, 0);
          vector.hamming(query, nativeCodes, offset, rows, actual);
          assertArrayEquals("native dim=" + dim + " rows=" + rows, expected, actual);
        }
      }
    }
  }
}
