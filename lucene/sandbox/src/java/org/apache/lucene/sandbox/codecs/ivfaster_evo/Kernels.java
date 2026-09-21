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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.apache.lucene.util.BitUtil;

/**
 * Scalar kernels with transparent SIMD replacements for coarse scans, packing, and INT8 reranking.
 */
class Kernels {
  static final Kernels INSTANCE = load();

  /** Loads SIMD kernels when the vector module is available. */
  private static Kernels load() {
    boolean simd = ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent();
    try {
      var type = Class.forName(Kernels.class.getName() + (simd ? "$Panama" : ""));
      return (Kernels) type.getDeclaredConstructor().newInstance();
    } catch (ReflectiveOperationException | LinkageError _) {
      return new Kernels();
    }
  }

  /** Computes Hamming distances for rows stored in a memory segment. */
  void hamming(byte[] q, MemorySegment codes, long offset, int rows, int[] out) {
    for (int r = 0; r < rows; r++) out[r] = hamming(q, codes, offset + (long) r * q.length);
  }

  /** Computes one Hamming distance from a memory segment. */
  int hamming(byte[] q, MemorySegment codes, long offset) {
    return tail(q, codes, offset, 0);
  }

  /** Computes Hamming distances for rows stored in a byte array. */
  void hamming(byte[] q, byte[] codes, int offset, int rows, int[] out) {
    for (int r = 0; r < rows; r++) out[r] = hamming(q, codes, offset + r * q.length);
  }

  /** Computes Hamming distances at arbitrary row offsets. */
  final void hammingAt(byte[] q, byte[] codes, int[] offsets, int count, int[] out) {
    for (int i = 0; i < count; i++) out[i] = hamming(q, codes, offsets[i]);
  }

  /** Computes one Hamming distance from a byte array. */
  int hamming(byte[] q, byte[] codes, int offset) {
    return scalarHamming(q, codes, offset);
  }

  /** Computes Hamming distance directly from heap arrays without foreign-memory wrappers. */
  private static int scalarHamming(byte[] q, byte[] codes, int offset) {
    int sum = 0;
    for (int i = 0; i < q.length; i++) {
      sum += Integer.bitCount((q[i] ^ codes[offset + i]) & 255);
    }
    return sum;
  }

  /** Finishes a Hamming distance with scalar XOR and popcount. */
  static int tail(byte[] q, MemorySegment codes, long offset, int from) {
    int sum = 0;
    for (int i = from; i < q.length; i++)
      sum += Integer.bitCount((q[i] ^ codes.get(ValueLayout.JAVA_BYTE, offset + i)) & 255);
    return sum;
  }

  /** Keeps distance positions that are at or below the threshold. */
  int filterAtMost(int[] distances, int from, int count, int threshold, int[] out) {
    int kept = 0;
    for (int i = 0; i < count; i++) if (distances[from + i] <= threshold) out[kept++] = i;
    return kept;
  }

  /** Computes one signed-byte dot product. */
  int dotProduct(byte[] query, byte[] records, int offset) {
    int sum = 0;
    for (int i = 0; i < query.length; i++) sum += query[i] * records[offset + i];
    return sum;
  }

  /** Computes signed-byte dot products for a batch of records. */
  void dotProducts(byte[] query, byte[] records, int stride, int count, int[] out) {
    for (int i = 0; i < count; i++) out[i] = dotProduct(query, records, i * stride);
  }

  /** Packs two threshold comparisons into adjacent bit planes. */
  void pack2(float[] vector, int dim, byte[] dest, int offset, float low, float high) {
    int planeBytes = (dim + 7) >>> 3;
    java.util.Arrays.fill(dest, offset, offset + 2 * planeBytes, (byte) 0);
    for (int d = 0; d < dim; d++) {
      int mask = 1 << (d & 7), at = offset + (d >>> 3);
      if (vector[d] >= low) dest[at] |= (byte) mask;
      if (vector[d] >= high) dest[at + planeBytes] |= (byte) mask;
    }
  }

  /** Vector API implementations for the scan and quantization hot paths. */
  static final class Panama extends Kernels {
    private static final VectorSpecies<Long> LONGS = LongVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Byte> BYTES = LONGS.withLanes(byte.class);
    private static final VectorSpecies<Byte> DOT_BYTES = ByteVector.SPECIES_64;
    private static final VectorSpecies<Integer> DOT_INTS = IntVector.SPECIES_256;
    private static final VectorSpecies<Float> FLOATS = FloatVector.SPECIES_PREFERRED;

    /** Uses SIMD for fixed-width native-memory Hamming batches. */
    @Override
    void hamming(byte[] q, MemorySegment codes, long offset, int rows, int[] out) {
      if (codes.isNative()) {
        if (q.length == BYTES.length() * 8) {
          hamming8Native(q, codes, offset, rows, out);
          return;
        }
      }
      super.hamming(q, codes, offset, rows, out);
    }

    /** Keeps the production scan small enough for C2 to retain all query vectors in registers. */
    private static void hamming8Native(
        byte[] q, MemorySegment codes, long offset, int rows, int[] out) {
      int step = BYTES.length(), len = 8 * step;
      var q0 = query(q, 0);
      var q1 = query(q, step);
      var q2 = query(q, 2 * step);
      var q3 = query(q, 3 * step);
      var q4 = query(q, 4 * step);
      var q5 = query(q, 5 * step);
      var q6 = query(q, 6 * step);
      var q7 = query(q, 7 * step);
      for (int r = 0; r < rows; r++) {
        long at = offset + (long) r * len;
        var s0 = popcount(q0, code(codes, at));
        var s1 = popcount(q1, code(codes, at + step));
        var s2 = popcount(q2, code(codes, at + 2L * step));
        var s3 = popcount(q3, code(codes, at + 3L * step));
        var s4 = popcount(q4, code(codes, at + 4L * step));
        var s5 = popcount(q5, code(codes, at + 5L * step));
        var s6 = popcount(q6, code(codes, at + 6L * step));
        var s7 = popcount(q7, code(codes, at + 7L * step));
        out[r] =
            (int)
                s0.add(s1)
                    .add(s2.add(s3))
                    .add(s4.add(s5).add(s6.add(s7)))
                    .reduceLanes(VectorOperators.ADD);
      }
    }

    /** Uses SIMD for fixed-width byte-array Hamming batches. */
    @Override
    void hamming(byte[] q, byte[] codes, int offset, int rows, int[] out) {
      if (q.length == BYTES.length() * 8) {
        var q0 = query(q, 0);
        var q1 = query(q, BYTES.length());
        var q2 = query(q, 2 * BYTES.length());
        var q3 = query(q, 3 * BYTES.length());
        var q4 = query(q, 4 * BYTES.length());
        var q5 = query(q, 5 * BYTES.length());
        var q6 = query(q, 6 * BYTES.length());
        var q7 = query(q, 7 * BYTES.length());
        for (int r = 0; r < rows; r++) {
          int at = offset + r * q.length;
          var s0 = popcount(q0, code(codes, at));
          var s1 = popcount(q1, code(codes, at + BYTES.length()));
          var s2 = popcount(q2, code(codes, at + 2 * BYTES.length()));
          var s3 = popcount(q3, code(codes, at + 3 * BYTES.length()));
          var s4 = popcount(q4, code(codes, at + 4 * BYTES.length()));
          var s5 = popcount(q5, code(codes, at + 5 * BYTES.length()));
          var s6 = popcount(q6, code(codes, at + 6 * BYTES.length()));
          var s7 = popcount(q7, code(codes, at + 7 * BYTES.length()));
          out[r] =
              (int)
                  s0.add(s1)
                      .add(s2.add(s3))
                      .add(s4.add(s5).add(s6.add(s7)))
                      .reduceLanes(VectorOperators.ADD);
        }
        return;
      }
      super.hamming(q, codes, offset, rows, out);
    }

    /** Uses SIMD for one native-memory Hamming distance. */
    @Override
    int hamming(byte[] q, MemorySegment codes, long offset) {
      if (codes.isNative() == false) return tail(q, codes, offset, 0);
      var sum = LongVector.zero(LONGS);
      int i = 0;
      for (; i <= q.length - BYTES.length(); i += BYTES.length()) {
        var code = LongVector.fromMemorySegment(LONGS, codes, offset + i, ByteOrder.nativeOrder());
        sum = sum.add(popcount(q, i, code));
      }
      return (int) sum.reduceLanes(VectorOperators.ADD) + tail(q, codes, offset, i);
    }

    /** Uses SIMD for one byte-array Hamming distance. */
    @Override
    int hamming(byte[] q, byte[] codes, int offset) {
      var sum = LongVector.zero(LONGS);
      int i = 0;
      for (; i <= q.length - BYTES.length(); i += BYTES.length()) {
        var code = ByteVector.fromArray(BYTES, codes, offset + i).reinterpretAsLongs();
        sum = sum.add(popcount(q, i, code));
      }
      int total = (int) sum.reduceLanes(VectorOperators.ADD);
      for (; i < q.length; i++) total += Integer.bitCount((q[i] ^ codes[offset + i]) & 255);
      return total;
    }

    /** XORs a query chunk with a code chunk and popcounts each lane. */
    private static LongVector popcount(byte[] q, int i, LongVector code) {
      return popcount(query(q, i), code);
    }

    /** Loads a query chunk as long lanes. */
    private static LongVector query(byte[] q, int offset) {
      return ByteVector.fromArray(BYTES, q, offset).reinterpretAsLongs();
    }

    /** Loads a byte-array code chunk as long lanes. */
    private static LongVector code(byte[] codes, int offset) {
      return ByteVector.fromArray(BYTES, codes, offset).reinterpretAsLongs();
    }

    /** Loads a native-memory code chunk as long lanes. */
    private static LongVector code(MemorySegment codes, long offset) {
      return LongVector.fromMemorySegment(LONGS, codes, offset, ByteOrder.nativeOrder());
    }

    /** XORs and popcounts two vectors lane by lane. */
    private static LongVector popcount(LongVector query, LongVector code) {
      return query.lanewise(VectorOperators.XOR, code).lanewise(VectorOperators.BIT_COUNT);
    }

    /** Uses SIMD for one signed-byte dot product. */
    @Override
    int dotProduct(byte[] query, byte[] records, int offset) {
      var sum = IntVector.zero(DOT_INTS);
      int i = 0, end = DOT_BYTES.loopBound(query.length);
      for (; i < end; i += DOT_BYTES.length()) {
        var q = ByteVector.fromArray(DOT_BYTES, query, i);
        var d = ByteVector.fromArray(DOT_BYTES, records, offset + i);
        var qi = q.convertShape(VectorOperators.B2I, DOT_INTS, 0);
        var di = d.convertShape(VectorOperators.B2I, DOT_INTS, 0);
        sum = sum.add(qi.mul(di));
      }
      int total = sum.reduceLanes(VectorOperators.ADD);
      for (; i < query.length; i++) total += query[i] * records[offset + i];
      return total;
    }

    /** Uses SIMD to score four signed-byte records at a time. */
    @Override
    void dotProducts(byte[] query, byte[] records, int stride, int count, int[] out) {
      int r = 0;
      for (; r + 4 <= count; r += 4) {
        int o0 = r * stride, o1 = o0 + stride, o2 = o1 + stride, o3 = o2 + stride;
        var a0 = IntVector.zero(DOT_INTS);
        var a1 = IntVector.zero(DOT_INTS);
        var a2 = IntVector.zero(DOT_INTS);
        var a3 = IntVector.zero(DOT_INTS);
        int i = 0, end = DOT_BYTES.loopBound(query.length);
        for (; i < end; i += DOT_BYTES.length()) {
          var q =
              (IntVector)
                  ByteVector.fromArray(DOT_BYTES, query, i)
                      .convertShape(VectorOperators.B2I, DOT_INTS, 0);
          a0 = signedFma(q, records, o0 + i, a0);
          a1 = signedFma(q, records, o1 + i, a1);
          a2 = signedFma(q, records, o2 + i, a2);
          a3 = signedFma(q, records, o3 + i, a3);
        }
        int s0 = a0.reduceLanes(VectorOperators.ADD);
        int s1 = a1.reduceLanes(VectorOperators.ADD);
        int s2 = a2.reduceLanes(VectorOperators.ADD);
        int s3 = a3.reduceLanes(VectorOperators.ADD);
        for (; i < query.length; i++) {
          int q = query[i];
          s0 += q * records[o0 + i];
          s1 += q * records[o1 + i];
          s2 += q * records[o2 + i];
          s3 += q * records[o3 + i];
        }
        out[r] = s0;
        out[r + 1] = s1;
        out[r + 2] = s2;
        out[r + 3] = s3;
      }
      for (; r < count; r++) out[r] = dotProduct(query, records, r * stride);
    }

    /** Multiplies signed byte lanes and accumulates them into integer lanes. */
    private static IntVector signedFma(IntVector query, byte[] records, int offset, IntVector sum) {
      var doc =
          (IntVector)
              ByteVector.fromArray(DOT_BYTES, records, offset)
                  .convertShape(VectorOperators.B2I, DOT_INTS, 0);
      return sum.add(query.mul(doc));
    }

    /** Uses SIMD comparisons to pack two threshold bit planes. */
    @Override
    void pack2(float[] vector, int dim, byte[] dest, int offset, float low, float high) {
      int planeBytes = (dim + 7) >>> 3;
      int vectorized = dim & -64;
      for (int base = 0; base < vectorized; base += 64) {
        long lowBits = 0, highBits = 0;
        for (int i = 0; i < 64; i += FLOATS.length()) {
          var values = FloatVector.fromArray(FLOATS, vector, base + i);
          lowBits |= values.compare(VectorOperators.GE, low).toLong() << i;
          highBits |= values.compare(VectorOperators.GE, high).toLong() << i;
        }
        int at = offset + (base >>> 3);
        BitUtil.VH_LE_LONG.set(dest, at, lowBits);
        BitUtil.VH_LE_LONG.set(dest, at + planeBytes, highBits);
      }
      if (vectorized < dim) {
        int tail = vectorized >>> 3;
        java.util.Arrays.fill(dest, offset + tail, offset + planeBytes, (byte) 0);
        java.util.Arrays.fill(dest, offset + planeBytes + tail, offset + 2 * planeBytes, (byte) 0);
        for (int d = vectorized; d < dim; d++) {
          int mask = 1 << (d & 7), at = offset + (d >>> 3);
          if (vector[d] >= low) dest[at] |= (byte) mask;
          if (vector[d] >= high) dest[at + planeBytes] |= (byte) mask;
        }
      }
    }
  }
}
