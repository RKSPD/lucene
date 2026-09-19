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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Random;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.IVFasterEvoVectorsFormat.FineTier;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.BitUtil;
import org.apache.lucene.util.VectorUtil;

/** Compact coarse and fine vector representations plus the shared rotation. */
final class Tiers {
  /** Encodes reranking vectors as either exact FP32 values or scaled INT8 values. */
  static final class FineCodec {
    private static final Kernels K = Kernels.INSTANCE;

    final FineTier tier;
    final int dim, codeBytes;

    /** Configures the fine vector encoding for a dimension. */
    FineCodec(FineTier tier, int dim) {
      this.tier = tier;
      this.dim = dim;
      codeBytes = tier == FineTier.FP32 ? dim * Float.BYTES : dim;
    }

    /** Encodes a rotated vector into its fine-tier record. */
    void encode(float[] rotated, byte[] record, int offset) {
      if (tier == FineTier.FP32) {
        var bytes = ByteBuffer.wrap(record, offset, codeBytes).order(ByteOrder.LITTLE_ENDIAN);
        bytes.asFloatBuffer().put(rotated, 0, dim);
        return;
      }
      float scale = quantize(rotated, record, offset);
      BitUtil.VH_LE_INT.set(
          record, offset + CodeRecord.scaleOffset(codeBytes), Float.floatToIntBits(scale));
    }

    /** Decodes a fine-tier record back into a rotated vector. */
    void decode(byte[] record, int offset, float[] dest) {
      if (tier == FineTier.FP32) {
        var bytes = ByteBuffer.wrap(record, offset, codeBytes).order(ByteOrder.LITTLE_ENDIAN);
        bytes.asFloatBuffer().get(dest, 0, dim);
        return;
      }
      float scale =
          Float.intBitsToFloat(
              (int) BitUtil.VH_LE_INT.get(record, offset + CodeRecord.scaleOffset(codeBytes)));
      for (int d = 0; d < dim; d++) dest[d] = scale * record[offset + d];
    }

    /** Prepares a reusable scorer for one rotated query vector. */
    Query query(float[] rotated, VectorSimilarityFunction similarity) {
      return new Query(rotated, similarity);
    }

    final class Query {
      private final VectorSimilarityFunction similarity;
      private final float[] query, decoded = new float[dim];
      private final byte[] levels = new byte[dim];
      private final float queryScale;
      private int[] dots = new int[0];

      /** Quantizes and stores the query state used for batch scoring. */
      private Query(float[] rotated, VectorSimilarityFunction similarity) {
        this.similarity = similarity;
        query = rotated.clone();
        queryScale = tier == FineTier.INT8 ? quantize(query, levels, 0) : 1f;
      }

      /** Scores a batch of encoded fine-tier records. */
      void score(byte[] records, int stride, int count, float[] scores) {
        if (tier == FineTier.INT8) {
          dots = ArrayUtil.growNoCopy(dots, count);
          K.dotProducts(levels, records, stride, count, dots);
        }
        for (int r = 0; r < count; r++) {
          if (tier == FineTier.FP32) {
            decode(records, r * stride, decoded);
            scores[r] = finish(VectorUtil.dotProduct(query, decoded));
          } else {
            float docScale =
                Float.intBitsToFloat(
                    (int)
                        BitUtil.VH_LE_INT.get(
                            records, r * stride + CodeRecord.scaleOffset(codeBytes)));
            scores[r] = finish((double) queryScale * docScale * dots[r]);
          }
        }
      }

      /** Converts a raw dot product to the requested Lucene similarity score. */
      private float finish(double dot) {
        return switch (similarity) {
          case null -> (float) dot;
          case EUCLIDEAN -> (float) (1 / (1 + Math.max(0, 2 - 2 * dot)));
          case DOT_PRODUCT, COSINE -> (float) Math.max(0, (1 + dot) / 2);
          case MAXIMUM_INNER_PRODUCT -> (float) (dot < 0 ? 1 / (1 - dot) : dot + 1);
        };
      }
    }

    /** Quantizes a float vector to signed bytes and returns its scale. */
    private float quantize(float[] vector, byte[] code, int offset) {
      float maxAbs = 0f;
      for (int d = 0; d < dim; d++) maxAbs = Math.max(maxAbs, Math.abs(vector[d]));
      if (maxAbs == 0f) {
        Arrays.fill(code, offset, offset + dim, (byte) 0);
        return 1f;
      }
      float scale = maxAbs / 127f, inverse = 127f / maxAbs;
      for (int d = 0; d < dim; d++) {
        code[offset + d] = (byte) Math.max(-127, Math.min(127, Math.round(vector[d] * inverse)));
      }
      return scale;
    }
  }

  /**
   * Extended Hamming code with a sign bit and magnitude bit where popcount is the similarity score.
   *
   * <p>The representation is crafted for a high-CPU-memory-bandwidth XOR and popcount scan, which
   * lets SIMD reject candidates with very low latency before fine reranking.
   */
  static final class Nitrox2 {
    static final int PLANES = 2;
    private static final Kernels K = Kernels.INSTANCE;

    /** Returns the byte width of one bit plane. */
    static int planeBytes(int dim) {
      return (dim + 7) >>> 3;
    }

    /** Returns the full two-plane code size for a vector. */
    static int bytesPerVector(int dim) {
      return PLANES * planeBytes(dim);
    }

    /** Encodes the sign and magnitude planes used by the coarse scan. */
    static void encode(float[] vector, int dim, byte[] dest, int offset) {
      float clip = (float) (1 / Math.sqrt(dim));
      K.pack2(vector, dim, dest, offset, -0.5f * clip, 0.5f * clip);
    }
  }

  static final class CodeRecord {
    /** Aligns a fine-code record and its metadata to a cache line. */
    static int length(int codeBytes) {
      return (codeBytes + 24 + 63) / 64 * 64;
    }

    /** Returns the stored primary-cell offset within a record. */
    static int primaryCellOffset(int codeBytes) {
      return codeBytes + 4;
    }

    /** Returns the stored quantization-scale offset within a record. */
    static int scaleOffset(int codeBytes) {
      return codeBytes + 8;
    }
  }

  /** Spreads vector energy across dimensions so the same coarse thresholds work for every lane. */
  record HadamardRotation(float[] signs, int[] perm) {
    /** Creates a deterministic signed permutation for the rotation. */
    static HadamardRotation create(int dim, long seed) {
      Random random = new Random(seed);
      float[] signs = new float[dim];
      int[] perm = new int[dim];
      for (int i = 0; i < dim; i++) signs[i] = random.nextBoolean() ? 1f : -1f;
      for (int i = 0; i < dim; i++) perm[i] = i;
      for (int i = dim - 1; i > 0; i--) {
        int j = random.nextInt(i + 1), tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
      }
      return new HadamardRotation(signs, perm);
    }

    /** Applies the signed permutation and block Hadamard transform. */
    void rotate(float[] in, float[] out) {
      for (int i = 0; i < perm.length; i++) out[i] = signs[perm[i]] * in[perm[i]];
      transform(out);
    }

    /** Reverses the Hadamard transform and signed permutation. */
    void inverseRotate(float[] in, float[] out) {
      float[] f = in.clone();
      transform(f);
      for (int i = 0; i < perm.length; i++) out[perm[i]] = signs[perm[i]] * f[i];
    }

    /** Applies normalized Hadamard transforms to power-of-two blocks. */
    private void transform(float[] a) {
      int dim = perm.length, offset = 0;
      for (int len = Integer.highestOneBit(dim); len != 0; len >>>= 1) {
        if ((dim & len) == 0) continue;
        for (int h = 1; h < len; h <<= 1)
          for (int i = offset; i < offset + len; i += h << 1)
            for (int p = i; p < i + h; p++) {
              float x = a[p], y = a[p + h];
              a[p] = x + y;
              a[p + h] = x - y;
            }
        for (int i = offset; i < offset + len; i++) a[i] *= (float) (1.0 / Math.sqrt(len));
        offset += len;
      }
    }
  }
}
