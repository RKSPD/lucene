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

import static org.apache.lucene.sandbox.codecs.ivfaster_evo.IVFasterEvoVectorsFormat.*;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.IntUnaryOperator;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.Centroids.CentroidCodes;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.Centroids.CentroidGraph;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.Clustering.Parallel;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.Clustering.WarmState;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.IVFasterEvoVectorsReader.Field;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.Tiers.CodeRecord;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.Tiers.FineCodec;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.Tiers.HadamardRotation;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.Tiers.Nitrox2;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.BitUtil;
import org.apache.lucene.util.IORunnable;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.packed.DirectMonotonicWriter;

/**
 * Builds IVFasterEvo fields by rotating, staging, clustering, spilling, and encoding their vectors.
 *
 * <p>Staging keeps the large build data sequential and reusable during clustering, while compatible
 * merges copy encoded rows and warm-start from existing centroid state.
 */
final class IVFasterEvoVectorsWriter extends KnnVectorsWriter {
  private static final int GATHER_AHEAD = 256;

  private final SegmentWriteState state;
  private final IVFasterEvoVectorsFormat format;
  private IndexOutput meta, data;
  private final List<BufferedField> fields = new ArrayList<>();

  /** Creates the metadata and data outputs for one segment. */
  IVFasterEvoVectorsWriter(SegmentWriteState state, IVFasterEvoVectorsFormat format)
      throws IOException {
    this.state = state;
    this.format = format;
    byte[] id = state.segmentInfo.getId();
    try {
      meta = create(META_EXTENSION);
      data = create(DATA_EXTENSION);
      CodecUtil.writeIndexHeader(meta, META_CODEC_NAME, VERSION_CURRENT, id, state.segmentSuffix);
      CodecUtil.writeIndexHeader(data, DATA_CODEC_NAME, VERSION_CURRENT, id, state.segmentSuffix);
    } catch (Throwable t) {
      IOUtils.closeWhileSuppressingExceptions(t, meta, data);
      throw t;
    }
  }

  /** Creates one segment output with the requested extension. */
  private IndexOutput create(String ext) throws IOException {
    String name = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, ext);
    return state.directory.createOutput(name, state.context);
  }

  /** Derives the deterministic rotation seed for a dimension. */
  static long rotationSeed(int dim) {
    return 0x9E3779B97F4A7C15L ^ dim;
  }

  /** Adds a buffered float vector field. */
  @Override
  public KnnFieldVectorsWriter<?> addField(FieldInfo info) {
    if (info.getVectorEncoding() != VectorEncoding.FLOAT32) {
      throw new IllegalArgumentException("IVFasterEvo supports only FLOAT32 vectors");
    }
    BufferedField field = new BufferedField(info);
    fields.add(field);
    return field;
  }

  /** Sorts, stages, clusters, and writes all buffered fields. */
  @Override
  public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    for (BufferedField field : fields) {
      int count = field.size, dim = field.info.getVectorDimension();
      float[][] vectors = field.vectors;
      long[] keys = new long[count];
      for (int i = 0; i < count; i++) {
        int doc = sortMap == null ? field.docIds[i] : sortMap.oldToNew(field.docIds[i]);
        keys[i] = (long) doc << 32 | i;
      }
      Arrays.sort(keys);
      var si = state.segmentInfo;
      try (var staged = new StagedVectors(state, new FineCodec(format.fineTier, dim), 0)) {
        for (int start = 0, n; start < count; start += n) {
          n = Math.min(StagedVectors.CHUNK_ORDS, count - start);
          staged.add(start, n, k -> (int) (keys[k] >>> 32), (k, _) -> vectors[(int) keys[k]]);
          for (int k = start; k < start + n; k++) vectors[(int) keys[k]] = null;
        }
        HotStart.Seed hs = HotStart.seed(si.dir, si.name, field.info, format.nlist, count);
        if (state.infoStream.isEnabled("IVFE")) {
          String src = hs == null ? "cold" : "segment=" + hs.segment() + " vectors=" + hs.vectors();
          String message = "flush field=" + field.info.name + " docs=" + count + " source=" + src;
          state.infoStream.message("IVFE", message);
        }
        writeField(
            field.info,
            staged,
            hs == null ? null : hs.centroids(),
            null,
            hs == null ? si.name : hs.lineage());
      }
      field.size = 0;
    }
  }

  private static final class MergeSub extends DocIDMerger.Sub {
    final KnnVectorValues.DocIndexIterator iterator;
    final int reader;

    /** Wraps one source iterator for document-ID merging. */
    MergeSub(MergeState.DocMap docMap, KnnVectorValues.DocIndexIterator iterator, int reader) {
      super(docMap);
      this.iterator = iterator;
      this.reader = reader;
    }

    /** Advances the wrapped source iterator. */
    @Override
    public int nextDoc() throws IOException {
      return iterator.nextDoc();
    }
  }

  /** Merges one field while reusing compatible encodings and warm state. */
  @Override
  public IORunnable mergeOneField(FieldInfo info, MergeState mergeState) throws IOException {
    int dim = info.getVectorDimension(), readers = mergeState.knnVectorsReaders.length;
    Field[] views = new Field[readers];
    HotStart.Seed[] snapshots = new HotStart.Seed[readers];
    FloatVectorValues[] vals = new FloatVectorValues[readers];
    List<MergeSub> subs = new ArrayList<>();
    int donor = -1, at = 0;
    for (int r = 0; r < readers; r++) {
      var reader = mergeState.knnVectorsReaders[r];
      if (reader == null) continue;
      if (reader.unwrapReaderForField(info.name) instanceof IVFasterEvoVectorsReader evo
          && evo.field(info.name) instanceof Field view
          && view.rotationSeed == rotationSeed(dim)
          && view.fineTier == format.fineTier) {
        views[r] = view;
        snapshots[r] = HotStart.snapshot(state.segmentInfo.dir, evo.segment(), info);
        boolean large = view.nlist >= Math.max(1, format.nlist / 2);
        if (large && (donor < 0 || view.count > views[donor].count)) donor = r;
      }
      vals[r] = reader.getFloatVectorValues(info.name);
      if (vals[r] != null) subs.add(new MergeSub(mergeState.docMaps[r], vals[r].iterator(), r));
    }
    DocIDMerger<MergeSub> merger = DocIDMerger.of(subs, mergeState.needsIndexSort);
    Field from = donor < 0 ? null : views[donor];
    HotStart.Seed donorSnapshot = donor < 0 ? null : snapshots[donor];
    int[] cells = new int[0], cell2 = new int[0];
    float[] d1 = new float[0], d2 = new float[0];
    int[][] seedMembers = new int[readers][];
    try (var staged = new StagedVectors(state, new FineCodec(format.fineTier, dim), readers)) {
      int max = StagedVectors.CHUNK_ORDS;
      int[] srcs = new int[max], ords = new int[max], docs = new int[max];
      StagedVectors.Rows rows =
          (j, local) -> {
            int r = srcs[j];
            if (views[r] != null) {
              views[r].copyRow(ords[j], docs[j], staged.chunk, j * staged.stride);
              return null;
            }
            if (local[r] == null) local[r] = vals[r].copy();
            return local[r].vectorValue(ords[j]);
          };
      for (int n = max; n == max; ) {
        mergeState.checkAborted();
        n = 0;
        for (MergeSub sub; n < max && (sub = merger.next()) != null; n++) {
          srcs[n] = sub.reader;
          ords[n] = sub.iterator.index();
          docs[n] = sub.mappedDocID;
        }
        staged.add(0, n, j -> docs[j], rows);
        if (from == null) continue;
        cells = ArrayUtil.grow(cells, at + n);
        cell2 = ArrayUtil.grow(cell2, at + n);
        d1 = ArrayUtil.grow(d1, at + n);
        d2 = ArrayUtil.grow(d2, at + n);
        for (int j = 0; j < n; j++) {
          int r = srcs[j], ord = ords[j];
          HotStart.Seed snapshot = snapshots[r];
          boolean sameLineage =
              snapshot != null
                  && donorSnapshot != null
                  && snapshot.lineage().equals(donorSnapshot.lineage())
                  && views[r].nlist == from.nlist;
          if (sameLineage) {
            int cell =
                snapshot.assignment() != null
                        && snapshot.assignment().length == views[r].count
                        && ord < snapshot.assignment().length
                    ? snapshot.assignment()[ord]
                    : views[r].cellOf(ord);
            cells[at] = cell;
            cell2[at] =
                snapshot.cell2() != null && ord < snapshot.cell2().length
                    ? snapshot.cell2()[ord]
                    : -1;
            d1[at] = d2[at] = Float.NaN;
            if (seedMembers[r] == null) seedMembers[r] = new int[from.nlist];
            seedMembers[r][cell]++;
          } else {
            cells[at] = r == donor ? from.cellOf(ord) : -1;
            cell2[at] = -1;
            d1[at] = d2[at] = r == donor ? Float.NaN : Float.MAX_VALUE;
            if (r == donor) {
              if (seedMembers[r] == null) seedMembers[r] = new int[from.nlist];
              seedMembers[r][cells[at]]++;
            }
          }
          at++;
        }
      }
      float[][] seed =
          from == null ? null : weightedSeed(views, snapshots, seedMembers, donor, dim);
      WarmState warm =
          seed == null
              ? null
              : new WarmState(
                  ArrayUtil.copyOfSubArray(cells, 0, at),
                  ArrayUtil.copyOfSubArray(cell2, 0, at),
                  ArrayUtil.copyOfSubArray(d1, 0, at),
                  ArrayUtil.copyOfSubArray(d2, 0, at));
      writeField(
          info,
          staged,
          seed,
          warm,
          donorSnapshot == null ? state.segmentInfo.name : donorSnapshot.lineage());
    }
    return null;
  }

  /** Averages corresponding same-lineage centroids by their live primary-cell populations. */
  private static float[][] weightedSeed(
      Field[] views,
      HotStart.Seed[] snapshots,
      int[][] members,
      int donor,
      int dim) {
    Field from = views[donor];
    HotStart.Seed donorSnapshot = snapshots[donor];
    float[][] seed = new float[from.nlist][dim];
    long[] weights = new long[from.nlist];
    for (int r = 0; r < views.length; r++) {
      if (views[r] == null || members[r] == null || views[r].nlist != from.nlist) continue;
      HotStart.Seed snapshot = snapshots[r];
      if (r != donor
          && (snapshot == null
              || donorSnapshot == null
              || snapshot.lineage().equals(donorSnapshot.lineage()) == false)) continue;
      for (int c = 0; c < from.nlist; c++) {
        int weight = members[r][c];
        if (weight == 0) continue;
        weights[c] += weight;
        float[] source = views[r].centroids[c], target = seed[c];
        for (int d = 0; d < dim; d++) target[d] += weight * source[d];
      }
    }
    for (int c = 0; c < from.nlist; c++) {
      if (weights[c] == 0) {
        System.arraycopy(from.centroids[c], 0, seed[c], 0, dim);
      } else {
        normalize(seed[c]);
      }
    }
    return seed;
  }

  /** Normalizes one centroid in place. */
  private static void normalize(float[] vector) {
    double norm = 0;
    for (float value : vector) norm += (double) value * value;
    if (norm == 0) return;
    float scale = (float) (1.0 / Math.sqrt(norm));
    for (int d = 0; d < vector.length; d++) vector[d] *= scale;
  }

  /** Clusters staged rows and writes all field sections. */
  private void writeField(
      FieldInfo info, StagedVectors staged, float[][] seed, WarmState warm, String lineage)
      throws IOException {
    RandomAccessInput rows = staged.finish();
    int dim = info.getVectorDimension(), count = staged.count;
    if (seed != null && seed.length > count) seed = null;
    if (seed == null) warm = null;
    int nlist = seed != null ? seed.length : Math.min(format.nlist, count);
    meta.writeInt(info.number);
    meta.writeByte((byte) format.fineTier.ordinal());
    meta.writeVInt(dim);
    meta.writeVInt(nlist);
    meta.writeVInt(count);
    meta.writeLong(rotationSeed(dim));
    meta.writeVInt(format.nprobe);
    meta.writeVInt(format.spillBits);
    if (nlist == 0) {
      for (int s = 0; s < 6; s++) meta.writeVLong(data.getFilePointer());
      return;
    }
    Clustering.Result cl = Clustering.cluster(staged, nlist, seed, warm, format.spillBits);
    HotStart.publish(state.segmentInfo.dir, state.segmentInfo.name, lineage, info, cl, count);
    int[] cellStart = new int[nlist + 1];
    for (int i = 0; i < count; i++) {
      for (int k = cl.cellCount(i) - 1; k >= 0; k--) cellStart[cl.cell(i, k) + 1]++;
    }
    for (int c = 0; c < nlist; c++) cellStart[c + 1] += cellStart[c];
    int[] slotRow = new int[cellStart[nlist]], primarySlot = new int[count];
    int[] next = ArrayUtil.copyOfSubArray(cellStart, 0, nlist);
    for (int i = 0; i < count; i++) {
      for (int k = 0, n = cl.cellCount(i); k < n; k++) {
        int slot = next[cl.cell(i, k)]++;
        slotRow[slot] = i;
        if (k == 0) primarySlot[i] = slot;
      }
    }
    meta.writeVLong(data.getFilePointer());
    for (float[] centroid : cl.centroids()) {
      for (float v : centroid) data.writeInt(Float.floatToIntBits(v));
    }
    byte[] row = new byte[staged.recordLen];
    int stride = staged.stride, primaryOffset = CodeRecord.primaryCellOffset(staged.fine.codeBytes);
    for (int pass = 0; pass < 2; pass++) {
      int from = pass * staged.recordLen, len = pass == 0 ? staged.recordLen : staged.coarseBytes;
      meta.writeVLong(data.getFilePointer());
      for (int slot = 0; slot < slotRow.length; slot++) {
        if ((slot & (GATHER_AHEAD - 1)) == 0) {
          int end = Math.min(slotRow.length, slot + GATHER_AHEAD);
          for (int p = slot; p < end; p++) rows.prefetch((long) slotRow[p] * stride, stride);
        }
        rows.readBytes((long) slotRow[slot] * stride + from, row, 0, len);
        if (pass == 0) BitUtil.VH_LE_INT.set(row, primaryOffset, cl.cell(slotRow[slot], 0));
        data.writeBytes(row, 0, len);
      }
    }
    meta.writeVLong(data.getFilePointer());
    if (nlist > 1) {
      CentroidGraph.build(new CentroidCodes(cl.centroids(), dim, null), dim).write(data);
    }
    meta.writeVLong(data.getFilePointer());
    for (int slot : primarySlot) data.writeInt(slot);
    meta.writeVLong(data.getFilePointer());
    var w = DirectMonotonicWriter.getInstance(meta, data, nlist + 1, DIRECT_MONOTONIC_BLOCK_SHIFT);
    for (int start : cellStart) w.add((long) start * Integer.BYTES);
    w.finish();
  }

  /** Writes end markers and checksums for both output files. */
  @Override
  public void finish() throws IOException {
    meta.writeInt(-1);
    CodecUtil.writeFooter(meta);
    CodecUtil.writeFooter(data);
  }

  /** Closes the metadata and data outputs. */
  @Override
  public void close() throws IOException {
    IOUtils.close(meta, data);
  }

  /** Estimates memory retained by buffered fields. */
  @Override
  public long ramBytesUsed() {
    long total = 0;
    for (BufferedField field : fields) total += field.ramBytesUsed();
    return total;
  }

  private static final class BufferedField extends KnnFieldVectorsWriter<float[]> {
    final FieldInfo info;
    float[][] vectors = new float[16][];
    int[] docIds = new int[16];
    int size;

    /** Creates an in-memory buffer for one field. */
    BufferedField(FieldInfo info) {
      this.info = info;
    }

    /** Buffers a copied vector for one document. */
    @Override
    public void addValue(int docID, float[] value) {
      if (size > 0 && docIds[size - 1] == docID) {
        throw new IllegalArgumentException(
            "field \"" + info.name + "\" appears more than once in document " + docID);
      }
      if (size == vectors.length) {
        vectors = ArrayUtil.grow(vectors, size + 1);
        docIds = ArrayUtil.growExact(docIds, vectors.length);
      }
      vectors[size] = copyValue(value);
      docIds[size++] = docID;
    }

    /** Copies a vector to the field's configured dimension. */
    @Override
    public float[] copyValue(float[] value) {
      return ArrayUtil.copyOfSubArray(value, 0, info.getVectorDimension());
    }

    /** Estimates memory retained by this field buffer. */
    @Override
    public long ramBytesUsed() {
      if (size == 0) return 0;
      long vector = RamUsageEstimator.NUM_BYTES_ARRAY_HEADER + 4L * info.getVectorDimension();
      long arrays = RamUsageEstimator.shallowSizeOf(vectors) + RamUsageEstimator.sizeOf(docIds);
      return arrays + size * RamUsageEstimator.alignObjectSize(vector);
    }
  }

  /**
   * Disk-backed build rows reused across clustering passes without retaining all vectors on heap.
   */
  static final class StagedVectors implements Closeable {
    static final int CHUNK_ORDS = 16_384;

    interface Rows {
      /** Returns one source vector, or null when the row is already encoded. */
      float[] vector(int index, FloatVectorValues[] local) throws IOException;
    }

    final FineCodec fine;
    final int dim, recordLen, coarseBytes, stride;
    private final SegmentWriteState state;
    private final String name;
    private final int readers;
    private byte[] chunk;
    private IndexOutput out;
    private IndexInput input;
    int count;

    /** Creates temporary storage for encoded build rows. */
    StagedVectors(SegmentWriteState state, FineCodec fine, int readers) throws IOException {
      this.state = state;
      this.fine = fine;
      this.readers = readers;
      dim = fine.dim;
      recordLen = CodeRecord.length(fine.codeBytes);
      coarseBytes = Nitrox2.bytesPerVector(dim);
      stride = recordLen + coarseBytes;
      out = state.directory.createTempOutput(state.segmentInfo.name, "ivfstage", state.context);
      name = out.getName();
    }

    /** Encodes and appends a chunk of source rows. */
    void add(int base, int n, IntUnaryOperator docs, Rows source) throws IOException {
      if (chunk == null) chunk = new byte[n * stride];
      int docIdOffset = fine.codeBytes;
      HadamardRotation rotation = HadamardRotation.create(dim, rotationSeed(dim));
      Parallel.overRange(
          n,
          (lo, hi) -> {
            FloatVectorValues[] local = new FloatVectorValues[readers];
            float[] unit = new float[dim], rotated = new float[dim];
            for (int j = lo; j < hi; j++) {
              float[] vector = source.vector(base + j, local);
              if (vector == null) continue;
              int at = j * stride;
              System.arraycopy(vector, 0, unit, 0, dim);
              VectorUtil.l2normalize(unit);
              rotation.rotate(unit, rotated);
              fine.encode(rotated, chunk, at);
              BitUtil.VH_LE_INT.set(chunk, at + docIdOffset, docs.applyAsInt(base + j));
              BitUtil.VH_LE_INT.set(chunk, at + CodeRecord.primaryCellOffset(fine.codeBytes), 0);
              Nitrox2.encode(rotated, dim, chunk, at + recordLen);
            }
          });
      out.writeBytes(chunk, 0, n * stride);
      count += n;
    }

    /** Finalizes staging and opens it for random access. */
    RandomAccessInput finish() throws IOException {
      out.close();
      out = null;
      input = state.directory.openInput(name, state.context);
      return input.randomAccessSlice(0, input.length());
    }

    /** Creates a cursor over staged rows. */
    Cursor cursor() throws IOException {
      return new Cursor();
    }

    final class Cursor {
      private final RandomAccessInput in = input.clone().randomAccessSlice(0, input.length());
      private final byte[] row = new byte[stride];
      private final float[] vector = new float[dim];
      private boolean decoded;

      /** Creates a cursor backed by an independent staged input slice. */
      private Cursor() throws IOException {}

      /** Loads one staged row. */
      void load(int ord) throws IOException {
        in.readBytes((long) ord * stride, row, 0, stride);
        decoded = false;
      }

      /** Lazily decodes the loaded row's fine vector. */
      float[] vector() {
        if (decoded == false) {
          fine.decode(row, 0, vector);
          VectorUtil.l2normalize(vector, false);
          decoded = true;
        }
        return vector;
      }

      /** Copies the loaded row's coarse code. */
      void coarseInto(byte[] dest) {
        System.arraycopy(row, recordLen, dest, 0, coarseBytes);
      }
    }

    /** Closes and deletes the temporary staging file. */
    @Override
    public void close() throws IOException {
      IOUtils.close(out, input, () -> IOUtils.deleteFilesIgnoringExceptions(state.directory, name));
    }
  }
}
