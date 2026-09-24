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

import static org.apache.lucene.codecs.CodecUtil.checkIndexHeader;
import static org.apache.lucene.sandbox.codecs.ivfaster_evo.IVFasterEvoVectorsFormat.*;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.packed.DirectMonotonicReader.loadMeta;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.Float16VectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.Centroids.CentroidCodes;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.Centroids.CentroidGraph;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.Tiers.CodeRecord;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.Tiers.FineCodec;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.Tiers.HadamardRotation;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.Tiers.Nitrox2;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MemorySegmentAccessInput;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitUtil;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.NumericUtils;
import org.apache.lucene.util.SuppressForbidden;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.packed.DirectMonotonicReader;

/**
 * Searches IVFasterEvo fields by probing cells, scanning coarse codes, and reranking a shortlist.
 *
 * <p>The coarse scan is intentionally bandwidth-oriented: compact Nitrox2 rows are processed with
 * XOR and popcount, then only a bounded set of survivors reaches the more expensive fine scorer.
 * Filtered search chooses between scanning selected cells and visiting accepted documents directly.
 */
final class IVFasterEvoVectorsReader extends KnnVectorsReader {
  private static final int SHORTLIST = 700, ADMIT_BLOCK = 256, FILTERED_PROBE_MULTIPLIER = 8;
  private static final int VERIFY_MIN = 64, VERIFY_MULTIPLIER = 2;
  private static final int DEDUP_MASK = (Integer.highestOneBit(SHORTLIST) << 2) - 1;
  private static final Kernels K = Kernels.INSTANCE;
  private static final boolean REPORT_ENGAGEMENT = Boolean.getBoolean("ivfaster.reportEngagement");

  /**
   * Opt-in: read fine records with io_uring O_DIRECT so they never displace cached coarse codes.
   */
  static final String URING_FINE_PROPERTY = "ivfaster.evo.uringFine";

  private static final AtomicLong SCAN_QUERIES = new AtomicLong(),
      SCANNED_SLOTS = new AtomicLong(),
      PROBED_CELLS = new AtomicLong();

  private final Map<String, Field> fields = new HashMap<>();
  private final IndexInput data;
  private final UringDirectReader uring;
  private final String segment;
  private final int segmentMaxDoc;

  /** Opens IVFasterEvo metadata and data for one segment. */
  IVFasterEvoVectorsReader(SegmentReadState state) throws IOException {
    String name = state.segmentInfo.name, sfx = state.segmentSuffix;
    segment = name;
    segmentMaxDoc = state.segmentInfo.maxDoc();
    byte[] id = state.segmentInfo.getId();
    int version = -1;
    String metaName = IndexFileNames.segmentFileName(name, sfx, META_EXTENSION);
    try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaName)) {
      Throwable prior = null;
      try {
        version = checkIndexHeader(meta, META_CODEC_NAME, VERSION_START, VERSION_CURRENT, id, sfx);
        for (int number = meta.readInt(); number != -1; number = meta.readInt()) {
          FieldInfo info = state.fieldInfos.fieldInfo(number);
          if (info == null) throw new CorruptIndexException("invalid field number " + number, meta);
          fields.put(info.name, new Field(meta, info, version));
        }
      } catch (Throwable t) {
        prior = t;
      } finally {
        CodecUtil.checkFooter(meta, prior);
      }
    }
    String dataName = IndexFileNames.segmentFileName(name, sfx, DATA_EXTENSION);
    data = state.directory.openInput(dataName, state.context);
    try {
      int seen = checkIndexHeader(data, DATA_CODEC_NAME, VERSION_START, VERSION_CURRENT, id, sfx);
      if (seen != version) throw new CorruptIndexException("version mismatch", data);
      CodecUtil.retrieveChecksum(data);
    } catch (Throwable t) {
      IOUtils.closeWhileSuppressingExceptions(t, data);
      throw t;
    }
    uring = openUring(state, dataName, version);
  }

  /**
   * Opens direct fine-record reads when requested and possible: a v2 index (so deduplication never
   * touches a fine record) in its own file of a file-system directory, on a kernel with io_uring.
   */
  private static UringDirectReader openUring(SegmentReadState state, String name, int version) {
    if (Boolean.getBoolean(URING_FINE_PROPERTY) == false || version < VERSION_SLOT_DOC) return null;
    if (FilterDirectory.unwrap(state.directory) instanceof FSDirectory fs) {
      try {
        return UringDirectReader.open(fs.getDirectory().resolve(name));
      } catch (IOException _) {
        // fine records are read through the mapped input instead
      }
    }
    return null;
  }

  final class Field {
    final VectorSimilarityFunction similarity;
    final FineTier fineTier;
    final int dim, nlist, count, nprobe, spillBits, recordLen, coarseBytes, docIdOffset;
    final long rotationSeed;
    final long[] sections;
    final DirectMonotonicReader.Meta postingOffsets;
    final HadamardRotation rotation;
    final FineCodec fine;

    RandomAccessInput records, coarse, slotDocs;
    MemorySegmentAccessInput coarseAccess;
    MemorySegment coarseSeg;
    int[] cellStart, ordToSlot, ordToDoc, allCells, primaryCells;
    float[][] centroids;
    private volatile CentroidCodes codes;
    CentroidGraph graph;

    /** Reads immutable metadata for one vector field. */
    Field(ChecksumIndexInput meta, FieldInfo info, int version) throws IOException {
      similarity = info.getVectorSimilarityFunction();
      fineTier = FineTier.values()[meta.readByte()];
      dim = meta.readVInt();
      nlist = meta.readVInt();
      count = meta.readVInt();
      rotationSeed = meta.readLong();
      nprobe = meta.readVInt();
      spillBits = meta.readVInt();
      // Sections: centroids, records, coarse, graph, ordToSlot, [slotDoc,] posting offsets.
      sections = new long[version >= VERSION_SLOT_DOC ? 7 : 6];
      for (int s = 0; s < sections.length; s++) sections[s] = meta.readVLong();
      postingOffsets = nlist == 0 ? null : loadMeta(meta, nlist + 1, DIRECT_MONOTONIC_BLOCK_SHIFT);
      if (dim != info.getVectorDimension()) throw new CorruptIndexException("dimension", meta);
      rotation = HadamardRotation.create(dim, rotationSeed);
      fine = new FineCodec(fineTier, dim);
      recordLen = CodeRecord.length(fine.codeBytes);
      coarseBytes = Nitrox2.bytesPerVector(dim);
      docIdOffset = fine.codeBytes;
    }

    /** Opens a random-access slice for one field section. */
    private RandomAccessInput section(int s) throws IOException {
      return data.randomAccessSlice(sections[s], sections[s + 1] - sections[s]);
    }

    /** Lazily loads postings, record slices, and centroids needed by every search. */
    synchronized Field open() throws IOException {
      if (records != null) return this;
      coarse = section(2);
      if (coarse instanceof MemorySegmentAccessInput in) {
        coarseAccess = in;
        coarseSeg = segmentOrNull(in, 0, coarse.length());
      }
      cellStart = new int[nlist + 1];
      if (nlist > 0) {
        long postings = sections[sections.length - 1];
        RandomAccessInput tail = data.randomAccessSlice(postings, data.length() - postings);
        var offsets = DirectMonotonicReader.getInstance(postingOffsets, tail);
        for (int c = 0; c <= nlist; c++) cellStart[c] = (int) (offsets.get(c) / Integer.BYTES);
      }
      RandomAccessInput recs = section(1);
      if (sections.length == 7) slotDocs = section(5);
      IndexInput all = data.clone();
      centroids = new float[nlist][dim];
      all.seek(sections[0]);
      for (float[] centroid : centroids) all.readFloats(centroid, 0, dim);
      records = recs;
      return this;
    }

    /** Reads one slot's document ID, from the slot-to-document section when the index has one. */
    private int docAt(int slot) throws IOException {
      return slotDocs != null
          ? slotDocs.readInt((long) slot * Integer.BYTES)
          : records.readInt((long) slot * recordLen + docIdOffset);
    }

    /** Lazily loads the compact ordinal-to-primary-slot table. */
    private synchronized void loadOrdToSlot() throws IOException {
      if (ordToSlot != null) return;
      int[] slots = new int[count];
      IndexInput all = data.clone();
      all.seek(sections[4]);
      all.readInts(slots, 0, count);
      ordToSlot = slots;
    }

    /** Lazily loads document IDs needed by vector values and sparse-vector filter lookup. */
    private synchronized void loadOrdinalMappings() throws IOException {
      if (ordToDoc != null) return;
      loadOrdToSlot();
      int[] docs = new int[count];
      for (int ord = 0; ord < count; ord++) docs[ord] = docAt(ordToSlot[ord]);
      ordToDoc = docs;
    }

    /** Returns a mapped slice when one mmap chunk covers the requested range. */
    private MemorySegment segmentOrNull(MemorySegmentAccessInput in, long offset, long length) {
      if (length == 0) return null;
      try {
        return in.segmentSliceOrNull(offset, length);
      } catch (IOException _) {
        return null;
      }
    }

    /**
     * Returns the mapping holding one cell run: the whole coarse section when mappable, otherwise a
     * slice rebased to the run, or null when neither is mapped. Offsets start at {@code runBase}.
     */
    private MemorySegment coarseRun(int slotBase, int rows) {
      if (coarseSeg != null) return coarseSeg;
      return coarseAccess == null
          ? null
          : segmentOrNull(coarseAccess, (long) slotBase * coarseBytes, (long) rows * coarseBytes);
    }

    /** Returns the byte offset of slot {@code slotBase} within its {@code coarseRun} mapping. */
    private long runBase(int slotBase) {
      return coarseSeg == null ? 0 : (long) slotBase * coarseBytes;
    }

    /** Lazily builds centroid codes and loads the navigation graph. */
    private synchronized void loadCodes() throws IOException {
      if (codes != null) return;
      allCells = new int[nlist];
      for (int c = 0; c < nlist; c++) allCells[c] = c;
      long graphLength = sections[4] - sections[3];
      graph = graphLength == 0 ? null : CentroidGraph.read(section(3), dim, graphLength);
      codes = new CentroidCodes(centroids, dim, fine);
    }

    /** Returns the persisted primary cell for one vector ordinal. */
    synchronized int cellOf(int ord) throws IOException {
      loadOrdToSlot();
      if (primaryCells == null) {
        primaryCells = new int[count];
        for (int o = 0; o < count; o++) {
          primaryCells[o] = records.readInt((long) ordToSlot[o] * recordLen + docIdOffset + 4);
        }
      }
      return primaryCells[ord];
    }

    /**
     * Per-query state for cell selection, coarse admission, deduplication, and fine reranking. Each
     * search opens its own slices, which shadow the field's: positional reads through a shared
     * slice are not thread-safe on every directory implementation.
     */
    final class Search {
      final RandomAccessInput records = section(1), coarse = section(2);
      final RandomAccessInput slotDocs = sections.length == 7 ? section(5) : null;
      final float[] rotated = new float[dim];
      final byte[] qCode = new byte[coarseBytes];
      final FineCodec.Query fine;
      final KnnCollector collector;
      final Scratch scratch = Scratch.LOCAL.get();
      final int bins = coarseBytes * 8 + 2, pool = SHORTLIST * (1 + spillBits);
      final int[] histogram = scratch.histogram = ArrayUtil.growNoCopy(scratch.histogram, bins);
      int size, admitted, threshold = bins - 1;

      /** Builds the encoded query and initializes reusable search state. */
      Search(float[] target, KnnCollector collector) throws IOException {
        this.collector = collector;
        Arrays.fill(histogram, 0, bins, 0);
        if (codes == null) loadCodes();
        rotation.rotate(VectorUtil.l2normalize(ArrayUtil.copyOfSubArray(target, 0, dim)), rotated);
        Nitrox2.encode(rotated, dim, qCode, 0);
        fine = Field.this.fine.query(rotated, similarity);
      }

      /** Reads one slot's document ID through this search's own slices. */
      private int docAt(int slot) throws IOException {
        return slotDocs != null
            ? slotDocs.readInt((long) slot * Integer.BYTES)
            : records.readInt((long) slot * recordLen + docIdOffset);
      }

      /** Chooses the filtered or unfiltered search path and executes it. */
      void run(AcceptDocs acceptDocs) throws IOException {
        int probe = nprobe;
        float margin = DEFAULT_PROBE_MARGIN;
        if (collector.getSearchStrategy() instanceof SearchStrategy strategy) {
          probe = strategy.numProbes;
          margin = strategy.probeMargin;
        }
        probe = Math.min(probe, nlist);
        Bits accept = acceptDocs == null ? null : acceptDocs.bits();
        if (accept instanceof BitSet filter) {
          int cost = acceptDocs.cost();
          double parity =
              Math.sqrt((double) SHORTLIST * cellStart[nlist] * coarseBytes / recordLen);
          if (cost > (int) Math.max(SHORTLIST, Math.min(Integer.MAX_VALUE, parity))) {
            filteredScan(selectCells(probe, 1f), filter, cost, probe);
            return;
          }
          boolean dense = count == segmentMaxDoc;
          if (dense) loadOrdToSlot();
          else loadOrdinalMappings();
          int[] slots = new int[64];
          int n = 0;
          DocIdSetIterator accepted = acceptDocs.iterator();
          for (int doc = accepted.nextDoc(); doc != NO_MORE_DOCS; doc = accepted.nextDoc()) {
            int ord = dense ? doc : Arrays.binarySearch(ordToDoc, doc);
            if (ord < 0) continue;
            slots = ArrayUtil.grow(slots, n + 1);
            slots[n++] = ordToSlot[ord];
          }
          rerank(slots, n);
        } else {
          scan(selectCells(probe, margin), accept);
        }
      }

      /** Selects probe cells using graph routing and exact verification. */
      private int[] selectCells(int probe, float margin) {
        int[] candidates = allCells;
        int got = nlist;
        if (graph != null) {
          candidates = scratch.candidates = ArrayUtil.growNoCopy(scratch.candidates, nlist);
          int[] coarse = scratch.coarse = ArrayUtil.growNoCopy(scratch.coarse, nlist);
          int ef = Math.max(CentroidGraph.MIN_EF, probe * CentroidGraph.EF_MULTIPLIER);
          got = graph.search(qCode, ef, candidates, coarse);
          int cap = Math.max(VERIFY_MIN, probe * VERIFY_MULTIPLIER);
          if (cap < got) {
            int[] counts = new int[bins];
            for (int i = 0; i < got; i++) counts[coarse[i]]++;
            int below = 0, bound = 0;
            while (below + counts[bound] <= cap) below += counts[bound++];
            int n = 0, ties = cap - below;
            for (int i = 0; i < got && n < cap; i++) {
              if (coarse[i] < bound || (coarse[i] == bound && ties-- > 0)) {
                candidates[n] = candidates[i];
                coarse[n++] = coarse[i];
              }
            }
            got = n;
          }
        }
        long[] ranked = rank(candidates, got);
        float[] exact = scratch.exact;
        int keep = Math.min(probe, got);
        if (margin != 1f && keep > 1) {
          float bound = exact[(int) ranked[0]] * margin;
          int k = 1;
          while (k < keep && exact[(int) ranked[k]] <= bound) k++;
          keep = k;
        }
        int[] cells = new int[keep];
        for (int i = 0; i < keep; i++) cells[i] = candidates[(int) ranked[i]];
        return cells;
      }

      /** Ranks cells by fine or exact centroid distance. */
      private long[] rank(int[] cells, int n) {
        float[] distances = scratch.exact = Scratch.floats(scratch.exact, n);
        codes.rankCandidates(rotated, cells, n, distances);
        long[] ranked = new long[n];
        for (int i = 0; i < n; i++) {
          ranked[i] = ((long) NumericUtils.floatToSortableInt(distances[i] + 0f) << 32) | i;
        }
        Arrays.sort(ranked);
        return ranked;
      }

      /** Prefetches the coarse rows for selected cells. */
      private int prefetch(int[] cells, int n) throws IOException {
        int total = 0;
        for (int i = 0; i < n; i++) {
          int start = cellStart[cells[i]], rows = cellStart[cells[i] + 1] - start;
          if (rows == 0) continue;
          total += rows;
          coarse.prefetch((long) start * coarseBytes, (long) rows * coarseBytes);
        }
        return total;
      }

      /** Tightens the coarse admission threshold to fit the rerank pool. */
      private void tighten() {
        while (threshold > 0 && admitted - histogram[threshold] >= pool) {
          admitted -= histogram[threshold--];
        }
      }

      /** Scans selected cells and admits the best coarse candidates. */
      private void scan(int[] cells, Bits liveDocs) throws IOException {
        int total = prefetch(cells, cells.length);
        if (REPORT_ENGAGEMENT) {
          SCAN_QUERIES.incrementAndGet();
          SCANNED_SLOTS.addAndGet(total);
          PROBED_CELLS.addAndGet(cells.length);
        }
        long[] packed = scratch.packed = ArrayUtil.growNoCopy(scratch.packed, total);
        for (int cell : cells) {
          int base = cellStart[cell], rows = cellStart[cell + 1] - base;
          int[] distances = scratch.distances = ArrayUtil.growNoCopy(scratch.distances, rows);
          MemorySegment run = coarseRun(base, rows);
          if (run != null) {
            K.hamming(qCode, run, runBase(base), rows, distances);
          } else {
            long at = (long) base * coarseBytes;
            scratch.bytes = ArrayUtil.growNoCopy(scratch.bytes, coarseBytes);
            for (int row = 0; row < rows; row++) {
              coarse.readBytes(at + (long) row * coarseBytes, scratch.bytes, 0, coarseBytes);
              distances[row] = K.hamming(qCode, scratch.bytes, 0);
            }
          }
          for (int from = 0; from < rows; from += ADMIT_BLOCK) {
            int block = Math.min(ADMIT_BLOCK, rows - from);
            int n = K.filterAtMost(distances, from, block, threshold, scratch.kept);
            for (int i = 0; i < n; i++) {
              int row = from + scratch.kept[i];
              histogram[distances[row]]++;
              packed[size++] = ((long) distances[row] << 32) | (base + row);
            }
            admitted += n;
            tighten();
          }
        }
        rerankPool(Math.min(pool, total), liveDocs);
      }

      /** Orders, deduplicates, and fine-reranks the coarse candidate pool. */
      private void rerankPool(int need, Bits live) throws IOException {
        int[] next = scratch.prefix = ArrayUtil.growNoCopy(scratch.prefix, bins + 1);
        int cut = 0, n = 0;
        for (next[0] = 0; cut < bins && next[cut] + histogram[cut] < need; cut++) {
          next[cut + 1] = next[cut] + histogram[cut];
        }
        int ties = need - next[cut];
        long[] ordered = scratch.ordered = ArrayUtil.growNoCopy(scratch.ordered, need);
        for (int i = 0; i < size; i++) {
          int d = (int) (scratch.packed[i] >>> 32);
          if (d < cut || (d == cut && ties-- > 0)) ordered[next[d]++] = scratch.packed[i];
        }
        scratch.newDedup();
        int[] shortlist = scratch.shortlist;
        for (int i = 0; i < next[cut] && n < SHORTLIST; i++) {
          int slot = (int) ordered[i], doc = docAt(slot);
          if ((live == null || live.get(doc)) && scratch.addDistinct(doc)) shortlist[n++] = slot;
        }
        rerank(shortlist, n);
      }

      /** Fine-reranks record slots and sends their scores to the collector. */
      private void rerank(int[] slots, int n) throws IOException {
        if (n == 0) return;
        int len = recordLen, total = Math.multiplyExact(n, len);
        byte[] raw = scratch.bytes = ArrayUtil.growNoCopy(scratch.bytes, total);
        float[] scores = scratch.scores = Scratch.floats(scratch.scores, n);
        readRecords(slots, n, raw);
        fine.score(raw, len, n, scores);
        for (int i = 0; i < n; i++) {
          collector.collect((int) BitUtil.VH_LE_INT.get(raw, i * len + docIdOffset), scores[i]);
        }
        collector.incVisitedCount(n);
      }

      /**
       * Reads the fine records of {@code slots}, in one io_uring batch when direct reads are on.
       */
      private void readRecords(int[] slots, int n, byte[] raw) throws IOException {
        int len = recordLen;
        if (uring != null) {
          long[] positions = new long[n];
          int[] lengths = new int[n];
          for (int i = 0; i < n; i++) {
            positions[i] = sections[1] + (long) slots[i] * len;
            lengths[i] = len;
          }
          try {
            MemorySegment[] runs = uring.readBatch(positions, lengths);
            MemorySegment dest = MemorySegment.ofArray(raw);
            for (int i = 0; i < n; i++) MemorySegment.copy(runs[i], 0, dest, (long) i * len, len);
            return;
          } catch (IOException _) {
            // fall back to the mapped input
          }
        }
        for (int i = 0; i < n; i++) records.readBytes((long) slots[i] * len, raw, i * len, len);
      }

      /** Expands cell probes while scanning a selective document filter. */
      private void filteredScan(int[] seed, BitSet filter, int cost, int probe) throws IOException {
        int targetProbe = probe;
        if (cost > 0 && cost < count) {
          double exponent = 0.7185 - 0.0578 * Math.log(Math.max(2, count / nlist));
          exponent = Math.min(0.50, Math.max(0.15, exponent));
          double widened = Math.ceil(probe * Math.pow((double) count / cost, exponent));
          targetProbe = (int) Math.min(nlist, Math.max(widened, probe));
        }
        int first = Math.max(1, seed.length);
        int maxProbe = Math.max(targetProbe, Math.min(nlist, first * FILTERED_PROBE_MULTIPLIER));
        boolean[] probed = new boolean[nlist];
        long[] ranked = null;
        int served = 0, rankedAt = 0, distinct = 0;
        scratch.newDedup();
        byte[] code = new byte[coarseBytes];
        long survivors = 0;
        int[] batch = new int[maxProbe];
        for (int done = 0, want = first; done < maxProbe; want <<= 1) {
          int n = 0;
          for (want = Math.min(want, maxProbe - done); n < want; ) {
            if (served == seed.length && ranked == null) ranked = rank(allCells, nlist);
            if (rankedAt == nlist) break;
            int cell = ranked == null ? seed[served++] : (int) ranked[rankedAt++];
            if (probed[cell] == false) batch[n++] = cell;
            probed[cell] = true;
          }
          if (n == 0) break;
          done += n;
          scratch.packed = ArrayUtil.grow(scratch.packed, size + prefetch(batch, n));
          for (int c = 0; c < n; c++) {
            int base = cellStart[batch[c]], end = cellStart[batch[c] + 1];
            MemorySegment run = coarseRun(base, end - base);
            long runBase = runBase(base);
            for (int slot = base; slot < end; slot++) {
              int doc = docAt(slot);
              if (filter.get(doc) == false) continue;
              int d;
              if (run != null) {
                d = K.hamming(qCode, run, runBase + (long) (slot - base) * coarseBytes);
              } else {
                coarse.readBytes((long) slot * coarseBytes, code, 0, coarseBytes);
                d = K.hamming(qCode, code, 0);
              }
              survivors++;
              if (distinct < SHORTLIST && scratch.addDistinct(doc)) distinct++;
              if (d > threshold) continue;
              histogram[d]++;
              scratch.packed[size++] = ((long) d << 32) | slot;
              if (++admitted > pool) tighten();
            }
          }
          if (done >= targetProbe && distinct >= SHORTLIST) break;
        }
        rerankPool((int) Math.min(pool, survivors), null);
      }
    }

    /** Decoded vector values over this thread's own record and coarse slices. */
    final class Values extends FloatVectorValues {
      private final RandomAccessInput records = section(1), coarse = section(2);
      private final byte[] record = new byte[recordLen];
      private final float[] rotated = new float[dim], value = new float[dim];

      /** Opens this view's slices. */
      Values() throws IOException {}

      /** Copies a staged row while replacing merge-specific metadata. */
      void copyRow(int ord, int docId, byte[] dest, int offset) throws IOException {
        loadOrdToSlot();
        records.readBytes((long) ordToSlot[ord] * recordLen, dest, offset, recordLen);
        BitUtil.VH_LE_INT.set(dest, offset + docIdOffset, docId);
        BitUtil.VH_LE_INT.set(dest, offset + CodeRecord.primaryCellOffset(docIdOffset), 0);
        coarse.readBytes(
            (long) ordToSlot[ord] * coarseBytes, dest, offset + recordLen, coarseBytes);
      }

      /** Returns the vector dimension. */
      @Override
      public int dimension() {
        return dim;
      }

      /** Returns the number of vectors. */
      @Override
      public int size() {
        return count;
      }

      /** Decodes and inverse-rotates one vector value. */
      @Override
      public float[] vectorValue(int ord) throws IOException {
        rotation.inverseRotate(rotatedValue(ord), value);
        return value;
      }

      /** Decodes one normalized vector value in the rotated space. */
      private float[] rotatedValue(int ord) throws IOException {
        loadOrdToSlot();
        records.readBytes((long) ordToSlot[ord] * recordLen, record, 0, recordLen);
        fine.decode(record, 0, rotated);
        VectorUtil.l2normalize(rotated, false);
        return rotated;
      }

      /** Maps a vector ordinal to its document ID. */
      @Override
      public int ordToDoc(int ord) {
        if (count == segmentMaxDoc) return ord;
        loadMappings();
        return ordToDoc[ord];
      }

      /** Creates an independent vector-values view. */
      @Override
      public FloatVectorValues copy() throws IOException {
        return new Values();
      }

      /** Creates an iterator over vector ordinals and documents. */
      @Override
      public DocIndexIterator iterator() {
        if (count == segmentMaxDoc) return createDenseIterator();
        loadMappings();
        return createSparseIterator();
      }

      /** Loads ordinal mappings for sparse fields from methods that cannot throw IOException. */
      private void loadMappings() {
        try {
          loadOrdinalMappings();
        } catch (IOException ioe) {
          throw new UncheckedIOException(ioe);
        }
      }

      /**
       * Creates an exact scorer over decoded vectors. The rotation is orthogonal, so the query is
       * rotated once instead of inverse-rotating every scored vector.
       */
      @Override
      public VectorScorer scorer(float[] target) throws IOException {
        Values values = new Values();
        DocIndexIterator iterator = values.iterator();
        float[] query = new float[dim];
        rotation.rotate(target, query);
        return new VectorScorer() {
          /** Scores the iterator's current vector. */
          @Override
          public float score() throws IOException {
            return similarity.compare(query, values.rotatedValue(iterator.index()));
          }

          /** Returns the scorer's shared vector iterator. */
          @Override
          public DocIdSetIterator iterator() {
            return iterator;
          }
        };
      }
    }
  }

  private static final class Scratch {
    static final ThreadLocal<Scratch> LOCAL = ThreadLocal.withInitial(Scratch::new);

    int[] histogram = new int[0], prefix = new int[0], distances = new int[0];
    int[] candidates = new int[0], coarse = new int[0];
    int[] kept = new int[ADMIT_BLOCK], shortlist = new int[SHORTLIST];
    int[] dedupKeys = new int[DEDUP_MASK + 1], dedupStamps = new int[DEDUP_MASK + 1];
    long[] packed = new long[0], ordered = new long[0];
    byte[] bytes = new byte[0];
    float[] exact = new float[0], scores = new float[0];
    int dedupStamp;

    /** Grows a reusable float buffer when needed. */
    static float[] floats(float[] array, int n) {
      return array.length >= n ? array : new float[ArrayUtil.oversize(n, Float.BYTES)];
    }

    /** Starts a new generation for the fixed-size dedup table. */
    void newDedup() {
      if (++dedupStamp == 0) {
        Arrays.fill(dedupStamps, 0);
        dedupStamp = 1;
      }
    }

    /** Adds a document ID if it has not appeared in this generation. */
    boolean addDistinct(int doc) {
      int h = (doc * 0x9E3779B9) >>> 1 & DEDUP_MASK;
      for (; dedupStamps[h] == dedupStamp; h = (h + 1) & DEDUP_MASK) {
        if (dedupKeys[h] == doc) return false;
      }
      dedupStamps[h] = dedupStamp;
      dedupKeys[h] = doc;
      return true;
    }
  }

  /** Searches a float vector field. */
  @Override
  public void search(String name, float[] target, KnnCollector collector, AcceptDocs acceptDocs)
      throws IOException {
    Field field = field(name);
    if (field != null) field.new Search(target, collector).run(acceptDocs);
  }

  /** Opens a nonempty field by name. */
  Field field(String name) throws IOException {
    Field field = fields.get(name);
    return field == null || field.count == 0 ? null : field.open();
  }

  /** Returns whether fine records are read with io_uring instead of the mapped input. */
  boolean readsFineDirectly() {
    return uring != null;
  }

  /** Returns the segment name used by hot-start tracking. */
  String segment() {
    return segment;
  }

  /** Returns decoded float vector values for a field. */
  @Override
  public FloatVectorValues getFloatVectorValues(String name) throws IOException {
    Field field = fields.get(name);
    return field == null ? null : field.open().new Values();
  }

  /** Reports that byte vectors are unsupported. */
  @Override
  public ByteVectorValues getByteVectorValues(String field) {
    return null;
  }

  /** Reports that float16 vectors are unsupported. */
  @Override
  public Float16VectorValues getFloat16VectorValues(String field) {
    return null;
  }

  /** Rejects byte-vector search. */
  @Override
  public void search(String field, byte[] target, KnnCollector collector, AcceptDocs acceptDocs) {
    throw new UnsupportedOperationException("IVFasterEvo supports only FLOAT32 vectors");
  }

  /** Rejects float16-vector search. */
  @Override
  public void search(String field, short[] target, KnnCollector collector, AcceptDocs acceptDocs) {
    throw new UnsupportedOperationException("IVFasterEvo supports only FLOAT32 vectors");
  }

  /** Verifies the data file checksum. */
  @Override
  public void checkIntegrity(MergePolicy.OneMerge merge) throws IOException {
    CodecUtil.checksumEntireFile(data);
  }

  /** Closes the segment data input and optionally reports scan engagement. */
  @Override
  @SuppressForbidden(reason = "opt-in benchmark engagement report")
  public void close() throws IOException {
    if (REPORT_ENGAGEMENT) {
      long queries = SCAN_QUERIES.get(), per = Math.max(1, queries);
      System.out.println(
          "[ivfaster-evo] queries="
              + queries
              + " slotsScanned/query="
              + SCANNED_SLOTS.get() / per
              + " cellsProbed/query="
              + PROBED_CELLS.get() / per);
    }
    IOUtils.close(data, uring);
  }
}
