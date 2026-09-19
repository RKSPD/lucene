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

import java.io.IOException;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.knn.KnnSearchStrategy;

/**
 * Configures the IVFasterEvo inverted-file vector format.
 *
 * <p>{@code nlist} controls the number of cells, {@code nprobe} controls how many cells a query
 * visits, spill bits add secondary cell placements near boundaries, and the fine tier controls
 * shortlist reranking precision.
 *
 * @lucene.experimental
 */
public final class IVFasterEvoVectorsFormat extends KnnVectorsFormat {
  static final String NAME = "IVFasterEvoVectorsFormat";
  static final String META_CODEC_NAME = NAME + "Meta", DATA_CODEC_NAME = NAME + "Data";
  static final String META_EXTENSION = "ivfm", DATA_EXTENSION = "ivfd";
  static final int VERSION_START = 1, VERSION_CURRENT = 1, DIRECT_MONOTONIC_BLOCK_SHIFT = 16;
  public static final int MAX_NLIST = 0xFFFF;
  final int nlist, nprobe, spillBits;
  final FineTier fineTier;

  public enum FineTier {
    INT8,
    FP32
  }

  /** Creates the default IVFasterEvo format. */
  public IVFasterEvoVectorsFormat() {
    this(1000, 32);
  }

  /** Creates a format with custom cell and probe counts. */
  public IVFasterEvoVectorsFormat(int nlist, int nprobe) {
    this(nlist, nprobe, 3, FineTier.INT8);
  }

  /** Creates a fully configured IVFasterEvo format. */
  public IVFasterEvoVectorsFormat(int nlist, int nprobe, int spillBits, FineTier fineTier) {
    super(NAME);
    if (nlist < 1 || nlist > MAX_NLIST || nprobe < 1 || spillBits < 0 || fineTier == null)
      throw new IllegalArgumentException("nlist 1..65535, nprobe >= 1, spillBits >= 0, fineTier");
    this.nlist = nlist;
    this.nprobe = nprobe;
    this.spillBits = spillBits;
    this.fineTier = fineTier;
  }

  static final float DEFAULT_PROBE_MARGIN = 0.75f;

  public static final class SearchStrategy extends KnnSearchStrategy {
    final int numProbes;
    final float probeMargin;

    /** Creates a search strategy with the default probe margin. */
    public SearchStrategy(int numProbes) {
      this(numProbes, DEFAULT_PROBE_MARGIN);
    }

    /** Creates a search strategy with explicit probes and margin. */
    public SearchStrategy(int numProbes, float probeMargin) {
      if (numProbes < 1 || (probeMargin > 0 && probeMargin <= 1) == false)
        throw new IllegalArgumentException("require numProbes >= 1 and 0 < probeMargin <= 1");
      this.numProbes = numProbes;
      this.probeMargin = probeMargin;
    }

    /** Advances no state because IVFaster probes cells in one search pass. */
    @Override
    public void nextVectorsBlock() {}

    /** Compares probe count and margin. */
    @Override
    public boolean equals(Object other) {
      return other instanceof SearchStrategy strategy
          && strategy.numProbes == numProbes
          && Float.compare(strategy.probeMargin, probeMargin) == 0;
    }

    /** Hashes probe count and margin. */
    @Override
    public int hashCode() {
      return 31 * numProbes + Float.hashCode(probeMargin);
    }
  }

  /** Creates the segment vector writer. */
  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new IVFasterEvoVectorsWriter(state, this);
  }

  /** Opens the segment vector reader. */
  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new IVFasterEvoVectorsReader(state);
  }

  /** Returns Lucene's default maximum vector dimension. */
  @Override
  public int getMaxDimensions(String fieldName) {
    return DEFAULT_MAX_DIMENSIONS;
  }

  /** Formats the IVFasterEvo configuration. */
  @Override
  public String toString() {
    String head = NAME + "(nlist=" + nlist + " nprobe=" + nprobe + " spillBits=" + spillBits;
    return head + " fineTier=" + fineTier + ")";
  }
}
