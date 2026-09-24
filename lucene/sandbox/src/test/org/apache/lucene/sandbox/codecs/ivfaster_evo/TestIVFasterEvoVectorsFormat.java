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

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.sandbox.codecs.ivfaster_evo.IVFasterEvoVectorsFormat.FineTier;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.VectorUtil;

@ThreadLeakFilters(defaultFilters = true, filters = IvfasterBuildThreadsFilter.class)
public class TestIVFasterEvoVectorsFormat extends LuceneTestCase {
  private static final int DIM = 24;

  private static IndexWriterConfig config(FineTier tier, int nprobe, boolean sorted) {
    IndexWriterConfig config =
        newIndexWriterConfig()
            .setCodec(
                TestUtil.alwaysKnnVectorsFormat(new IVFasterEvoVectorsFormat(8, nprobe, 2, tier)));
    if (sorted) config.setIndexSort(new Sort(new SortField("sort", SortField.Type.LONG)));
    return config;
  }

  private static float[] randomVector() {
    float[] vector = new float[DIM];
    for (int d = 0; d < DIM; d++) vector[d] = random().nextFloat() - 0.5f;
    return vector;
  }

  public void testRoundTripDeletesSortSparseFieldsAndMerge() throws Exception {
    for (FineTier tier : FineTier.values()) {
      for (VectorSimilarityFunction similarity : VectorSimilarityFunction.values()) {
        try (Directory dir = newDirectory();
            IndexWriter writer = new IndexWriter(dir, config(tier, 8, true))) {
          for (int i = 0; i < 240; i++) {
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
            doc.add(new NumericDocValuesField("sort", 240 - i));
            float[] vector = randomVector();
            VectorUtil.l2normalize(vector);
            if (i % 7 != 0) doc.add(new KnnFloatVectorField("v", vector, similarity));
            if (i % 4 == 0) doc.add(new KnnFloatVectorField("other", vector, similarity));
            writer.addDocument(doc);
            if (i % 60 == 59) writer.commit();
          }
          for (int i = 0; i < 240; i += 13) {
            writer.deleteDocuments(new Term("id", Integer.toString(i)));
          }
          for (int stage = 0; stage < 2; stage++) {
            if (stage == 1) writer.forceMerge(1);
            try (DirectoryReader reader = DirectoryReader.open(writer)) {
              for (var leaf : reader.leaves()) {
                for (String field : List.of("v", "other")) {
                  assertSearch(leaf.reader(), field, similarity, tier, null);
                }
              }
            }
          }
          writer.commit();
          TestUtil.checkIndex(dir);
        }
      }
    }
  }

  public void testFilteredWalkAndExactSmallFilter() throws Exception {
    for (FineTier tier : FineTier.values()) {
      try (Directory dir = newDirectory();
          IndexWriter writer = new IndexWriter(dir, config(tier, 2, false))) {
        for (int i = 0; i < 3000; i++) {
          Document doc = new Document();
          doc.add(new KnnFloatVectorField("v", randomVector(), VectorSimilarityFunction.COSINE));
          writer.addDocument(doc);
        }
        writer.forceMerge(1);
        try (DirectoryReader reader = DirectoryReader.open(writer)) {
          LeafReader leaf = reader.leaves().get(0).reader();
          // Half the documents: far above the exact bound, so the widening walk runs.
          FixedBitSet half = new FixedBitSet(leaf.maxDoc());
          for (int doc = 0; doc < leaf.maxDoc(); doc += 2) half.set(doc);
          assertSearch(leaf, "v", VectorSimilarityFunction.COSINE, tier, half);
          // A handful: reranked whole, so the result is exact to the tier's precision.
          FixedBitSet few = new FixedBitSet(leaf.maxDoc());
          for (int doc = 5; doc < leaf.maxDoc(); doc += 97) few.set(doc);
          assertSearch(leaf, "v", VectorSimilarityFunction.COSINE, tier, few);
          TopKnnCollector none = new TopKnnCollector(10, Integer.MAX_VALUE);
          leaf.searchNearestVectors("v", randomVector(), none, accepting(new FixedBitSet(3000)));
          assertEquals(0, none.topDocs().scoreDocs.length);
        }
      }
    }
  }

  public void testExactScorerMatchesUnrotatedFp32Vectors() throws Exception {
    for (VectorSimilarityFunction similarity :
        new VectorSimilarityFunction[] {
          VectorSimilarityFunction.COSINE, VectorSimilarityFunction.EUCLIDEAN
        }) {
      try (Directory dir = newDirectory();
          IndexWriter writer = new IndexWriter(dir, config(FineTier.FP32, 8, false))) {
        List<float[]> vectors = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
          float[] vector = VectorUtil.l2normalize(randomVector());
          vectors.add(vector);
          Document doc = new Document();
          doc.add(new KnnFloatVectorField("v", vector, similarity));
          doc.add(new NumericDocValuesField("id", i));
          writer.addDocument(doc);
        }
        writer.forceMerge(1);
        try (DirectoryReader reader = DirectoryReader.open(writer)) {
          LeafReader leaf = getOnlyLeafReader(reader);
          float[] target = randomVector();
          var scorer = leaf.getFloatVectorValues("v").scorer(target);
          var ids = leaf.getNumericDocValues("id");
          DocIdSetIterator it = scorer.iterator();
          for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
            assertTrue(ids.advanceExact(doc));
            float[] vector = vectors.get((int) ids.longValue());
            assertEquals(similarity.compare(target, vector), scorer.score(), 1e-4f);
          }
        }
      }
    }
  }

  public void testDirectFineReadsMatchMappedReads() throws Exception {
    try (Directory dir = newFSDirectory(createTempDir())) {
      // One flushed, non-compound segment, so the data file can be opened directly.
      IndexWriterConfig config =
          config(FineTier.INT8, 8, false)
              .setUseCompoundFile(false)
              .setMergePolicy(NoMergePolicy.INSTANCE)
              .setMaxBufferedDocs(10_000)
              .setRAMBufferSizeMB(256);
      try (IndexWriter writer = new IndexWriter(dir, config)) {
        for (int i = 0; i < 2000; i++) {
          Document doc = new Document();
          doc.add(new KnnFloatVectorField("v", randomVector(), VectorSimilarityFunction.COSINE));
          writer.addDocument(doc);
        }
      }
      float[] target = randomVector();
      var mapped = search(dir, target);
      String previous = System.getProperty(IVFasterEvoVectorsReader.URING_FINE_PROPERTY);
      System.setProperty(IVFasterEvoVectorsReader.URING_FINE_PROPERTY, "true");
      try {
        try (DirectoryReader reader = DirectoryReader.open(dir)) {
          var codecReader = (CodecReader) getOnlyLeafReader(reader);
          var evo =
              (IVFasterEvoVectorsReader) codecReader.getVectorReader().unwrapReaderForField("v");
          assumeTrue("io_uring is unavailable here", evo.readsFineDirectly());
        }
        var direct = search(dir, target);
        assertEquals(mapped.length, direct.length);
        for (int i = 0; i < mapped.length; i++) {
          assertEquals(mapped[i].doc, direct[i].doc);
          assertEquals(mapped[i].score, direct[i].score, 0f);
        }
      } finally {
        if (previous == null) System.clearProperty(IVFasterEvoVectorsReader.URING_FINE_PROPERTY);
        else System.setProperty(IVFasterEvoVectorsReader.URING_FINE_PROPERTY, previous);
      }
    }
  }

  private static ScoreDoc[] search(Directory dir, float[] target) throws Exception {
    try (DirectoryReader reader = DirectoryReader.open(dir)) {
      TopKnnCollector collector =
          new TopKnnCollector(
              10, Integer.MAX_VALUE, new IVFasterEvoVectorsFormat.SearchStrategy(8));
      getOnlyLeafReader(reader).searchNearestVectors("v", target, collector, null);
      return collector.topDocs().scoreDocs;
    }
  }

  public void testGlobalRerankMatchesSegmentRerankOnOneSegment() throws Exception {
    try (Directory dir = newDirectory();
        IndexWriter writer = new IndexWriter(dir, config(FineTier.INT8, 8, false))) {
      for (int i = 0; i < 3000; i++) {
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("v", randomVector(), VectorSimilarityFunction.COSINE));
        writer.addDocument(doc);
      }
      writer.forceMerge(1);
      try (DirectoryReader reader = DirectoryReader.open(writer)) {
        var searcher = newSearcher(reader);
        var strategy = new IVFasterEvoVectorsFormat.SearchStrategy(4, 0.9f);
        float[] target = randomVector();
        TopDocs segment =
            searcher.search(new KnnFloatVectorQuery("v", target, 10, null, strategy), 10);
        TopDocs global =
            searcher.search(new IVFasterEvoKnnQuery("v", target, 10, null, strategy), 10);
        assertEquals(segment.scoreDocs.length, global.scoreDocs.length);
        for (int i = 0; i < segment.scoreDocs.length; i++) {
          assertEquals(segment.scoreDocs[i].doc, global.scoreDocs[i].doc);
          assertEquals(segment.scoreDocs[i].score, global.scoreDocs[i].score, 0f);
        }
      }
    }
  }

  public void testGlobalRerankAcrossSegmentsIsExactWhenEverythingIsReranked() throws Exception {
    try (Directory dir = newDirectory();
        IndexWriter writer = new IndexWriter(dir, config(FineTier.FP32, 8, false))) {
      List<float[]> vectors = new ArrayList<>();
      for (int i = 0; i < 150; i++) {
        float[] vector = VectorUtil.l2normalize(randomVector());
        vectors.add(vector);
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("v", vector, VectorSimilarityFunction.COSINE));
        doc.add(new NumericDocValuesField("id", i));
        writer.addDocument(doc);
        if (i % 50 == 49) writer.commit();
      }
      try (DirectoryReader reader = DirectoryReader.open(writer)) {
        float[] target = VectorUtil.l2normalize(randomVector());
        var strategy = new IVFasterEvoVectorsFormat.SearchStrategy(8, 1f);
        TopDocs hits =
            newSearcher(reader)
                .search(new IVFasterEvoKnnQuery("v", target, 10, null, strategy), 10);
        List<Float> expected = new ArrayList<>();
        for (float[] vector : vectors) {
          expected.add(VectorSimilarityFunction.COSINE.compare(target, vector));
        }
        expected.sort(Comparator.reverseOrder());
        assertEquals(10, hits.scoreDocs.length);
        for (int i = 0; i < 10; i++) assertEquals(expected.get(i), hits.scoreDocs[i].score, 1e-4f);
      }
    }
  }

  public void testFilteredGlobalRerankAcrossSegmentsIsExactWhenEverythingIsReranked()
      throws Exception {
    try (Directory dir = newDirectory();
        IndexWriter writer = new IndexWriter(dir, config(FineTier.FP32, 8, false))) {
      List<float[]> vectors = new ArrayList<>();
      for (int i = 0; i < 3000; i++) {
        float[] vector = VectorUtil.l2normalize(randomVector());
        vectors.add(vector);
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("v", vector, VectorSimilarityFunction.COSINE));
        doc.add(new NumericDocValuesField("id", i));
        doc.add(new StringField("keep", i % 7 == 0 ? "y" : "n", Field.Store.NO));
        writer.addDocument(doc);
        if (i % 1000 == 999) writer.commit();
      }
      try (DirectoryReader reader = DirectoryReader.open(writer)) {
        float[] target = VectorUtil.l2normalize(randomVector());
        var filter = new TermQuery(new Term("keep", "y"));
        var strategy = new IVFasterEvoVectorsFormat.SearchStrategy(8, 1f);
        var searcher = newSearcher(reader);
        TopDocs hits =
            searcher.search(new IVFasterEvoKnnQuery("v", target, 10, filter, strategy), 10);
        List<Float> expected = new ArrayList<>();
        for (int i = 0; i < vectors.size(); i += 7) {
          expected.add(VectorSimilarityFunction.COSINE.compare(target, vectors.get(i)));
        }
        expected.sort(Comparator.reverseOrder());
        assertEquals(10, hits.scoreDocs.length);
        // Scores equal to the filtered brute force also show no unfiltered document leaked in.
        for (int i = 0; i < 10; i++) assertEquals(expected.get(i), hits.scoreDocs[i].score, 1e-4f);
      }
    }
  }

  public void testProbeMarginMustBeANumberInRange() {
    for (float margin : new float[] {Float.NaN, 0f, -1f, 1.5f}) {
      expectThrows(
          IllegalArgumentException.class,
          () -> new IVFasterEvoVectorsFormat.SearchStrategy(8, margin));
    }
  }

  public void testMergeDropsADonorWithMoreCellsThanSurvivingVectors() throws Exception {
    try (Directory dir = newDirectory();
        IndexWriter writer = new IndexWriter(dir, config(FineTier.INT8, 8, false))) {
      for (int i = 0; i < 60; i++) {
        Document doc = new Document();
        doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
        doc.add(new KnnFloatVectorField("v", randomVector(), VectorSimilarityFunction.COSINE));
        writer.addDocument(doc);
        if (i == 39) writer.commit();
      }
      // Both segments keep fewer vectors than the eight cells the larger one was clustered into.
      for (int i = 0; i < 60; i++) {
        if (i % 20 != 0) writer.deleteDocuments(new Term("id", Integer.toString(i)));
      }
      writer.forceMerge(1);
      try (DirectoryReader reader = DirectoryReader.open(writer)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        assertEquals(3, leaf.numDocs());
        var evo =
            (IVFasterEvoVectorsReader)
                ((org.apache.lucene.index.CodecReader) leaf)
                    .getVectorReader()
                    .unwrapReaderForField("v");
        assertEquals(3, evo.field("v").nlist);
        assertSearch(leaf, "v", VectorSimilarityFunction.COSINE, FineTier.INT8, null);
      }
    }
  }

  /** Compares a search with brute force over the field's own (unit) vectors. */
  private static void assertSearch(
      LeafReader reader,
      String field,
      VectorSimilarityFunction similarity,
      FineTier tier,
      FixedBitSet filter)
      throws Exception {
    float[] target = randomVector();
    VectorUtil.l2normalize(target);
    var values = reader.getFloatVectorValues(field);
    if (values == null) return; // a small random segment may hold no vectors for this field
    Bits live = reader.getLiveDocs();
    List<float[]> expected = new ArrayList<>(); // {doc, score}
    var iterator = values.iterator();
    for (int doc = iterator.nextDoc();
        doc != DocIdSetIterator.NO_MORE_DOCS;
        doc = iterator.nextDoc()) {
      float[] vector = values.vectorValue(iterator.index());
      assertEquals(1f, VectorUtil.dotProduct(vector, vector), 1e-3f);
      if ((live == null || live.get(doc)) && (filter == null || filter.get(doc))) {
        expected.add(new float[] {doc, similarity.compare(target, vector)});
      }
    }
    expected.sort(Comparator.comparingDouble(hit -> -hit[1]));
    TopKnnCollector collector =
        new TopKnnCollector(
            10, Integer.MAX_VALUE, new IVFasterEvoVectorsFormat.SearchStrategy(8, 1f));
    reader.searchNearestVectors(
        field,
        target,
        collector,
        filter == null ? AcceptDocs.fromLiveDocs(live, reader.maxDoc()) : accepting(filter));
    var hits = collector.topDocs().scoreDocs;
    assertEquals(Math.min(10, expected.size()), hits.length);
    Set<Integer> truth = new HashSet<>();
    for (int i = 0; i < hits.length; i++) truth.add((int) expected.get(i)[0]);
    int found = 0;
    for (var hit : hits) {
      assertTrue(live == null || live.get(hit.doc));
      assertTrue(filter == null || filter.get(hit.doc));
      if (truth.contains(hit.doc)) found++;
    }
    // Every cell is probed (or the filter walk covers the index), so FP32 is near exact and U8
    // loses only what eight bits lose.
    int floor = tier == FineTier.FP32 ? hits.length - 1 : (int) (0.7 * hits.length);
    assertTrue(found + " of " + hits.length, found >= floor);
    if (tier == FineTier.FP32 && hits.length > 0) {
      assertEquals(expected.get(0)[1], hits[0].score, 1e-4f);
    }
  }

  private static AcceptDocs accepting(FixedBitSet accepted) {
    return new AcceptDocs() {
      @Override
      public Bits bits() {
        return accepted;
      }

      @Override
      public int cost() {
        return accepted.cardinality();
      }

      @Override
      public DocIdSetIterator iterator() {
        return new BitSetIterator(accepted, accepted.cardinality());
      }
    };
  }
}
