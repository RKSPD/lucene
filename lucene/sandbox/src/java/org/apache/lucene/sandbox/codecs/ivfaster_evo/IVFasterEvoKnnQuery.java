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
import java.util.Objects;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BulkScorer;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;

/**
 * IVFasterEvo query variant that materializes dense filters for filter-aware vector search.
 *
 * @lucene.experimental
 */
public final class IVFasterEvoKnnQuery extends KnnFloatVectorQuery {
  private final Query denseFilter;
  private final Weight filterWeight;

  /** Creates an IVFaster query with optional dense filtering. */
  public IVFasterEvoKnnQuery(String field, float[] target, int k, Query filter, int numProbes) {
    super(field, target, k, null, new IVFasterEvoVectorsFormat.SearchStrategy(numProbes));
    this.denseFilter = filter;
    this.filterWeight = null;
  }

  /** Carries a rewritten filter weight into segment search. */
  private IVFasterEvoKnnQuery(IVFasterEvoKnnQuery query, Weight filterWeight) {
    super(query.field, query.target, query.k, null, query.searchStrategy);
    this.denseFilter = query.denseFilter;
    this.filterWeight = filterWeight;
  }

  /** Rewrites dense filters so IVFaster can choose the filtered search path. */
  @Override
  public Query rewrite(IndexSearcher searcher) throws IOException {
    if (denseFilter == null || filterWeight != null) return super.rewrite(searcher);
    var both = new BooleanQuery.Builder();
    both.add(denseFilter, Occur.FILTER).add(new FieldExistsQuery(field), Occur.FILTER);
    Query rewritten = searcher.rewrite(both.build());
    if (rewritten.getClass() == MatchNoDocsQuery.class) return rewritten;
    if ((rewritten instanceof BooleanQuery b && b.clauses().size() > 1) == false) {
      return new KnnFloatVectorQuery(field, target, k, denseFilter, searchStrategy)
          .rewrite(searcher);
    }
    Weight weight = rewritten.createWeight(searcher, ScoreMode.COMPLETE_NO_SCORES, 1f);
    return new IVFasterEvoKnnQuery(this, weight).rewrite(searcher);
  }

  /** Materializes the dense filter before running approximate vector search. */
  @Override
  protected TopDocs approximateSearch(
      LeafReaderContext context, AcceptDocs live, int limit, KnnCollectorManager manager)
      throws IOException {
    if (filterWeight == null) return super.approximateSearch(context, live, limit, manager);
    BulkScorer scorer = filterWeight.bulkScorer(context);
    if (scorer == null) return TopDocsCollector.EMPTY_TOPDOCS;
    int maxDoc = context.reader().maxDoc();
    FixedBitSet accepted = new FixedBitSet(maxDoc);
    LeafCollector collector =
        new LeafCollector() {
          @Override
          public void setScorer(Scorable scorer) {}

          @Override
          public void collect(int doc) {
            accepted.set(doc);
          }
        };
    scorer.score(collector, live.bits(), 0, maxDoc);
    int n = accepted.cardinality();
    if (n <= k) return exactSearch(context, new BitSetIterator(accepted, n), null);
    var it = AcceptDocs.fromIteratorSupplier(() -> new BitSetIterator(accepted, n), null, maxDoc);
    return super.approximateSearch(context, it, n + 1, manager);
  }

  /** Formats the query and its dense filter. */
  @Override
  public String toString(String field) {
    return "IVFasterEvoKnnQuery(" + super.toString(field) + ", filter=" + denseFilter + ")";
  }

  /** Compares the base query and dense filter. */
  @Override
  public boolean equals(Object other) {
    return super.equals(other)
        && Objects.equals(denseFilter, ((IVFasterEvoKnnQuery) other).denseFilter);
  }

  /** Hashes the base query and dense filter. */
  @Override
  public int hashCode() {
    return 31 * super.hashCode() + Objects.hashCode(denseFilter);
  }
}
