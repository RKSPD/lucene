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
package org.apache.lucene.tests.search;

import com.carrotsearch.randomizedtesting.generators.RandomNumbers;
import java.io.IOException;
import java.util.Random;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.AbstractDocIdSetIterator;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FilterWeight;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TwoPhaseIterator;
import org.apache.lucene.search.Weight;

/** A {@link Query} that adds random approximations to its scorers. */
public class RandomApproximationQuery extends Query {

  private final Query query;
  private final Random random;

  public RandomApproximationQuery(Query query, Random random) {
    this.query = query;
    this.random = random;
  }

  @Override
  public Query rewrite(IndexSearcher indexSearcher) throws IOException {
    final Query rewritten = query.rewrite(indexSearcher);
    if (rewritten != query) {
      return new RandomApproximationQuery(rewritten, random);
    }
    return super.rewrite(indexSearcher);
  }

  @Override
  public void visit(QueryVisitor visitor) {
    query.visit(visitor);
  }

  @Override
  public boolean equals(Object other) {
    return sameClassAs(other) && query.equals(((RandomApproximationQuery) other).query);
  }

  @Override
  public int hashCode() {
    return 31 * classHash() + query.hashCode();
  }

  @Override
  public String toString(String field) {
    return query.toString(field);
  }

  @Override
  public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost)
      throws IOException {
    final Weight weight = query.createWeight(searcher, scoreMode, boost);
    return new RandomApproximationWeight(weight, new Random(random.nextLong()));
  }

  private static class RandomApproximationWeight extends FilterWeight {

    private final Random random;

    RandomApproximationWeight(Weight weight, Random random) {
      super(weight);
      this.random = random;
    }

    @Override
    public ScorerSupplier scorerSupplier(LeafReaderContext context) throws IOException {
      final Scorer scorer;
      final var scorerSupplier = in.scorerSupplier(context);
      if (scorerSupplier == null) {
        return null;
      } else {
        final var subScorer = scorerSupplier.get(Long.MAX_VALUE);
        scorer = new RandomApproximationScorer(subScorer, new Random(random.nextLong()));
      }
      return new DefaultScorerSupplier(scorer);
    }
  }

  private static class RandomApproximationScorer extends Scorer {

    private final Scorer scorer;
    private final RandomTwoPhaseView twoPhaseView;

    RandomApproximationScorer(Scorer scorer, Random random) {
      this.scorer = scorer;
      this.twoPhaseView = new RandomTwoPhaseView(random, scorer.iterator());
    }

    @Override
    public TwoPhaseIterator twoPhaseIterator() {
      return twoPhaseView;
    }

    @Override
    public float score() throws IOException {
      return scorer.score();
    }

    @Override
    public int advanceShallow(int target) throws IOException {
      if (scorer.docID() > target && twoPhaseView.approximation().docID() != scorer.docID()) {
        // The random approximation can return doc ids that are not present in the underlying
        // scorer. These additional doc ids are always *before* the next matching doc so we
        // cannot use them to shallow advance the main scorer which is already ahead.
        target = scorer.docID();
      }
      return scorer.advanceShallow(target);
    }

    @Override
    public float getMaxScore(int upTo) throws IOException {
      return scorer.getMaxScore(upTo);
    }

    @Override
    public int docID() {
      return twoPhaseView.approximation().docID();
    }

    @Override
    public DocIdSetIterator iterator() {
      return TwoPhaseIterator.asDocIdSetIterator(twoPhaseView);
    }
  }

  /**
   * A wrapper around a {@link DocIdSetIterator} that matches the same documents, but introduces
   * false positives that need to be verified via {@link TwoPhaseIterator#matches()}.
   */
  public static class RandomTwoPhaseView extends TwoPhaseIterator {

    private final DocIdSetIterator disi;
    private int lastDoc = -1;
    private final float randomMatchCost;

    /** Constructor. */
    public RandomTwoPhaseView(Random random, DocIdSetIterator disi) {
      super(new RandomApproximation(random, disi));
      this.disi = disi;
      this.randomMatchCost = random.nextFloat() * 200; // between 0 and 200
    }

    @Override
    public boolean matches() throws IOException {
      if (approximation.docID() == -1 || approximation.docID() == DocIdSetIterator.NO_MORE_DOCS) {
        throw new AssertionError(
            "matches() should not be called on doc ID " + approximation.docID());
      }
      if (lastDoc == approximation.docID()) {
        throw new AssertionError(
            "matches() has been called twice on doc ID " + approximation.docID());
      }
      lastDoc = approximation.docID();
      return approximation.docID() == disi.docID();
    }

    @Override
    public float matchCost() {
      return randomMatchCost;
    }

    @Override
    public int docIDRunEnd() throws IOException {
      if (approximation.docID() == disi.docID()) {
        return disi.docIDRunEnd();
      }
      return super.docIDRunEnd();
    }
  }

  private static class RandomApproximation extends AbstractDocIdSetIterator {

    private final Random random;
    private final DocIdSetIterator disi;

    public RandomApproximation(Random random, DocIdSetIterator disi) {
      this.random = random;
      this.disi = disi;
    }

    @Override
    public int nextDoc() throws IOException {
      return advance(doc + 1);
    }

    @Override
    public int advance(int target) throws IOException {
      if (disi.docID() < target) {
        disi.advance(target);
      }
      if (disi.docID() == NO_MORE_DOCS) {
        return doc = NO_MORE_DOCS;
      }
      return doc = RandomNumbers.randomIntBetween(random, target, disi.docID());
    }

    @Override
    public long cost() {
      return disi.cost();
    }
  }
}
