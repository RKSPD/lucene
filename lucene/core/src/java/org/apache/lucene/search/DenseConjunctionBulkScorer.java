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
package org.apache.lucene.search;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import org.apache.lucene.internal.vectorization.BitSetConjunctionSupport;
import org.apache.lucene.internal.vectorization.VectorizationProvider;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.MathUtil;

/**
 * BulkScorer implementation of {@link ConjunctionScorer} that is specialized for dense clauses.
 * Whenever sensible, it intersects clauses by loading their matches into a bit set and computing
 * the intersection of clauses by and-ing these bit sets.
 */
final class DenseConjunctionBulkScorer extends BulkScorer {

  // Small groups use the direct range intersection to avoid SIMD setup overhead. Larger groups are
  // capped so that the density threshold is checked regularly.
  private static final int MIN_MATERIALIZED_FILTER_GROUP_SIZE = 4;
  private static final int MAX_MATERIALIZED_FILTER_GROUP_SIZE = 8;
  private static final BitSetConjunctionSupport BIT_SET_CONJUNCTION_SUPPORT =
      VectorizationProvider.getInstance().getBitSetConjunctionSupport();

  private record DisiWrapper(DocIdSetIterator approximation, TwoPhaseIterator twoPhase) {
    DisiWrapper(DocIdSetIterator iterator) {
      this(iterator, null);
    }

    DisiWrapper(TwoPhaseIterator twoPhase) {
      this(twoPhase.approximation(), twoPhase);
    }

    int docID() {
      return approximation().docID();
    }

    int docIDRunEnd() throws IOException {
      if (twoPhase() == null) {
        return approximation().docIDRunEnd();
      } else {
        return twoPhase().docIDRunEnd();
      }
    }

    void intoBitSet(int upTo, FixedBitSet bitSet, int offset) throws IOException {
      if (twoPhase() == null) {
        approximation().intoBitSet(upTo, bitSet, offset);
      } else {
        twoPhase().intoBitSet(upTo, bitSet, offset);
      }
    }

    void andIntoBitSet(int upTo, FixedBitSet bitSet, FixedBitSet scratch, int offset)
        throws IOException {
      approximation().andIntoBitSet(upTo, bitSet, scratch, offset);
    }

    FixedBitSet fixedBitSet() {
      return twoPhase() == null ? BitSetIterator.getFixedBitSetOrNull(approximation()) : null;
    }
  }

  // Use a small-ish window size to make sure that we can take advantage of gaps in the postings of
  // clauses that are not leading iteration.
  static final int WINDOW_SIZE = 4096;
  // Only use bit sets to compute the intersection if more than 1/32th of the docs are expected to
  // match. Experiments suggested that values that are a bit higher than this would work better, but
  // we're erring on the conservative side.
  static final int DENSITY_THRESHOLD_INVERSE = Long.SIZE / 2;

  private final int maxDoc;
  private final List<DisiWrapper> iterators;
  private final boolean materializedFilters;
  private final SimpleScorable scorable;

  private final FixedBitSet windowMatches = new FixedBitSet(WINDOW_SIZE);
  private final FixedBitSet clauseWindowMatches = new FixedBitSet(WINDOW_SIZE);
  private final List<DisiWrapper> windowClauses = new ArrayList<>();
  // Reused by the tiered bit-set path. Plain iterators run first, then cheap two-phase masks, then
  // expensive per-document confirmations.
  private final List<DisiWrapper> bulkClauses = new ArrayList<>();
  private final List<DisiWrapper> maskClauses = new ArrayList<>();
  private final List<DisiWrapper> confirmClauses = new ArrayList<>();
  // Reused by the leap-frog path.
  private final List<DocIdSetIterator> windowApproximations = new ArrayList<>();
  private final List<TwoPhaseIterator> windowTwoPhases = new ArrayList<>();
  private final FixedBitSet[] materializedFilterGroup =
      new FixedBitSet[MAX_MATERIALIZED_FILTER_GROUP_SIZE];

  static DenseConjunctionBulkScorer of(List<Scorer> filters, int maxDoc, float constantScore) {
    List<DocIdSetIterator> iterators = new ArrayList<>();
    List<TwoPhaseIterator> twoPhases = new ArrayList<>();
    for (Scorer filter : filters) {
      TwoPhaseIterator twoPhase = filter.twoPhaseIterator();
      if (twoPhase != null) {
        twoPhases.add(twoPhase);
      } else {
        iterators.add(filter.iterator());
      }
    }
    return new DenseConjunctionBulkScorer(iterators, twoPhases, maxDoc, constantScore);
  }

  DenseConjunctionBulkScorer(
      List<DocIdSetIterator> iterators,
      List<TwoPhaseIterator> twoPhases,
      int maxDoc,
      float constantScore) {
    if (iterators.isEmpty() && twoPhases.isEmpty()) {
      throw new IllegalArgumentException("Expected one or more iterators, got 0");
    }
    this.maxDoc = maxDoc;
    this.iterators = new ArrayList<>();
    for (DocIdSetIterator iterator : iterators) {
      this.iterators.add(new DisiWrapper(iterator));
    }
    for (TwoPhaseIterator twoPhase : twoPhases) {
      this.iterators.add(new DisiWrapper(twoPhase));
    }
    // Plain approximations before two-phase ones, so matches() only runs on docs that already
    // satisfy every approximation; within each group, cheapest approximation first (lead skipping).
    // Two-phase clauses that tie on approximation cost are further ordered by matchCost, so a cheap
    // confirmation (e.g. a doc values range) runs before an expensive one (e.g. a script) even when
    // both report the same approximation cost.
    this.iterators.sort(
        Comparator.<DisiWrapper>comparingInt(w -> w.twoPhase() == null ? 0 : 1)
            .thenComparingLong(w -> w.approximation().cost())
            .thenComparingDouble(w -> w.twoPhase() == null ? 0 : w.twoPhase().matchCost()));
    this.materializedFilters = this.iterators.stream().allMatch(w -> w.fixedBitSet() != null);
    this.scorable = new SimpleScorable();
    scorable.score = constantScore;
  }

  @Override
  public int score(LeafCollector collector, Bits acceptDocs, int min, int max) throws IOException {
    collector.setScorer(scorable);

    List<DisiWrapper> iterators = this.iterators;
    if (collector.competitiveIterator() != null) {
      iterators = new ArrayList<>(iterators);
      iterators.add(new DisiWrapper(collector.competitiveIterator()));
    }
    boolean materializedFastPath = iterators == this.iterators && materializedFilters;

    for (DisiWrapper w : iterators) {
      min = Math.max(min, w.approximation().docID());
    }

    max = Math.min(max, maxDoc);

    DisiWrapper lead = iterators.get(0);
    if (lead.docID() < min) {
      min = lead.approximation.advance(min);
    }

    while (min < max) {
      if (scorable.minCompetitiveScore > scorable.score) {
        return DocIdSetIterator.NO_MORE_DOCS;
      }
      min = scoreWindow(collector, acceptDocs, iterators, materializedFastPath, min, max);
    }

    if (lead.docID() > max) {
      return lead.docID();
    } else if (max >= maxDoc) {
      return DocIdSetIterator.NO_MORE_DOCS;
    } else {
      return max;
    }
  }

  private static int advance(FixedBitSet set, int i) {
    if (i >= WINDOW_SIZE) {
      return DocIdSetIterator.NO_MORE_DOCS;
    } else {
      return set.nextSetBit(i);
    }
  }

  private int scoreWindow(
      LeafCollector collector,
      Bits acceptDocs,
      List<DisiWrapper> iterators,
      boolean materializedFastPath,
      int min,
      int max)
      throws IOException {

    // Advance all iterators to the first doc that is greater than or equal to min. This is
    // important as this is the only place where we can take advantage of a large gap between
    // consecutive matches in any clause.
    for (DisiWrapper w : iterators) {
      if (w.docID() >= min) {
        min = w.docID();
      } else {
        min = w.approximation().advance(min);
      }
      if (min >= max) {
        return min;
      }
    }

    // Partition clauses of the conjunction into:
    //  - clauses that don't fully match the first half of the window and get evaluated via
    // #loadIntoBitSet or leaf-frog,
    //  - other clauses that are used to compute the greatest possible window size that they fully
    // match.
    // This logic helps align scoring windows with the natural #docIDRunEnd() boundaries of the
    // data, which helps evaluate fewer clauses per window - without allowing windows to become too
    // small thanks to the WINDOW_SIZE/2 threshold.
    int minDocIDRunEnd = max;
    final int minRunEndThreshold = MathUtil.unsignedMin(min + WINDOW_SIZE / 2, max);
    for (DisiWrapper w : iterators) {
      int docIdRunEnd = w.docIDRunEnd();
      if (w.docID() > min || docIdRunEnd < minRunEndThreshold) {
        windowClauses.add(w);
      } else {
        minDocIDRunEnd = Math.min(minDocIDRunEnd, docIdRunEnd);
      }
    }

    if (acceptDocs == null && windowClauses.isEmpty()) {
      // We have a large range of doc IDs that all match.
      collector.collectRange(min, minDocIDRunEnd);
      return minDocIDRunEnd;
    }

    int bitsetWindowMax = MathUtil.unsignedMin(minDocIDRunEnd, WINDOW_SIZE + min);

    // The bit-set path pays a fixed per-window cost (materialize the lead, applyMask, cardinality,
    // BitSetDocIdStream); it wins when most candidates must be tested anyway. But for a sparse
    // window confirmed by a two-phase clause that can be skipped, plain leap-frog -- which only
    // touches surviving docs and never materializes a bit set -- is cheaper.
    //
    // A two-phase clause whose approximation matches every doc (cost >= maxDoc, e.g. a skip-indexed
    // doc-values range, whose block iterator reports NO_MORE_DOCS) cannot be skipped and stays on
    // the bit-set path for its vectorized intoBitSet. A selective approximation (cost < maxDoc,
    // e.g. a phrase) can be skipped, so a sparse window is cheaper via leap-frog.
    boolean skippableTwoPhase = false;
    for (DisiWrapper w : windowClauses) {
      if (w.twoPhase() != null && w.approximation().cost() < maxDoc) {
        skippableTwoPhase = true;
        break;
      }
    }
    // "sparse" means the lead clause's cost estimates it matches at most 1/4 of the document
    // space -- cheap enough that leap-frog's per-survivor confirmation beats materializing a bit
    // set for the whole window.
    boolean sparse =
        windowClauses.isEmpty() == false
            && windowClauses.get(0).approximation().cost() <= (long) maxDoc / 4;

    if (skippableTwoPhase && sparse) {
      for (DisiWrapper w : windowClauses) {
        windowApproximations.add(w.approximation());
        if (w.twoPhase() != null) {
          windowTwoPhases.add(w.twoPhase());
        }
      }
      windowTwoPhases.sort(Comparator.comparingDouble(TwoPhaseIterator::matchCost));
      scoreWindowUsingLeapFrog(
          collector, acceptDocs, windowApproximations, windowTwoPhases, min, bitsetWindowMax);
      windowApproximations.clear();
      windowTwoPhases.clear();
    } else {
      scoreWindowUsingBitSet(
          collector,
          acceptDocs,
          windowClauses,
          materializedFastPath,
          min,
          bitsetWindowMax);
    }
    windowClauses.clear();

    return bitsetWindowMax;
  }

  private void scoreWindowUsingBitSet(
      LeafCollector collector,
      Bits acceptDocs,
      List<DisiWrapper> iterators,
      boolean materializedFastPath,
      int windowBase,
      int windowMax)
      throws IOException {
    assert windowMax > windowBase;
    assert windowMatches.scanIsEmpty();
    assert clauseWindowMatches.scanIsEmpty();

    if (iterators.isEmpty()) {
      // This happens if all clauses fully matched the window and there are deleted docs.
      windowMatches.set(0, windowMax - windowBase);
    } else {
      DisiWrapper lead = iterators.get(0);
      if (lead.docID() < windowBase) {
        lead.approximation().advance(windowBase);
      }
      lead.intoBitSet(windowMax, windowMatches, windowBase);
    }

    if (acceptDocs != null) {
      // Apply live docs.
      acceptDocs.applyMask(windowMatches, windowBase);
    }

    int windowSize = windowMax - windowBase;
    int threshold = windowSize / DENSITY_THRESHOLD_INVERSE;
    if (materializedFastPath) {
      scoreMaterializedWindow(collector, iterators, windowBase, windowMax, windowSize, threshold);
      windowMatches.clear();
      return;
    }

    for (int i = 1; i < iterators.size(); ++i) {
      DisiWrapper clause = iterators.get(i);
      if (clause.twoPhase() == null) {
        bulkClauses.add(clause);
      } else if (clause.twoPhase().matchCost() <= 10f) {
        maskClauses.add(clause);
      } else {
        confirmClauses.add(clause);
      }
    }

    int bulkUpTo = 0;
    int cardinality = windowMatches.cardinality();
    while (bulkUpTo < bulkClauses.size() && cardinality >= threshold) {
      DisiWrapper other = bulkClauses.get(bulkUpTo);
      FixedBitSet fixedBitSet = other.fixedBitSet();
      if (fixedBitSet != null) {
        int groupSize = 0;
        while (bulkUpTo + groupSize < bulkClauses.size()
            && groupSize < MAX_MATERIALIZED_FILTER_GROUP_SIZE) {
          FixedBitSet next = bulkClauses.get(bulkUpTo + groupSize).fixedBitSet();
          if (next == null) {
            break;
          }
          materializedFilterGroup[groupSize++] = next;
        }
        if (groupSize >= MIN_MATERIALIZED_FILTER_GROUP_SIZE) {
          BIT_SET_CONJUNCTION_SUPPORT.andBitSets(
              materializedFilterGroup, groupSize, windowBase, windowMatches, windowSize);
          for (int i = 0; i < groupSize; ++i) {
            bulkClauses.get(bulkUpTo + i).approximation().advance(windowMax);
          }
          bulkUpTo += groupSize;
        } else {
          BIT_SET_CONJUNCTION_SUPPORT.andBitSet(
              materializedFilterGroup[0], windowBase, windowMatches, 0, windowSize);
          other.approximation().advance(windowMax);
          ++bulkUpTo;
        }
        for (int i = 0; i < groupSize; ++i) {
          materializedFilterGroup[i] = null;
        }
      } else {
        if (other.docID() < windowBase) {
          other.approximation().advance(windowBase);
        }
        other.andIntoBitSet(windowMax, windowMatches, clauseWindowMatches, windowBase);
        ++bulkUpTo;
      }
      cardinality = windowMatches.cardinality();
    }

    // Cheap two-phase clauses can classify the candidate bit set in bulk, so run all of them
    // before considering per-document fallback.
    for (DisiWrapper clause : maskClauses) {
      if (clause.docID() < windowBase) {
        clause.approximation().advance(windowBase);
      }
      clause.twoPhase().applyMask(windowMax, windowMatches, windowBase);
    }

    int confirmUpTo = 0;
    cardinality = windowMatches.cardinality();
    while (confirmUpTo < confirmClauses.size() && cardinality >= threshold) {
      DisiWrapper clause = confirmClauses.get(confirmUpTo++);
      if (clause.docID() < windowBase) {
        clause.approximation().advance(windowBase);
      }
      clause.twoPhase().applyMask(windowMax, windowMatches, windowBase);
      cardinality = windowMatches.cardinality();
    }

    if (bulkUpTo < bulkClauses.size() || confirmUpTo < confirmClauses.size()) {
      // If the leading clause is sparse on this doc ID range or if the intersection became sparse
      // after applying a few clauses, we finish evaluating the intersection using the traditional
      // leap-frog approach. This proved important with a query such as "+secretary +of +state" on
      // wikibigall, where the intersection becomes sparse after intersecting "secretary" and
      // "state". As the leap-frog only visits surviving docs, two-phase clauses confirm matches()
      // here only on docs that no cheaper clause already excluded.
      advanceHead:
      for (int windowMatch = windowMatches.nextSetBit(0);
          windowMatch != DocIdSetIterator.NO_MORE_DOCS; ) {
        int doc = windowBase + windowMatch;
        // First confirm every remaining approximation is on doc...
        for (int i = bulkUpTo; i < bulkClauses.size(); ++i) {
          DocIdSetIterator other = bulkClauses.get(i).approximation();
          int otherDoc = other.docID();
          if (otherDoc < doc) {
            otherDoc = other.advance(doc);
          }
          if (doc != otherDoc) {
            windowMatch = advance(windowMatches, otherDoc - windowBase);
            continue advanceHead;
          }
        }
        for (int i = confirmUpTo; i < confirmClauses.size(); ++i) {
          DocIdSetIterator other = confirmClauses.get(i).approximation();
          int otherDoc = other.docID();
          if (otherDoc < doc) {
            otherDoc = other.advance(doc);
          }
          if (doc != otherDoc) {
            windowMatch = advance(windowMatches, otherDoc - windowBase);
            continue advanceHead;
          }
        }
        // ...then run the (more expensive) two-phase confirmations, only on surviving docs.
        for (int i = confirmUpTo; i < confirmClauses.size(); ++i) {
          if (confirmClauses.get(i).twoPhase().matches() == false) {
            windowMatch = advance(windowMatches, windowMatch + 1);
            continue advanceHead;
          }
        }
        collector.collect(doc);
        windowMatch = advance(windowMatches, windowMatch + 1);
      }
    } else {
      collector.collect(new BitSetDocIdStream(windowMatches, windowBase));
    }

    bulkClauses.clear();
    maskClauses.clear();
    confirmClauses.clear();
    windowMatches.clear();
  }

  private void scoreMaterializedWindow(
      LeafCollector collector,
      List<DisiWrapper> iterators,
      int windowBase,
      int windowMax,
      int windowSize,
      int threshold)
      throws IOException {
    int upTo = 1;
    int cardinality = windowMatches.cardinality();
    while (upTo < iterators.size() && cardinality >= threshold) {
      int groupSize = Math.min(MAX_MATERIALIZED_FILTER_GROUP_SIZE, iterators.size() - upTo);
      for (int i = 0; i < groupSize; ++i) {
        materializedFilterGroup[i] = iterators.get(upTo + i).fixedBitSet();
      }
      if (groupSize >= MIN_MATERIALIZED_FILTER_GROUP_SIZE) {
        BIT_SET_CONJUNCTION_SUPPORT.andBitSets(
            materializedFilterGroup, groupSize, windowBase, windowMatches, windowSize);
        for (int i = 0; i < groupSize; ++i) {
          iterators.get(upTo + i).approximation().advance(windowMax);
        }
        upTo += groupSize;
      } else {
        BIT_SET_CONJUNCTION_SUPPORT.andBitSet(
            materializedFilterGroup[0], windowBase, windowMatches, 0, windowSize);
        iterators.get(upTo).approximation().advance(windowMax);
        ++upTo;
      }
      for (int i = 0; i < groupSize; ++i) {
        materializedFilterGroup[i] = null;
      }
      cardinality = windowMatches.cardinality();
    }

    if (upTo < iterators.size()) {
      advanceHead:
      for (int windowMatch = windowMatches.nextSetBit(0);
          windowMatch != DocIdSetIterator.NO_MORE_DOCS; ) {
        int doc = windowBase + windowMatch;
        for (int i = upTo; i < iterators.size(); ++i) {
          DocIdSetIterator other = iterators.get(i).approximation();
          int otherDoc = other.docID();
          if (otherDoc < doc) {
            otherDoc = other.advance(doc);
          }
          if (doc != otherDoc) {
            windowMatch = advance(windowMatches, otherDoc - windowBase);
            continue advanceHead;
          }
        }
        collector.collect(doc);
        windowMatch = advance(windowMatches, windowMatch + 1);
      }
    } else {
      collector.collect(new BitSetDocIdStream(windowMatches, windowBase));
    }
  }

  // Confirm two-phase matches() only on docs that survived every approximation, without
  // materializing a window bit set. Cheaper than the bit-set path for sparse windows with a
  // skippable two-phase clause (see scoreWindow).
  private static void scoreWindowUsingLeapFrog(
      LeafCollector collector,
      Bits acceptDocs,
      List<DocIdSetIterator> approximations,
      List<TwoPhaseIterator> twoPhases,
      int min,
      int max)
      throws IOException {
    assert twoPhases.size() > 0;
    assert approximations.size() >= twoPhases.size();

    if (approximations.size() == 1) {
      // scoreWindowUsingLeapFrog is only used if there is at least one two-phase iterator, so our
      // single clause is a two-phase iterator
      assert twoPhases.size() == 1;
      DocIdSetIterator approximation = approximations.get(0);
      TwoPhaseIterator twoPhase = twoPhases.get(0);
      if (approximation.docID() < min) {
        approximation.advance(min);
      }
      for (int doc = approximation.docID(); doc < max; doc = approximation.nextDoc()) {
        if ((acceptDocs == null || acceptDocs.get(doc)) && twoPhase.matches()) {
          collector.collect(doc);
        }
      }
    } else {
      DocIdSetIterator lead1 = approximations.get(0);
      DocIdSetIterator lead2 = approximations.get(1);

      if (lead1.docID() < min) {
        lead1.advance(min);
      }

      advanceHead:
      for (int doc = lead1.docID(); doc < max; ) {
        if (acceptDocs != null && acceptDocs.get(doc) == false) {
          doc = lead1.nextDoc();
          continue;
        }
        int doc2 = lead2.docID();
        if (doc2 < doc) {
          doc2 = lead2.advance(doc);
        }
        if (doc != doc2) {
          doc = lead1.advance(Math.min(doc2, max));
          continue;
        }
        for (int i = 2; i < approximations.size(); ++i) {
          DocIdSetIterator other = approximations.get(i);
          int docN = other.docID();
          if (docN < doc) {
            docN = other.advance(doc);
          }
          if (doc != docN) {
            doc = lead1.advance(Math.min(docN, max));
            continue advanceHead;
          }
        }
        for (TwoPhaseIterator twoPhase : twoPhases) {
          if (twoPhase.matches() == false) {
            doc = lead1.nextDoc();
            continue advanceHead;
          }
        }
        collector.collect(doc);
        doc = lead1.nextDoc();
      }
    }
  }

  @Override
  public long cost() {
    return iterators.get(0).approximation().cost();
  }
}
