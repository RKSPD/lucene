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
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.MathUtil;

/**
 * BulkScorer implementation of {@link ConjunctionScorer} that is specialized for dense clauses.
 * Whenever sensible, it intersects clauses by loading their matches into a bit set and computing
 * the intersection of clauses by and-ing these bit sets.
 *
 * @lucene.experimental
 */
public final class DenseConjunctionBulkScorer extends BulkScorer {

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
  }

  // Use a small-ish window size to make sure that we can take advantage of gaps in the postings of
  // clauses that are not leading iteration.
  static final int WINDOW_SIZE = 4096;
  static final int WORDS_PER_WINDOW = WINDOW_SIZE / Long.SIZE;
  // Only use bit sets to compute the intersection if more than 1/32th of the docs are expected to
  // match. Experiments suggested that values that are a bit higher than this would work better, but
  // we're erring on the conservative side.
  static final int DENSITY_THRESHOLD_INVERSE = Long.SIZE / 2;

  private final int maxDoc;
  private final List<DisiWrapper> iterators;
  private final SimpleScorable scorable;

  private final FixedBitSet windowMatches = new FixedBitSet(WINDOW_SIZE);
  private final FixedBitSet clauseWindowMatches = new FixedBitSet(WINDOW_SIZE);
  private final List<DisiWrapper> windowClauses = new ArrayList<>();

  // Tiered pipeline: bulk (plain iterators) → mask (two-phase with bulk applyMask) → confirm
  // (two-phase with per-doc matches). Each tier shrinks the survivor set before the next runs.
  private final List<DisiWrapper> tierBulk = new ArrayList<>();
  private final List<DisiWrapper> tierMask = new ArrayList<>();
  private final List<DisiWrapper> tierConfirm = new ArrayList<>();

  // Reused by the leap-frog path.
  private final List<DocIdSetIterator> windowApproximations = new ArrayList<>();
  private final List<TwoPhaseIterator> windowTwoPhases = new ArrayList<>();

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

  /**
   * Creates a dense conjunction of iterators and two-phase iterators, with a constant score. The
   * iterators are consumed by this scorer. At least one iterator is required.
   *
   * @param iterators single-phase iterators to intersect
   * @param twoPhases two-phase iterators to intersect
   * @param maxDoc maximum document ID, exclusive
   * @param constantScore score assigned to every matching document
   */
  public DenseConjunctionBulkScorer(
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
      min = scoreWindow(collector, acceptDocs, iterators, min, max);
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
      LeafCollector collector, Bits acceptDocs, List<DisiWrapper> iterators, int min, int max)
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
      scoreWindowUsingBitSet(collector, acceptDocs, windowClauses, min, bitsetWindowMax);
    }
    windowClauses.clear();

    return bitsetWindowMax;
  }

  private void scoreWindowUsingBitSet(
      LeafCollector collector,
      Bits acceptDocs,
      List<DisiWrapper> iterators,
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

    // === TIERED PIPELINE ===
    // Partition clauses into tiers by resolution speed. Each tier shrinks the survivor
    // set before the next tier runs, so expensive per-doc filters see the smallest
    // possible candidate set.
    for (int i = 1; i < iterators.size(); i++) {
      DisiWrapper w = iterators.get(i);
      if (w.twoPhase() == null) {
        tierBulk.add(w);
      } else if (w.twoPhase().matchCost() <= 10f) {
        // Cheap two-phase: block-classified doc values ranges, skip-indexed fields.
        // Their applyMask can classify whole spans without per-doc matches().
        tierMask.add(w);
      } else {
        // Expensive two-phase: slow doc values ranges, scripts, per-doc confirmation.
        tierConfirm.add(w);
      }
    }

    // Tier 1 — Bulk: plain iterators. andIntoBitSet lets the PostingsReader skip
    // fully-dense blocks at zero cost (4 long comparisons → skip 256 docs).
    // Clauses whose run covers the entire window are still skipped for free.
    // Sampled density check bails to leap-frog when the intersection becomes sparse.
    int bulkUpTo = 0;
    int densityThreshold = (windowMax - windowBase) / DENSITY_THRESHOLD_INVERSE;
    for (; bulkUpTo < tierBulk.size(); bulkUpTo++) {
      DisiWrapper other = tierBulk.get(bulkUpTo);
      if (other.docID() < windowBase) {
        other.approximation().advance(windowBase);
      }
      if (other.docIDRunEnd() >= windowMax) {
        continue;
      }
      other.andIntoBitSet(windowMax, windowMatches, clauseWindowMatches, windowBase);
      // 9 sampled POPCNTs (stride 7, coprime with 4-word block alignment),
      // extrapolate. Catches empty + sparse intersections.
      long[] bits = windowMatches.getBits();
      int sample = 0;
      for (int s = 0; s < WORDS_PER_WINDOW; s += 7) {
        sample += Long.bitCount(bits[s]);
      }
      if (sample * 7 < densityThreshold) {
        bulkUpTo++;
        break;
      }
    }

    // Tier 2 — Mask: cheap two-phase iterators (block-classified doc values ranges,
    // skip-indexed fields). Their applyMask can classify whole blocks as YES/NO without
    // per-doc confirmation, so they run without cardinality gating.
    for (int i = 0; i < tierMask.size(); i++) {
      DisiWrapper other = tierMask.get(i);
      if (other.docID() < windowBase) {
        other.approximation().advance(windowBase);
      }
      other.twoPhase().applyMask(windowMax, windowMatches, windowBase);
    }

    // Tier 3 — Confirm: expensive two-phase iterators (slow doc values, scripts).
    // These pay per-doc matches() cost, so bail to leap-frog when survivors become sparse.
    int upTo = 0;
    if (!tierConfirm.isEmpty()) {
      int windowSize = windowMax - windowBase;
      int threshold = windowSize / DENSITY_THRESHOLD_INVERSE;
      for (int cardinality = windowMatches.cardinality();
          upTo < tierConfirm.size() && cardinality >= threshold;
          upTo++, cardinality = windowMatches.cardinality()) {
        DisiWrapper other = tierConfirm.get(upTo);
        if (other.docID() < windowBase) {
          other.approximation().advance(windowBase);
        }
        other.twoPhase().applyMask(windowMax, windowMatches, windowBase);
      }
    }

    boolean hasRemaining = bulkUpTo < tierBulk.size() || upTo < tierConfirm.size();
    if (hasRemaining) {
      // Leap-frog remaining unprocessed clauses (bulk that bailed + confirm-tier).
      advanceHead:
      for (int windowMatch = windowMatches.nextSetBit(0);
          windowMatch != DocIdSetIterator.NO_MORE_DOCS; ) {
        int doc = windowBase + windowMatch;
        for (int i = bulkUpTo; i < tierBulk.size(); ++i) {
          DocIdSetIterator other = tierBulk.get(i).approximation();
          int otherDoc = other.docID();
          if (otherDoc < doc) {
            otherDoc = other.advance(doc);
          }
          if (doc != otherDoc) {
            windowMatch = advance(windowMatches, otherDoc - windowBase);
            continue advanceHead;
          }
        }
        for (int i = upTo; i < tierConfirm.size(); ++i) {
          DocIdSetIterator other = tierConfirm.get(i).approximation();
          int otherDoc = other.docID();
          if (otherDoc < doc) {
            otherDoc = other.advance(doc);
          }
          if (doc != otherDoc) {
            windowMatch = advance(windowMatches, otherDoc - windowBase);
            continue advanceHead;
          }
        }
        for (int i = upTo; i < tierConfirm.size(); ++i) {
          TwoPhaseIterator twoPhase = tierConfirm.get(i).twoPhase();
          if (twoPhase != null && twoPhase.matches() == false) {
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

    tierBulk.clear();
    tierMask.clear();
    tierConfirm.clear();

    windowMatches.clear();
  }

  /**
   * Fused AND + clear: ANDs source into target and zeros source for reuse. Fully vectorizable
   * (no loop-carried dependency) — the JIT maps this to NEON VAND + zero-store or AVX VPAND.
   */
  private static void andClear(FixedBitSet target, FixedBitSet source) {
    long[] t = target.getBits();
    long[] s = source.getBits();
    for (int j = 0; j < WORDS_PER_WINDOW; j++) {
      t[j] &= s[j];
      s[j] = 0;
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
