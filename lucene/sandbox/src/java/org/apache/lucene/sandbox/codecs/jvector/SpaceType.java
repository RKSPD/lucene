package org.apache.lucene.sandbox.codecs.jvector;

import org.apache.lucene.index.VectorSimilarityFunction;

public enum SpaceType {
    L2(VectorSimilarityFunction.EUCLIDEAN),
    COSINE(VectorSimilarityFunction.COSINE),
    INNER_PRODUCT(VectorSimilarityFunction.DOT_PRODUCT),
    UNDEFINED(null);

    private final VectorSimilarityFunction luceneSim;

    SpaceType(VectorSimilarityFunction luceneSim) {
        this.luceneSim = luceneSim;
    }

    public VectorSimilarityFunction toLuceneSim() {
        if (luceneSim == null) throw new IllegalArgumentException("No mapping for UNDEFINED");
        return luceneSim;
    }

    public float scoreTranslation(float rawScore) {
        return switch (this) {
            case L2 -> 1 / (1 + rawScore);
            case COSINE -> Math.max((2.0f - rawScore) / 2.0f, 0.f);
            case INNER_PRODUCT -> rawScore >= 0 ? 1 / (1 + rawScore) : -rawScore + 1;
            default -> throw new IllegalArgumentException("Unrecognized space type: " + this);
        };
    }
    public void validateVector(byte[] vector) {
        VectorUtils.validateNotZeroVector(vector, "SpaceType " + this.name());
    }


}
