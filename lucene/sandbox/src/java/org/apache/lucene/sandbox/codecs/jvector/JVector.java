package org.apache.lucene.sandbox.codecs.jvector;

import org.apache.lucene.util.Version;
//import org.gradle.internal.impldep.com.google.common.collect.ImmutableMap;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import static org.apache.lucene.sandbox.codecs.jvector.JVectorConstants.DISK_ANN;
import java.util.Collections;
import java.util.List;

public class JVector {
    public static final String EXTENSION = "jvec";
    public static final String COMPOUND_EXTENSION = "cjvec";

    public static final JVector INSTANCE = new JVector();

    private JVector() {}

    public String getExtension() {
        return EXTENSION;
    }

    public String getCompoundExtension() {
        return COMPOUND_EXTENSION;
    }

    public List<String> mmapFileExtensions() {
        return Collections.emptyList();
    }

    public float score(float rawScore, SpaceType spaceType) {
        return spaceType.scoreTranslation(rawScore);
    }

    public float distanceToRadialThreshold(float distance, SpaceType spaceType) {
        return distance;
    }

    public float scoreToRadialThreshold(float score, SpaceType spaceType) {
        return score;
    }
}
