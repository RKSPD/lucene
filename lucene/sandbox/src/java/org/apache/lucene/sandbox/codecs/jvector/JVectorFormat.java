package org.apache.lucene.sandbox.codecs.jvector;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorSimilarityFunction;

import java.io.IOException;

// FULLY PORTED

public class JVectorFormat extends KnnVectorsFormat {
    public static final String NAME = "JVectorFormat";
    public static final String META_CODEC_NAME = "JVectorVectorsFormatMeta";
    public static final String VECTOR_INDEX_CODEC_NAME = "JVectorVectorsFormatIndex";
    public static final String JVECTOR_FILES_SUFFIX = "jvector";
    public static final String META_EXTENSION = "meta-" + JVECTOR_FILES_SUFFIX;
    public static final String VECTOR_INDEX_EXTENSION = "data-" + JVECTOR_FILES_SUFFIX;
    public static final int DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION = 1024; // The minimum number of vectors required to trigger
    // quantization
    //public static final VectorSimilarityFunction DEFAULT_VECTOR_SIMILARITY_FUNCTION = VectorSimilarityFunction.DOT_PRODUCT;
    public static final int VERSION_START = 0;
    public static final int VERSION_CURRENT = VERSION_START;
    private static final int DEFAULT_MAX_CONN = 32;
    private static final int DEFAULT_BEAM_WIDTH = 100;
    private static final float DEFAULT_DEGREE_OVERFLOW = 2f;
    private static final float DEFAULT_ALPHA = 2f;

    private final int maxConn;
    private final int beamWidth;
    private final int minBatchSizeForQuantization;

    public JVectorFormat() {
        this(NAME, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH, 500000000);
    }

    public JVectorFormat(int minBatchSizeForQuantization) {
        this(NAME, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH, minBatchSizeForQuantization);
    }

    public JVectorFormat(int maxConn, int beamWidth) {
        this(NAME, maxConn, beamWidth, 50);
    }

    public JVectorFormat(String name, int maxConn, int beamWidth, int minBatchSizeForQuantization) {
        super(name);
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
        this.minBatchSizeForQuantization = minBatchSizeForQuantization;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new JVectorWriter(state, maxConn, beamWidth, DEFAULT_DEGREE_OVERFLOW, DEFAULT_ALPHA, minBatchSizeForQuantization);
        //return new JVectorWriter(state);
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new JVectorReader(state);
    }

    @Override
    public int getMaxDimensions(String dim) {
        return 8192;
    }
}
