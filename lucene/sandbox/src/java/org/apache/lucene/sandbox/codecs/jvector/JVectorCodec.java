package org.apache.lucene.sandbox.codecs.jvector;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene103.Lucene103Codec;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;


// FULLY PORTED
public class JVectorCodec extends FilterCodec {

    public static final String CODEC_NAME = "JVectorCodec";
    private int minBatchSizeForQuantization;
    //private VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;

    public JVectorCodec() {
        this(CODEC_NAME, new Lucene103Codec(), JVectorFormat.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION);
    }

    public JVectorCodec(int minBatchSizeForQuantization) {
        this(CODEC_NAME, new Lucene103Codec(), minBatchSizeForQuantization);
    }

    public JVectorCodec(String codecName, Codec delegate, int minBatchSizeForQuantization) {
        super(codecName, delegate);
        this.minBatchSizeForQuantization = minBatchSizeForQuantization;
    }

    // TODO: Make vector similarity function user selectable
//    public JVectorCodec(String codecName, Codec delegate, int minBatchSizeForQuantization, VectorSimilarityFunction function) {
//        super(codecName, delegate);
//        this.minBatchSizeForQuantization = minBatchSizeForQuantization;
//        this.vector_similarity_function = function;
//    }

    @Override
    public KnnVectorsFormat knnVectorsFormat() { return new JVectorFormat(minBatchSizeForQuantization);}
}