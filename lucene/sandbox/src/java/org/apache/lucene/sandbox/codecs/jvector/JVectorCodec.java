package org.apache.lucene.sandbox.codecs.jvector;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene103.Lucene103Codec;

//public class JVectorCodec extends FilterCodec {
//    private final KnnVectorsFormat knnFormat = new JVectorFormat();
//
//    public JVectorCodec() {
//        super("JVector", new Lucene103Codec());
//    }
//
//    @Override
//    public KnnVectorsFormat knnVectorsFormat() {
//        return knnFormat;
//    }
//}

public class JVectorCodec extends FilterCodec {

    public static final String CODEC_NAME = "JVectorCodec";
    private int minBatchSizeForQuantization;

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

    @Override
    public KnnVectorsFormat knnVectorsFormat() { return new JVectorFormat(minBatchSizeForQuantization);}
}