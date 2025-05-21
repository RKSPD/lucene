package org.apache.lucene.sandbox.codecs.jvector;

import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene103.Lucene103Codec;

public class JVectorCodec extends FilterCodec {
    private final KnnVectorsFormat knnFormat = new JVectorFormat();

    public JVectorCodec() {
        super("JVector", new Lucene103Codec());
    }

    @Override
    public KnnVectorsFormat knnVectorsFormat() {
        return knnFormat;
    }
}