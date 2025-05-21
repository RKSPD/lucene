package org.apache.lucene.sandbox.codecs.jvector;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;


import java.io.IOException;

public class JVectorFormat extends KnnVectorsFormat {

    public JVectorFormat() {
        super("JVectorFormat");
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new JVectorWriter(state);
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new JVectorReader(state);
    }

    @Override
    public int getMaxDimensions(String dim) {
        return 2048;
    }
}
