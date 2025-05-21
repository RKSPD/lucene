package org.apache.lucene.sandbox.codecs.jvector;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.*;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.Bits;
import io.github.jbellis.jvector.disk.ReaderSupplier;


import java.io.IOException;

public class JVectorReader extends KnnVectorsReader {
    public JVectorReader(SegmentReadState state) throws IOException {}

    public TopDocs search(String field, float[] target, int k, Bits acceptDocs) throws IOException {
        return null;
    }

    @Override
    public void close() throws IOException {}

    public long ramBytesUsed() {
        return 0;
    }

    @Override
    public void checkIntegrity() throws IOException {}

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        return null;
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        return null;
    }

    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {

    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {

    }

}
