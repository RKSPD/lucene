package org.apache.lucene.sandbox.codecs.jvector;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.*;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.*;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;


import java.io.IOException;
import java.util.HashMap;


// ----------- SKELETON -------------
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


// -------------- DEVELOPMENT -------------
//public class JVectorReader extends KnnVectorsReader {
//    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
//    private static final int DEFAULT_OVER_QUERY_FACTOR = 5;
//
//    private final FieldInfos fieldInfos;
//    private final String baseDataFileName;
//
//    //FIELD ENTRY IS DEFINED LATER IN JVectorReader.java
//    // TODO: private final Map<String, FieldEntry> fieldEntryMap = new HashMap<>(1);
//
//    private final Directory directory;
//    private final SegmentReadState state;
//
//    public JVectorReader(SegmentReadState state) throws IOException {
//        this.state = state;
//        this.fieldInfos = state.fieldInfos;
//        this.baseDataFileName = state.segmentInfo.name + "_" + state.segmentSuffix;
//        String metaFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, JVectorFormat.META_EXTENSION);
//        this.directory = state.directory;
//        boolean success = false;
//        try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
//            CodecUtil.checkIndexHeader(
//                    meta,
//                    JVectorFormat.META_CODEC_NAME,
//                    JVectorFormat.VERSION_START,
//                    JVectorFormat.VERSION_CURRENT,
//                    state.segmentInfo.getId(),
//                    state.segmentSuffix
//            );
//            // TODO: add readFields(meta) after implementing the fields object
//            CodecUtil.checkFooter(meta);
//            success = true;
//        } finally {
//            if (!success) {
//                IOUtils.closeWhileHandlingException(this);
//            }
//        }
//    }
//
//    @Override
//    public void checkIntegrity() throws IOException {
//        // Implement with field entry
//    }
//
//    // TODO: Implement this
////    @Override
////    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
////
////    }
//
//    class FieldEntry {
//        private final FieldInfo fieldInfo;
//        private final VectorEncoding vectorEncoding;
//        private final VectorSimilarityFunction vectorSimilarityFunction;
//        private final int dimension;
//        private final long vectorIndexOffset;
//        private final long vectorIndexLength;
//        private final long pqCodebooksAndVectorsOffset;
//        private final long pqCodebooksAndVectorsLength;
//        private final String vectorIndexFieldFileName;
//        private final ReaderSupplier readerSupplier;
//        private final OnDiskGraphIndex graphIndex;
//        private final PQVectors pqVectors; // pq vectors and their associated codebooks
//
//    }
//}