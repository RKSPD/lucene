package org.apache.lucene.sandbox.codecs.jvector;

import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.*;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.util.AttributeSource;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;

import javax.swing.text.Segment;
import java.io.DataOutput;
import java.io.IOException;
import java.util.*;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readVectorEncoding;

public class JVectorWriter extends KnnVectorsWriter {

    private static final long SHALLOW_RAM_BYTES_USED = RamUsageEstimator.shallowSizeOfInstance(JVectorWriter.class);
    private final List<FieldWriter<?>> fields = new ArrayList<>();

    private final IndexOutput meta;
    private final IndexOutput vectorIndex;
    private final String indexDataFileName;
    private final String baseDataFileName;
    private final SegmentWriteState segmentWriteState;
    private final int maxConn;
    private final int beamWidth;
    private final float degreeOverflow;
    private final float alpha;
    private final int minimumBatchSizeForQuantization;

    private boolean finished = false;


    public JVectorWriter(
            SegmentWriteState segmentWriteState,
            int maxConn,
            int beamWidth,
            float degreeOverflow,
            float alpha,
            int minimumBatchSizeForQuantization
    ) throws IOException {
        this.segmentWriteState = segmentWriteState;
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
        this.degreeOverflow = degreeOverflow;
        this.alpha = alpha;
        this.minimumBatchSizeForQuantization = minimumBatchSizeForQuantization;
        String metaFileName = IndexFileNames.segmentFileName(
                segmentWriteState.segmentInfo.name,
                segmentWriteState.segmentSuffix,
                JVectorFormat.META_EXTENSION
        );

        this.indexDataFileName = IndexFileNames.segmentFileName(
                segmentWriteState.segmentInfo.name,
                segmentWriteState.segmentSuffix,
                JVectorFormat.VECTOR_INDEX_EXTENSION
        );
        this.baseDataFileName = segmentWriteState.segmentInfo.name + "_" + segmentWriteState.segmentSuffix;

        boolean success = false;
        try {
            meta = segmentWriteState.directory.createOutput(metaFileName, segmentWriteState.context);
            vectorIndex = segmentWriteState.directory.createOutput(indexDataFileName, segmentWriteState.context);
            CodecUtil.writeIndexHeader(
                    meta,
                    JVectorFormat.META_CODEC_NAME,
                    JVectorFormat.VERSION_CURRENT,
                    segmentWriteState.segmentInfo.getId(),
                    segmentWriteState.segmentSuffix
            );

            CodecUtil.writeIndexHeader(
                    vectorIndex,
                    JVectorFormat.VECTOR_INDEX_CODEC_NAME,
                    JVectorFormat.VERSION_CURRENT,
                    segmentWriteState.segmentInfo.getId(),
                    segmentWriteState.segmentSuffix
            );

            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(this);
            }
        }

    }

    // SKELETON CONSTRUCTOR
    // public JVectorWriter(SegmentWriteState state) throws IOException {}

    // IMPLEMENT THESE ONE AT A TIME
    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        if (fieldInfo.getVectorEncoding() == VectorEncoding.BYTE) {
            final String errorMessage = "byte[] vectors are not supported in JVector. "
                    + "Instead you should only use float vectors and leverage product quantization during indexing."
                    + "This can provides much greater savings in storage and memory";
            //System.out.println(errorMessage);
            throw new UnsupportedOperationException(errorMessage);
        }
        FieldWriter<?> newField = new FieldWriter<>(fieldInfo, segmentWriteState.segmentInfo.name);
        fields.add(newField);
        return newField;
    }

    @SuppressWarnings("unchecked")
    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        var success = false;
        try {
            switch (fieldInfo.getVectorEncoding()) {
                case BYTE:
                    FieldWriter<byte[]> byteWriter = (FieldWriter<byte[]>) addField(fieldInfo);
                    ByteVectorValues mergedBytes = MergedVectorValues.mergeByteVectorValues(fieldInfo, mergeState);
                    var iterator = mergedBytes.iterator();
                    for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                        byteWriter.addValue(doc, mergedBytes.vectorValue(doc));
                    }
                    writeField(byteWriter);
                    break;
                case FLOAT32:
                    FieldWriter<float[]> floatVectorFieldWriter = (FieldWriter<float[]>) addField(fieldInfo);
                    var mergedFloats = MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
                    var mergeStateIterator = mergedFloats.iterator();
                    for (int doc = mergeStateIterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = mergeStateIterator.nextDoc()) {
                        floatVectorFieldWriter.addValue(doc, mergedFloats.vectorValue(doc));
                    }
                    writeField(floatVectorFieldWriter);
                    break;
            }
            success = true;
        } finally {
            if (!success) {
                //IOUtils.closeWhileHandlingException(this);
            } else {
                //IOUtils.close(this);
            }
        }
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        //System.out.println("Flushing " + fields.size() + " fields");

        for (FieldWriter<?> field : fields) {
            if (sortMap == null) {
                writeField(field);
            } else {
                throw new UnsupportedOperationException("Not implemented yet");
            }
        }
    }

    private void writeField(FieldWriter<?> fieldData) throws IOException {
        OnHeapGraphIndex graph = fieldData.getGraph();
        //final var vectorIndexFieldMetadata = writeGraph(graph, fieldData);
        meta.writeInt(fieldData.fieldInfo.number);
        //vectorIndexFieldMetadata.toOutput(meta);

    }

    private VectorIndexFieldMetadata writeGraph(OnHeapGraphIndex graph, FieldWriter<?> fieldData) throws IOException {
        final String vectorIndexFieldFileName = baseDataFileName
                + "_"
                + fieldData.fieldInfo.name
                + "."
                + JVectorFormat.VECTOR_INDEX_EXTENSION;

        try (
                IndexOutput indexOutput = segmentWriteState.directory.createOutput(vectorIndexFieldFileName, segmentWriteState.context);
                final var randomAccessWriter = new JVectorRandomAccessWriter(indexOutput);
        ) {
            CodecUtil.writeIndexHeader(
                    indexOutput,
                    JVectorFormat.VECTOR_INDEX_CODEC_NAME,
                    JVectorFormat.VERSION_CURRENT,
                    segmentWriteState.segmentInfo.getId(),
                    segmentWriteState.segmentSuffix
            );
            final long startOffset = indexOutput.getFilePointer();

            //System.out.println("Writing graph to " + vectorIndexFieldFileName);

            long pqOffset = 0;
            long pqLength = 0;

            try (
                    var writer = new OnDiskGraphIndexWriter.Builder(graph, randomAccessWriter)
                            .with(new InlineVectors(fieldData.randomAccessVectorValues.dimension()))
                            .withStartOffset(startOffset)
                            .build();
            ) {
                var suppliers = Feature.singleStateFactory(
                        FeatureId.INLINE_VECTORS,
                        nodeId -> new InlineVectors.State(fieldData.randomAccessVectorValues.getVector(nodeId))
                );

                writer.write(suppliers);

                long endGraphOffset = randomAccessWriter.position();

                if (fieldData.randomAccessVectorValues.size() >= minimumBatchSizeForQuantization) {
                    pqOffset = endGraphOffset;
                    writePQCodebooksAndVectors(randomAccessWriter, fieldData);
                    pqLength = randomAccessWriter.position() - pqOffset;
                }

                CodecUtil.writeFooter(indexOutput);

                return new VectorIndexFieldMetadata(
                        fieldData.fieldInfo.number,
                        fieldData.fieldInfo.getVectorEncoding(),
                        fieldData.fieldInfo.getVectorSimilarityFunction(),
                        fieldData.randomAccessVectorValues.dimension(),
                        startOffset,
                        endGraphOffset - startOffset,
                        pqOffset,
                        pqLength
                );

            }
        }
    }

    /**
     * Writes the product quantization (PQ) codebooks and encoded vectors to a DataOutput stream.
     * This method compresses the original vector data using product quantization and encodes
     * all vector values into a smaller, compressed form for storage or transfer.
     *
     * @param out The DataOutput stream where the compressed PQ codebooks and encoded vectors will be written.
     * @param fieldData The field writer object providing access to the vector data to be compressed.
     * @throws IOException If an I/O error occurs during writing.
     */
    private static void writePQCodebooksAndVectors(DataOutput out, FieldWriter<?> fieldData) throws IOException {

        // TODO: should we make this configurable?
        // Compress the original vectors using PQ. this represents a compression ratio of 128 * 4 / 16 = 32x
        final var M = Math.min(fieldData.randomAccessVectorValues.dimension(), 16); // number of subspaces
        final var numberOfClustersPerSubspace = Math.min(256, fieldData.randomAccessVectorValues.size()); // number of centroids per
        // subspace
        ProductQuantization pq = ProductQuantization.compute(
                fieldData.randomAccessVectorValues,
                M, // number of subspaces
                numberOfClustersPerSubspace, // number of centroids per subspace
                true
        ); // center the dataset
        var pqv = pq.encodeAll(fieldData.randomAccessVectorValues);
        // write the compressed vectors to disk
        pqv.write(out);
    }

    public static class VectorIndexFieldMetadata {
        int fieldNumber;
        VectorEncoding vectorEncoding;
        VectorSimilarityFunction vectorSimilarityFunction;
        int vectorDimension;
        long vectorIndexOffset;
        long vectorIndexLength;
        long pqCodebooksAndVectorsOffset;
        long pqCodebooksAndVectorsLength;

        public void toOutput(IndexOutput out) throws IOException {
            out.writeInt(fieldNumber);
            out.writeInt(vectorEncoding.ordinal());
            // TODO: Update this once reader is implemented
            //out.writeInt(JVectorReader.VectorSimilarityMapper.distFuncToOrd(vectorSimilarityFunction));
            out.writeVInt(vectorDimension);
            out.writeVLong(vectorIndexOffset);
            out.writeVLong(vectorIndexLength);
            out.writeVLong(pqCodebooksAndVectorsOffset);
            out.writeVLong(pqCodebooksAndVectorsLength);
        }
        public VectorIndexFieldMetadata(
                int fieldNumber,
                VectorEncoding vectorEncoding,
                VectorSimilarityFunction vectorSimilarityFunction,
                int vectorDimension,
                long vectorIndexOffset,
                long vectorIndexLength,
                long pqCodebooksAndVectorsOffset,
                long pqCodebooksAndVectorsLength
        ) {
            this.fieldNumber = fieldNumber;
            this.vectorEncoding = vectorEncoding;
            this.vectorSimilarityFunction = vectorSimilarityFunction;
            this.vectorDimension = vectorDimension;
            this.vectorIndexOffset = vectorIndexOffset;
            this.vectorIndexLength = vectorIndexLength;
            this.pqCodebooksAndVectorsOffset = pqCodebooksAndVectorsOffset;
            this.pqCodebooksAndVectorsLength = pqCodebooksAndVectorsLength;
        }

        public VectorIndexFieldMetadata(IndexInput in) throws IOException {
            this.fieldNumber = in.readInt();
            this.vectorEncoding = readVectorEncoding(in);
            this.vectorSimilarityFunction = VectorSimilarityFunction.EUCLIDEAN;
            // JVectorReader.VectorSimilarityMapper.ordToLuceneDistFunc(in.readInt());
            this.vectorDimension = in.readVInt();
            this.vectorIndexOffset = in.readVLong();
            this.vectorIndexLength = in.readVLong();
            this.pqCodebooksAndVectorsOffset = in.readVLong();
            this.pqCodebooksAndVectorsLength = in.readVLong();
        }

    }
    @Override
    public void finish() throws IOException {
        //System.out.println("Finishing segment " + segmentWriteState.segmentInfo.name);
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;

        if (meta == null) {
            meta.writeInt(-1);
            CodecUtil.writeFooter(meta);
        }

        if (vectorIndex == null) {
            CodecUtil.writeFooter(vectorIndex);
        }
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(meta, vectorIndex);
    }

    @Override
    public long ramBytesUsed() {
        long total = SHALLOW_RAM_BYTES_USED;
        for (FieldWriter<?> field : fields) {
            total += field.ramBytesUsed();
        }
        return total;
    }

    class FieldWriter<T> extends KnnFieldVectorsWriter<T> {
        private final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
        private final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(FieldWriter.class);
        private final FieldInfo fieldInfo;
        private int lastDocID = -1;
        private final GraphIndexBuilder graphIndexBuilder;
        private final List<VectorFloat<?>> floatVectors = new ArrayList<>();
        private final String segmentName;
        private final RandomAccessVectorValues randomAccessVectorValues;
        private final BuildScoreProvider buildScoreProvider;

        FieldWriter(FieldInfo fieldInfo, String segmentName) {
            this.fieldInfo = fieldInfo;
            this.segmentName = segmentName;
            var originalDimension = fieldInfo.getVectorDimension();
            this.randomAccessVectorValues = new ListRandomAccessVectorValues(floatVectors, originalDimension);
            this.buildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(
                    randomAccessVectorValues,
                    getVectorSimilarityFunction(fieldInfo)
            );
            this.graphIndexBuilder = new GraphIndexBuilder(
                    buildScoreProvider,
                    randomAccessVectorValues.dimension(),
                    maxConn,
                    beamWidth,
                    degreeOverflow,
                    alpha,
                    false
            );
        }

        @Override
        public void addValue(int docID, T vectorValue) throws IOException {
            if (docID == lastDocID) {
                throw new IllegalArgumentException(
                        "VectorValuesField \""
                        + fieldInfo.name
                        + "\" appears more than once in this document (only one value is allowed per field)"
                );
            }
            if (vectorValue instanceof float[]) {
                var floats = (float[]) vectorValue;
                var vector = VECTOR_TYPE_SUPPORT.createFloatVector(floats);
                floatVectors.add(vector);
                graphIndexBuilder.addGraphNode(docID, vector);
            } else if (vectorValue instanceof byte[]) {
                final String errorMessage = "byte[] vectors are not supported in JVector. "
                        + "Instead you should only use float vectors and leverage product quantization during indexing."
                        + "This can provides much greater savings in storage and memory";
                throw new UnsupportedOperationException(errorMessage);
            } else {
                throw new IllegalArgumentException("Unsupported vector type: " + vectorValue.getClass());
            }

            lastDocID = docID;
        }

        @Override
        public T copyValue(T vectorValue) { throw new UnsupportedOperationException("copyValue not supported"); }

        @Override
        public long ramBytesUsed() {
            return SHALLOW_SIZE + graphIndexBuilder.getGraph().ramBytesUsed() + RamUsageEstimator.sizeOfCollection(floatVectors);
        }

        io.github.jbellis.jvector.vector.VectorSimilarityFunction getVectorSimilarityFunction(FieldInfo fieldInfo) {
            switch (fieldInfo.getVectorSimilarityFunction()) {
                case EUCLIDEAN:
                    return io.github.jbellis.jvector.vector.VectorSimilarityFunction.EUCLIDEAN;
                case COSINE:
                    return io.github.jbellis.jvector.vector.VectorSimilarityFunction.COSINE;
                case DOT_PRODUCT:
                    return io.github.jbellis.jvector.vector.VectorSimilarityFunction.DOT_PRODUCT;
                default:
                    throw new IllegalArgumentException("Unsupported similarity function: " + fieldInfo.getVectorSimilarityFunction());
            }
        }

        /**
         * This method will return the graph index for the field
         * @return OnHeapGraphIndex
         * @throws IOException IOException
         */
        public OnHeapGraphIndex getGraph() throws IOException {
            graphIndexBuilder.cleanup();
            return graphIndexBuilder.getGraph();
        }
    }
}
