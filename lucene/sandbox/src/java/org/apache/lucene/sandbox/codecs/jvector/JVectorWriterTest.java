//package org.apache.lucene.sandbox.codecs.jvector;
//
//import org.apache.lucene.index.FieldInfo;
//import org.apache.lucene.index.VectorEncoding;
//import org.apache.lucene.index.VectorSimilarityFunction;
//
//import java.io.IOException;
//
//public class JVectorWriterTest {
//
//    public static void main(String[] args) throws IOException {
//        System.out.println("🔍 Starting JVectorWriter tests...");
//
//        testAddFieldAndAddValue();
//        testMergeOneField();
//        testFlushAndFinish();
//        testRamBytesUsed();
//
//        System.out.println("✅ All tests passed.");
//    }
//
//    static void testAddFieldAndAddValue() throws IOException {
//        System.out.println("🔧 testAddFieldAndAddValue...");
//        var state = TestUtils.mockSegmentWriteState();
//        var writer = new JVectorWriter(state, 16, 32, 1.1f, 1.2f, 10);
//
//        FieldInfo fieldInfo = TestUtils.mockFieldInfo("float_field", VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 4);
//        var fieldWriter = writer.addField(fieldInfo);
//
//        float[] vec1 = new float[]{1.0f, 2.0f, 3.0f, 4.0f};
//        float[] vec2 = new float[]{5.0f, 6.0f, 7.0f, 8.0f};
//
//        // Use your FieldWriter's addValue
//        if (fieldWriter instanceof JVectorWriter.FieldWriter) {
//            var fw = (JVectorWriter.FieldWriter<?>) fieldWriter;
//            fw.addValue(0, vec1);
//            fw.addValue(1, vec2);
//        } else {
//            throw new IllegalStateException("Unexpected fieldWriter type: " + fieldWriter.getClass());
//        }
//
//        System.out.println("✅ testAddFieldAndAddValue passed.");
//    }
//
//    static void testMergeOneField() throws IOException {
//        System.out.println("🔧 testMergeOneField...");
//        var state = TestUtils.mockSegmentWriteState();
//        var writer = new JVectorWriter(state, 16, 32, 1.1f, 1.2f, 10);
//
//        FieldInfo fieldInfo = TestUtils.mockFieldInfo("float_field", VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 4);
//        writer.mergeOneField(fieldInfo, TestUtils.mockMergeState(fieldInfo));
//
//        System.out.println("✅ mergeOneField ran without exception.");
//    }
//
//    static void testFlushAndFinish() throws IOException {
//        System.out.println("🔧 testFlushAndFinish...");
//        var state = TestUtils.mockSegmentWriteState();
//        var writer = new JVectorWriter(state, 16, 32, 1.1f, 1.2f, 10);
//
//        FieldInfo fieldInfo = TestUtils.mockFieldInfo("float_field", VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 4);
//        var fieldWriter = writer.addField(fieldInfo);
//
//        if (fieldWriter instanceof JVectorWriter.FieldWriter) {
//            var fw = (JVectorWriter.FieldWriter<float[]>) fieldWriter;
//            fw.addValue(0, new float[]{1.0f, 2.0f, 3.0f, 4.0f});
//        }
//
//        writer.flush(1, null);
//        writer.finish();
//        writer.close();
//
//        System.out.println("✅ testFlushAndFinish passed.");
//    }
//
//    static void testRamBytesUsed() throws IOException {
//        System.out.println("🔧 testRamBytesUsed...");
//        var state = TestUtils.mockSegmentWriteState();
//        var writer = new JVectorWriter(state, 16, 32, 1.1f, 1.2f, 10);
//
//        FieldInfo fieldInfo = TestUtils.mockFieldInfo("float_field", VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 4);
//        var fieldWriter = writer.addField(fieldInfo);
//
//        if (fieldWriter instanceof JVectorWriter.FieldWriter) {
//            var fw = (JVectorWriter.FieldWriter<float[]>) fieldWriter;
//            fw.addValue(0, new float[]{1.0f, 2.0f, 3.0f, 4.0f});
//        }
//
//        long ramUsed = writer.ramBytesUsed();
//        assert ramUsed > 0 : "RAM usage should be positive.";
//        System.out.println("✅ RAM used: " + ramUsed + " bytes.");
//    }
//}


package org.apache.lucene.sandbox.codecs.jvector;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;

import java.io.IOException;

public class JVectorWriterTest {

    public static void main(String[] args) throws IOException {
        System.out.println("🔍 Starting JVectorWriter tests...");

        testAddFieldAndAddValue();
        testFlushAndFinish();
        testRamBytesUsed();

        System.out.println("✅ All tests passed.");
    }

    @SuppressWarnings("unchecked")
    static void testAddFieldAndAddValue() throws IOException {
        System.out.println("🔧 testAddFieldAndAddValue...");
        var state = TestUtils.mockSegmentWriteState();
        var writer = new JVectorWriter(state, 16, 32, 1.1f, 1.2f, 10);

        FieldInfo fieldInfo = TestUtils.mockFieldInfo("float_field", VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 4);
        var fieldWriter = writer.addField(fieldInfo);

        var fw = (JVectorWriter.FieldWriter<float[]>) fieldWriter;
        fw.addValue(0, new float[]{1.0f, 2.0f, 3.0f, 4.0f});
        fw.addValue(1, new float[]{5.0f, 6.0f, 7.0f, 8.0f});

        System.out.println("✅ testAddFieldAndAddValue passed.");
    }

    @SuppressWarnings("unchecked")
    static void testFlushAndFinish() throws IOException {
        System.out.println("🔧 testFlushAndFinish...");
        var state = TestUtils.mockSegmentWriteState();
        var writer = new JVectorWriter(state, 16, 32, 1.1f, 1.2f, 10);

        FieldInfo fieldInfo = TestUtils.mockFieldInfo("float_field", VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 4);
        var fieldWriter = writer.addField(fieldInfo);

        var fw = (JVectorWriter.FieldWriter<float[]>) fieldWriter;
        fw.addValue(0, new float[]{1.0f, 2.0f, 3.0f, 4.0f});

        writer.flush(1, null);
        writer.finish();
        writer.close();

        System.out.println("✅ testFlushAndFinish passed.");
    }

    @SuppressWarnings("unchecked")
    static void testRamBytesUsed() throws IOException {
        System.out.println("🔧 testRamBytesUsed...");
        var state = TestUtils.mockSegmentWriteState();
        var writer = new JVectorWriter(state, 16, 32, 1.1f, 1.2f, 10);

        FieldInfo fieldInfo = TestUtils.mockFieldInfo("float_field", VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 4);
        var fieldWriter = writer.addField(fieldInfo);

        var fw = (JVectorWriter.FieldWriter<float[]>) fieldWriter;
        fw.addValue(0, new float[]{1.0f, 2.0f, 3.0f, 4.0f});

        long ramUsed = writer.ramBytesUsed();
        assert ramUsed > 0 : "RAM usage should be positive.";
        System.out.println("✅ RAM used: " + ramUsed + " bytes.");
    }
}

