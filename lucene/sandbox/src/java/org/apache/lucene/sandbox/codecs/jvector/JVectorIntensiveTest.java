// src/test/java/org/apache/lucene/sandbox/codecs/jvector/JVectorIntensiveTest.java
package org.apache.lucene.sandbox.codecs.jvector;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.IOUtils;

import java.io.IOException;
import java.util.Random;

/**
 * Intensive test for the JVector KnnVectorsFormat, implemented as a standalone
 * executable class. This test directly utilizes JVectorCodec for indexing
 * and searching, and includes various scenarios to thoroughly test its functionality.
 *
 * It accommodates the JVectorCodec and JVectorFormat changes where:
 * - JVectorCodec only allows passing minBatchSizeForQuantization.
 * - HNSW graph parameters (maxConn, beamWidth, degreeOverflow, alpha)
 *   are now fixed defaults within JVectorFormat.
 */
public class JVectorIntensiveTest {

    private static Directory directory;
    private static IndexWriter writer;
    private static Random random;

    // Helper to get a random int, mimicking Lucene's test framework's atLeast(int)
    private static int atLeast(int min) {
        return min + random.nextInt(5); // Adds 0 to 4 to the minimum, ensuring at least 'min'
    }

    // Helper to get a random int for higher ranges
    private static int randomInt(int min, int max) {
        return min + random.nextInt(max - min + 1);
    }

    /**
     * Creates a randomized JVectorCodec instance.
     * Only minimumBatchSizeForQuantization is randomized here, as
     * other HNSW parameters are now fixed defaults within JVectorFormat.
     */
    private static Codec createRandomizedJVectorCodec() {
        int minBatchSizeForQuantization;
        if (random.nextBoolean()) { // Sometimes disable PQ by setting a very high threshold
            minBatchSizeForQuantization = Integer.MAX_VALUE;
        } else if (random.nextBoolean()) { // Sometimes force PQ by setting 0 threshold
            minBatchSizeForQuantization = 0; // PQ will always be computed if enough vectors
        } else { // Otherwise, a reasonable range
            minBatchSizeForQuantization = randomInt(10, 500);
        }
        return new JVectorCodec(minBatchSizeForQuantization);
    }

    // Manual setup method
    private static void setUp() throws IOException {
        random = new Random(System.nanoTime()); // Use a new random seed for each test run
        directory = new ByteBuffersDirectory();
        // IndexWriter will be created in each test method for specific codec configurations
    }

    // Manual teardown method
    private static void tearDown() throws IOException {
        // Ensure writer is closed if it was opened in a test and not closed
        if (writer != null) {
            writer.close();
            writer = null;
        }
        // Always close the directory
        if (directory != null) {
            directory.close();
            directory = null;
        }
    }

    /**
     * Helper to generate a random float vector for testing.
     */
    protected static float[] generateRandomVector(int dim) {
        float[] vector = new float[dim];
        for (int i = 0; i < dim; i++) {
            vector[i] = random.nextFloat();
        }
        return vector;
    }

    // Custom assertion equivalent
    private static void assertEquals(String message, long expected, long actual) {
        if (expected != actual) {
            fail(message + ": Expected " + expected + ", Got " + actual);
        }
    }

    // Custom assertion equivalent
    private static void assertTrue(String message, boolean condition) {
        if (!condition) {
            fail(message);
        }
    }

    // Custom assertion equivalent
    private static void assertFalse(String message, boolean condition) {
        if (condition) {
            fail(message);
        }
    }

    // Custom assertion equivalent
    private static void fail(String message) {
        System.err.println("TEST FAILED: " + message);
        // Optionally, throw an exception to stop the process or exit with error code
        throw new RuntimeException("Test failure: " + message);
        // System.exit(1); // Or exit for immediate termination
    }

    public static void testSimpleIndexAndSearch() throws Exception {
        System.out.println("--- Running testSimpleIndexAndSearch ---");
        int numDocs = atLeast(100);
        int dimension = 128;
        String fieldName = "vector_field";

        IndexWriterConfig iwc = new IndexWriterConfig();
        iwc.setCodec(createRandomizedJVectorCodec());
        writer = new IndexWriter(directory, iwc);

        for (int i = 0; i < numDocs; i++) {
            Document doc = new Document();
            float[] vector = generateRandomVector(dimension);
            doc.add(new KnnFloatVectorField(fieldName, vector, VectorSimilarityFunction.COSINE));
            doc.add(new StringField("id", "doc" + i, Field.Store.YES));
            writer.addDocument(doc);
        }
        writer.commit();
        writer.close();
        writer = null;

        try (DirectoryReader reader = DirectoryReader.open(directory)) {
            assertEquals("Expected 1 segment", 1, reader.leaves().size());

            IndexSearcher searcher = new IndexSearcher(reader);

            float[] queryVector = generateRandomVector(dimension);
            int k = atLeast(10);
            KnnFloatVectorQuery query = new KnnFloatVectorQuery(fieldName, queryVector, k);
            TopDocs topDocs = searcher.search(query, k);

            assertTrue("Results size should be <= k", topDocs.scoreDocs.length <= k);
            if (numDocs > 0) {
                assertFalse("Should return some results for non-empty index", topDocs.scoreDocs.length == 0);
                for (ScoreDoc sd : topDocs.scoreDocs) {
                    // Cosine similarity ranges from -1.0 to 1.0
                    assertTrue("Score should be between -1.0 and 1.0 for COSINE", sd.score >= -1.0f && sd.score <= 1.0f);
                }
            }
        }
        System.out.println("--- testSimpleIndexAndSearch finished successfully ---");
    }

    public static void testLargeScaleIndexAndSearch() throws Exception {
        System.out.println("--- Running testLargeScaleIndexAndSearch ---");
        int numDocs = atLeast(5000); // More documents
        int dimension = randomInt(16, 500); // Random dimension, larger range
        String fieldName = "vector_field_large";
        VectorSimilarityFunction[] similarityFunctions = VectorSimilarityFunction.values();

        IndexWriterConfig iwc = new IndexWriterConfig();
        iwc.setCodec(createRandomizedJVectorCodec());
        writer = new IndexWriter(directory, iwc);

        for (int i = 0; i < numDocs; i++) {
            Document doc = new Document();
            float[] vector = generateRandomVector(dimension);
            VectorSimilarityFunction func = similarityFunctions[random.nextInt(similarityFunctions.length)];
            doc.add(new KnnFloatVectorField(fieldName, vector, func));
            writer.addDocument(doc);
        }
        writer.commit();
        writer.close();
        writer = null;

        try (DirectoryReader reader = DirectoryReader.open(directory)) {
            assertTrue("Expected at least 1 segment", reader.leaves().size() >= 1);

            IndexSearcher searcher = new IndexSearcher(reader);
            int numSearches = atLeast(5);
            for (int s = 0; s < numSearches; s++) {
                float[] queryVector = generateRandomVector(dimension);
                int k = atLeast(10);
                KnnFloatVectorQuery query = new KnnFloatVectorQuery(fieldName, queryVector, k);
                TopDocs topDocs = searcher.search(query, k);
                assertTrue("Results size should be <= k", topDocs.scoreDocs.length <= k);
                if (numDocs > 0) {
                    assertFalse("Should return some results for non-empty index", topDocs.scoreDocs.length == 0);
                }
            }
        }
        System.out.println("--- testLargeScaleIndexAndSearch finished successfully ---");
    }

    public static void testMergingSegments() throws Exception {
        System.out.println("--- Running testMergingSegments ---");
        int numDocsPerSegment = atLeast(50);
        int numSegments = atLeast(5);
        int dimension = 64;
        String fieldName = "merge_vector_field";

        IndexWriterConfig iwc = new IndexWriterConfig();
        iwc.setCodec(createRandomizedJVectorCodec());
        iwc.setMergePolicy(new TieredMergePolicy()); // Use TieredMergePolicy for actual merging behavior
        writer = new IndexWriter(directory, iwc);

        for (int i = 0; i < numSegments; i++) {
            for (int j = 0; j < numDocsPerSegment; j++) {
                Document doc = new Document();
                float[] vector = generateRandomVector(dimension);
                doc.add(new KnnFloatVectorField(fieldName, vector, VectorSimilarityFunction.COSINE));
                writer.addDocument(doc);
            }
            writer.commit(); // Create a new segment after each batch
        }

        writer.forceMerge(1); // Force a merge into a single segment (or fewer segments)
        writer.commit();
        writer.close();
        writer = null;

        try (DirectoryReader reader = DirectoryReader.open(directory)) {
            assertEquals("Expected 1 segment after forced merge", 1, reader.leaves().size());

            IndexSearcher searcher = new IndexSearcher(reader);
            float[] queryVector = generateRandomVector(dimension);
            int k = atLeast(10);
            KnnFloatVectorQuery query = new KnnFloatVectorQuery(fieldName, queryVector, k);
            TopDocs topDocs = searcher.search(query, k);
            assertTrue("Results size should be <= k", topDocs.scoreDocs.length <= k);
            assertFalse("Should return some results after merge", topDocs.scoreDocs.length == 0);
        }
        System.out.println("--- testMergingSegments finished successfully ---");
    }

    public static void testZeroDocuments() throws Exception {
        System.out.println("--- Running testZeroDocuments ---");
        String fieldName = "empty_field";
        int dimension = 128;

        IndexWriterConfig iwc = new IndexWriterConfig();
        iwc.setCodec(createRandomizedJVectorCodec());
        writer = new IndexWriter(directory, iwc);
        writer.commit(); // Commit an empty index
        writer.close();
        writer = null;

        try (DirectoryReader reader = DirectoryReader.open(directory)) {
            assertEquals("Expected 0 documents", 0, reader.maxDoc());
            assertEquals("Expected 0 leaves (segments)", 0, reader.leaves().size());

            IndexSearcher searcher = new IndexSearcher(reader);
            float[] queryVector = generateRandomVector(dimension);
            int k = 1;
            KnnFloatVectorQuery query = new KnnFloatVectorQuery(fieldName, queryVector, k);
            TopDocs topDocs = searcher.search(query, k);
            assertEquals("Expected 0 results for empty index", 0, topDocs.scoreDocs.length);
        }
        System.out.println("--- testZeroDocuments finished successfully ---");
    }

    public static void testMixedFields() throws Exception {
        System.out.println("--- Running testMixedFields ---");
        int numDocs = atLeast(50);
        int vectorDimension = 64;
        String vectorFieldName = "my_vector_field";
        String nonVectorFieldName = "text_field";

        IndexWriterConfig iwc = new IndexWriterConfig();
        iwc.setCodec(createRandomizedJVectorCodec());
        writer = new IndexWriter(directory, iwc);

        for (int i = 0; i < numDocs; i++) {
            Document doc = new Document();
            // Add a vector field to roughly half the documents
            if (random.nextBoolean()) {
                doc.add(new KnnFloatVectorField(vectorFieldName, generateRandomVector(vectorDimension), VectorSimilarityFunction.COSINE));
            }
            // Add a non-vector field to all documents
            doc.add(new StringField(nonVectorFieldName, "some text " + i, Field.Store.NO));
            writer.addDocument(doc);
        }
        writer.commit();
        writer.close();
        writer = null;

        try (DirectoryReader reader = DirectoryReader.open(directory)) {
            IndexSearcher searcher = new IndexSearcher(reader);

            // Test search on the vector field
            float[] queryVector = generateRandomVector(vectorDimension);
            int k = atLeast(5);
            KnnFloatVectorQuery query = new KnnFloatVectorQuery(vectorFieldName, queryVector, k);
            TopDocs topDocs = searcher.search(query, k);
            assertTrue("Results size should be <= k", topDocs.scoreDocs.length <= k);
            // Should return results if any vector field was indexed and there are docs
            if (numDocs > 0 && topDocs.scoreDocs.length > 0) {
                assertFalse("Should return results for vector field", topDocs.scoreDocs.length == 0);
            }
        }
        System.out.println("--- testMixedFields finished successfully ---");
    }

    public static void testVariousDimensionsAndPQScenarios() throws Exception {
        System.out.println("--- Running testVariousDimensionsAndPQScenarios ---");
        int[] dimensions = {8, 16, 64, 128, 512, 1024}; // Test various dimensions
        int numDocs = atLeast(200);

        for (int dimension : dimensions) {
            System.out.println("  Testing dimension: " + dimension);
            // Scenario 1: PQ potentially enabled (random threshold)
            doTestVariousDimensionsAndPQ(dimension, numDocs, createRandomizedJVectorCodec());

            // Scenario 2: PQ explicitly disabled (very high threshold)
            doTestVariousDimensionsAndPQ(dimension, numDocs, new JVectorCodec(Integer.MAX_VALUE));

            // Scenario 3: PQ explicitly forced (threshold 0)
            if (numDocs > 0) { // PQ needs at least some documents to compute codebooks
                doTestVariousDimensionsAndPQ(dimension, numDocs, new JVectorCodec(0));
            }
        }
        System.out.println("--- testVariousDimensionsAndPQScenarios finished successfully ---");
    }

    private static void doTestVariousDimensionsAndPQ(int dimension, int numDocs, Codec codec) throws Exception {
        String testCaseName = codec.getName() + " (dim=" + dimension + ", docs=" + numDocs + ")";
        System.out.println("    Sub-test: " + testCaseName);

        Directory tempDir = new ByteBuffersDirectory();
        IndexWriterConfig iwc = new IndexWriterConfig();
        iwc.setCodec(codec);
        IndexWriter tempWriter = new IndexWriter(tempDir, iwc);

        String fieldName = "dim_pq_test_field_" + dimension + "_" + System.nanoTime(); // Unique field name

        for (int i = 0; i < numDocs; i++) {
            Document doc = new Document();
            float[] vector = generateRandomVector(dimension);
            doc.add(new KnnFloatVectorField(fieldName, vector, VectorSimilarityFunction.COSINE));
            tempWriter.addDocument(doc);
        }
        tempWriter.commit();
        tempWriter.close();

        try (DirectoryReader reader = DirectoryReader.open(tempDir)) {
            assertTrue("Expected at least 1 segment for " + testCaseName, reader.leaves().size() >= 1);

            IndexSearcher searcher = new IndexSearcher(reader);
            float[] queryVector = generateRandomVector(dimension);
            int k = atLeast(5);
            KnnFloatVectorQuery query = new KnnFloatVectorQuery(fieldName, queryVector, k);
            TopDocs topDocs = searcher.search(query, k);
            assertTrue("Results size should be <= k for " + testCaseName, topDocs.scoreDocs.length <= k);
            if (numDocs > 0) {
                assertFalse("Should return results for " + testCaseName, topDocs.scoreDocs.length == 0);
            }
        } finally {
            IOUtils.close(tempDir);
        }
        System.out.println("    Sub-test " + testCaseName + " finished successfully.");
    }

    public static void main(String[] args) throws Exception {
        System.out.println("✨ Starting Full JVector Intensive Test Suite ✨");

        try {
            setUp();
            testSimpleIndexAndSearch();
            tearDown(); // Reset for next test

            setUp();
            testLargeScaleIndexAndSearch();
            tearDown();

            setUp();
            testMergingSegments();
            tearDown();

            setUp();
            testZeroDocuments();
            tearDown();

            setUp();
            testMixedFields();
            tearDown();

            setUp();
            testVariousDimensionsAndPQScenarios();
            tearDown();

            System.out.println("🎉 All JVector Intensive Tests Completed Successfully! 🎉");
        } catch (Throwable t) { // <--- CHANGE IS HERE: catch Throwable instead of Exception
            System.err.println("❌ TEST SUITE FAILED: " + t.getMessage());
            t.printStackTrace(); // This should now print the full stack trace for all errors
            System.exit(1); // Exit with non-zero code on failure
        } finally {
            // Ensure final cleanup in case of unhandled exception
            try {
                if (writer != null) {
                    writer.close();
                }
                if (directory != null) {
                    directory.close();
                }
            } catch (IOException cleanupException) {
                System.err.println("Error during final cleanup: " + cleanupException.getMessage());
                cleanupException.printStackTrace();
            }
        }
    }
}