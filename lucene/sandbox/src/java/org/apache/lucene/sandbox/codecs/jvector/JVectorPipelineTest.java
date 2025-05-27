package org.apache.lucene.sandbox.codecs.jvector;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene103.Lucene103Codec;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.util.Version;

import java.io.IOException;

public class JVectorPipelineTest {

    public static void main(String[] args) throws IOException {
        System.out.println("🔍 Starting Full JVector Pipeline Test...");

        // Setup index directory
        ByteBuffersDirectory directory = new ByteBuffersDirectory();

        // Configure IndexWriter with your JVectorCodec
        IndexWriterConfig config = new IndexWriterConfig();
        config.setCodec(new Lucene103Codec() {
            @Override
            public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return new JVectorFormat(); // Your JVectorFormat
            }
        });

        IndexWriter writer = new IndexWriter(directory, config);

        // Add documents with float[] vectors
        for (int i = 0; i < 5; i++) {
            Document doc = new Document();
            doc.add(new KnnFloatVectorField("vector", new float[]{i, i + 1, i + 2, i + 3}));
            doc.add(new StringField("id", "doc" + i, Field.Store.YES));
            writer.addDocument(doc);
        }

        writer.commit();
        writer.close();

        System.out.println("✅ Indexing pipeline completed.");

        // Open index reader
        DirectoryReader reader = DirectoryReader.open(directory);
        System.out.println("✅ Reader opened with " + reader.maxDoc() + " documents.");

        // Perform a dummy search (assuming you have JVectorReader ready)
        IndexSearcher searcher = new IndexSearcher(reader);
        KnnFloatVectorQuery query = new KnnFloatVectorQuery("vector", new float[]{1f, 2f, 3f, 4f}, 3);
        System.out.println("✅ Running KNN query...");
        var topDocs = searcher.search(query, 3);
        System.out.println("✅ Top " + topDocs.scoreDocs.length + " results retrieved.");

        reader.close();
        directory.close();

        System.out.println("✅ Full JVector pipeline test passed.");
    }
}

