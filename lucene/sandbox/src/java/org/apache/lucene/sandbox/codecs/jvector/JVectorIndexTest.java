package org.apache.lucene.sandbox.codecs.jvector;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;
import org.apache.lucene.sandbox.codecs.jvector.JVectorCodec;

import java.io.IOException;
import java.nio.file.Path;

public class JVectorIndexTest {

    public static void main(String[] args) throws IOException {
        Path indexPath = Path.of("test-jvector-index");
        FSDirectory dir = FSDirectory.open(indexPath);

        IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());
        config.setCodec(new JVectorCodec());  // <- this ensures your codec is used

        try (IndexWriter writer = new IndexWriter(dir, config)) {
            Document doc = new Document();
            doc.add(new TextField("title", "Lucene with JVector", Field.Store.YES));
            doc.add(new KnnFloatVectorField("embedding", new float[]{0.1f, 0.3f, 0.9f}));

            writer.addDocument(doc);
            writer.commit();
        }
    }
}

