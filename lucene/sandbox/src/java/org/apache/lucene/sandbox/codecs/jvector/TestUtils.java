package org.apache.lucene.sandbox.codecs.jvector;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.*;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;

import java.io.IOException;
import java.util.Collections;

public class TestUtils {

    public static SegmentWriteState mockSegmentWriteState() throws IOException {
        return new SegmentWriteState(
                InfoStream.NO_OUTPUT, // infoStream
                new ByteBuffersDirectory(), // directory
                new SegmentInfo(
                        new ByteBuffersDirectory(),
                        Version.LATEST,
                        Version.LATEST,
                        "test",
                        1,
                        false,
                        false,
                        Codec.getDefault(),
                        Collections.emptyMap(),
                        new byte[16],
                        Collections.emptyMap(),
                        null // indexSort
                ),
                new FieldInfos(new FieldInfo[0]), // fieldInfos
                null, // segUpdates (BufferedUpdates)
                IOContext.DEFAULT, // context
                "testSuffix" // segmentSuffix
        );
    }

    public static FieldInfo mockFieldInfo(String name, VectorEncoding encoding, VectorSimilarityFunction sim, int dim) {
        return new FieldInfo(
                name,
                0,
                false,
                false,
                false,
                IndexOptions.NONE,
                DocValuesType.NONE,
                DocValuesSkipIndexType.NONE,
                -1L,
                Collections.emptyMap(),
                0,
                0,
                0,
                dim,
                encoding,
                sim,
                false,
                false
        );
    }
}
