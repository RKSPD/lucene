package org.apache.lucene.sandbox.codecs.jvector;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class JVectorRandomAccessReader implements RandomAccessReader {
    private final byte[] internalBuffer = new byte[Long.BYTES];
    private final IndexInput indexInputDelegate;
    private volatile boolean closed = false;

    public JVectorRandomAccessReader(IndexInput indexInputDelegate) { this.indexInputDelegate = indexInputDelegate; }

    @Override
    public void seek(long offset) throws IOException {
        indexInputDelegate.seek(offset);
    }

    @Override
    public long getPosition() throws IOException {
        return indexInputDelegate.getFilePointer();
    }

    @Override
    public int readInt() throws IOException {
        return indexInputDelegate.readInt();
    }

    @Override
    public float readFloat() throws IOException {
        indexInputDelegate.readBytes(internalBuffer, 0, Float.BYTES);
        FloatBuffer buffer = ByteBuffer.wrap(internalBuffer).asFloatBuffer();
        return buffer.get();
    }

    @Override
    public long readLong() throws IOException {
        return indexInputDelegate.readLong();
    }

    @Override
    public void readFully(byte[] bytes) throws IOException {
        indexInputDelegate.readBytes(bytes, 0, bytes.length);
    }

    @Override
    public void readFully(ByteBuffer buffer) throws IOException {
        long remainingInFile = indexInputDelegate.length() - indexInputDelegate.getFilePointer();
        if (buffer.remaining() < remainingInFile) {
            throw new EOFException("Requested " + buffer.remaining() + " bytes but only " + remainingInFile + " available");
        }

        if (buffer.hasArray()) {
            int off = buffer.arrayOffset() + buffer.position();
            int len = buffer.remaining();
            indexInputDelegate.readBytes(buffer.array(), off, len);
            buffer.position(buffer.limit());
            return;
        }

        while (buffer.hasRemaining()) {
            final int bytesToRead = Math.min(buffer.remaining(), Long.BYTES);
            indexInputDelegate.readBytes(this.internalBuffer, 0, bytesToRead);
            buffer.put(this.internalBuffer, 0, bytesToRead);
        }
    }

    @Override
    public void readFully(long[] vector) throws IOException {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = readLong();
        }
    }

    @Override
    public void read(int[] ints, int offset, int count) throws IOException {
        for (int i = 0; i < count; i++) {
            ints[offset + i] = readInt();
        }
    }

    @Override
    public void read(float[] floats, int offset, int count) throws IOException {
        for (int i = 0; i < count; i++) {
            floats[offset + i] = readFloat();
        }
    }

    @Override
    public void close() throws IOException {
        this.closed = true;
    }

    public static class Supplier implements ReaderSupplier {
        private final Directory directory;
        private final String fileName;
        private final IOContext context;
        private final AtomicInteger readerCount = new AtomicInteger(0);
        private final AtomicReference<IndexInput> currentInput = new AtomicReference<>(null);
        private final ConcurrentHashMap<Integer, RandomAccessReader> readers = new ConcurrentHashMap<>();

        public Supplier(Directory directory, String fileName, IOContext context) {
            this.directory = directory;
            this.fileName = fileName;
            this.context = context;
        }

        @Override
        public RandomAccessReader get() throws IOException {
            synchronized (directory) {
                if (currentInput.get() == null) {
                    currentInput.set(directory.openInput(fileName, context));
                }
                IndexInput input = currentInput.get().clone();

                var reader = new JVectorRandomAccessReader(input);
                int readerId = readerCount.getAndIncrement();
                readers.put(readerId, reader);
                return reader;
            }
        }

        @Override
        public void close() throws IOException {
            var input = currentInput.get();
            if (input != null) {
                IOUtils.closeWhileHandlingException(input);
            }

            for(RandomAccessReader reader : readers.values()) {
                IOUtils.closeWhileHandlingException(reader::close);
            }

            readers.clear();
            readerCount.set(0);
        }
    }
}
