//package org.apache.lucene.sandbox.codecs.jvector;
//
//import io.github.jbellis.jvector.disk.RandomAccessWriter;
//import io.github.jbellis.jvector.disk.RandomAccessWriter;
//import org.apache.lucene.store.IndexOutput;
//
//import java.io.IOException;
//import java.nio.ByteBuffer;
//
//public class JVectorRandomAccessWriter implements RandomAccessWriter {
//    private final byte[] writeBuffer = new byte[Long.BYTES];
//    private final IndexOutput indexOutputDelegate;
//
//    public JVectorRandomAccessWriter(IndexOutput indexOutputDelegate) {
//        this.indexOutputDelegate = indexOutputDelegate;
//    }
//
//    @Override
//    public void seek(long position) throws IOException {
//        //System.out.println("Pretending to be seeking to position " + position +  "in JVectorRandomAccessWriter," +
//                //" in practice it is not seeking anywhere.");
//    }
//
//    @Override
//    public long position() throws IOException {
//        return indexOutputDelegate.getFilePointer();
//    }
//
//    @Override
//    public void flush() throws IOException {
//        //System.out.println("Pretending to be flushing JVectorRandomAccessWriter," +
//                //"will be flushed after IndexOutput is closed");
//    }
//
//    @Override
//    public long checksum(long startOffset, long endOffset) throws IOException {
//        //System.out.println("Pretending to be computing checksum for range [" + startOffset + ", " + endOffset + "]");
//        return 0;
//    }
//
//    @Override
//    public void close() throws IOException {
//        indexOutputDelegate.close();
//    }
//
//    @Override
//    public void write(int b) throws IOException {
//        indexOutputDelegate.writeByte((byte) b);
//    }
//
//    @Override
//    public void write(byte[] b) throws IOException {
//        indexOutputDelegate.writeBytes(b, 0, b.length);
//    }
//
//    @Override
//    public void write(byte[] b, int off, int len) throws IOException {
//        indexOutputDelegate.writeBytes(b, off, len);
//    }
//
//    @Override
//    public void writeBoolean(boolean v) throws IOException {
//        indexOutputDelegate.writeByte((byte) (v ? 1 : 0));
//    }
//
//    @Override
//    public void writeByte(int v) throws IOException {
//        indexOutputDelegate.writeByte((byte) v);
//    }
//
//    @Override
//    public void writeShort(int v) throws IOException {
//        indexOutputDelegate.writeShort((short) v);
//    }
//
//    @Override
//    public void writeChar(int v) throws IOException {
//        throw new UnsupportedOperationException("JVectorRandomAccessWriter does not support writing chars");
//    }
//
//    @Override
//    public void writeInt(int v) throws IOException {
//        indexOutputDelegate.writeInt(v);
//    }
//
//    @Override
//    public void writeLong(long v) throws IOException {
//        indexOutputDelegate.writeLong(v);
//    }
//
//    @Override
//    public void writeFloat(float v) throws IOException {
//        ByteBuffer.wrap(writeBuffer).putFloat(v);
//        indexOutputDelegate.writeBytes(writeBuffer, 0, Float.BYTES);
//    }
//
//    @Override
//    public void writeDouble(double v) throws IOException {
//        writeLong(Double.doubleToLongBits(v));
//    }
//
//    @Override
//    public void writeBytes(String s) throws IOException {
//        throw new UnsupportedOperationException("JVectorRandomAccessWriter does not support writing bytes");
//    }
//
//    @Override
//    public void writeChars(String s) throws IOException {
//        throw new UnsupportedOperationException("JVectorRandomAccessWriter does not support writing chars");
//    }
//
//    @Override
//    public void writeUTF(String s) throws IOException {
//        throw new UnsupportedOperationException("JVectorRandomAccessWriter does not support writing UTF strings");
//    }
//}


package org.apache.lucene.sandbox.codecs.jvector;

import io.github.jbellis.jvector.disk.RandomAccessWriter;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;
import java.nio.ByteBuffer;

public class JVectorRandomAccessWriter implements RandomAccessWriter {
    private final byte[] writeBuffer = new byte[Long.BYTES];
    private final IndexOutput indexOutputDelegate;

    public JVectorRandomAccessWriter(IndexOutput indexOutputDelegate) {
        this.indexOutputDelegate = indexOutputDelegate;
        System.out.println("🟢 JVectorRandomAccessWriter: Created with IndexOutput: " + indexOutputDelegate);
    }

    @Override
    public void seek(long position) throws IOException {
        //System.out.println("🔵 JVectorRandomAccessWriter.seek(" + position + ")");
    }

    @Override
    public long position() throws IOException {
        long pos = indexOutputDelegate.getFilePointer();
        //System.out.println("🔵 JVectorRandomAccessWriter.position() = " + pos);
        return pos;
    }

    @Override
    public void flush() throws IOException {
        //System.out.println("🔵 JVectorRandomAccessWriter.flush()");
    }

    @Override
    public long checksum(long startOffset, long endOffset) throws IOException {
        //System.out.println("🔵 JVectorRandomAccessWriter.checksum(" + startOffset + ", " + endOffset + ")");
        return 0;
    }

    @Override
    public void close() throws IOException {
        //System.out.println("🚨 JVectorRandomAccessWriter.close() called! Closing indexOutputDelegate...");
        indexOutputDelegate.close();
    }

    @Override
    public void write(int b) throws IOException {
        //System.out.println("📝 write(int): " + b);
        indexOutputDelegate.writeByte((byte) b);
    }

    @Override
    public void write(byte[] b) throws IOException {
        //System.out.println("📝 write(byte[]) of length " + b.length);
        indexOutputDelegate.writeBytes(b, 0, b.length);
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
        //System.out.println("📝 write(byte[], " + off + ", " + len + ")");
        indexOutputDelegate.writeBytes(b, off, len);
    }

    @Override
    public void writeBoolean(boolean v) throws IOException {
        //System.out.println("📝 writeBoolean: " + v);
        indexOutputDelegate.writeByte((byte) (v ? 1 : 0));
    }

    @Override
    public void writeByte(int v) throws IOException {
        //System.out.println("📝 writeByte: " + v);
        indexOutputDelegate.writeByte((byte) v);
    }

    @Override
    public void writeShort(int v) throws IOException {
        //System.out.println("📝 writeShort: " + v);
        indexOutputDelegate.writeShort((short) v);
    }

    @Override
    public void writeChar(int v) throws IOException {
        throw new UnsupportedOperationException("JVectorRandomAccessWriter does not support writing chars");
    }

    @Override
    public void writeInt(int v) throws IOException {
        //System.out.println("📝 writeInt: " + v);
        indexOutputDelegate.writeInt(v);
    }

    @Override
    public void writeLong(long v) throws IOException {
        //System.out.println("📝 writeLong: " + v);
        indexOutputDelegate.writeLong(v);
    }

    @Override
    public void writeFloat(float v) throws IOException {
        //System.out.println("📝 writeFloat: " + v);
        ByteBuffer.wrap(writeBuffer).putFloat(v);
        indexOutputDelegate.writeBytes(writeBuffer, 0, Float.BYTES);
    }

    @Override
    public void writeDouble(double v) throws IOException {
        //System.out.println("📝 writeDouble: " + v);
        writeLong(Double.doubleToLongBits(v));
    }

    @Override
    public void writeBytes(String s) throws IOException {
        throw new UnsupportedOperationException("JVectorRandomAccessWriter does not support writing bytes");
    }

    @Override
    public void writeChars(String s) throws IOException {
        throw new UnsupportedOperationException("JVectorRandomAccessWriter does not support writing chars");
    }

    @Override
    public void writeUTF(String s) throws IOException {
        throw new UnsupportedOperationException("JVectorRandomAccessWriter does not support writing UTF strings");
    }
}

