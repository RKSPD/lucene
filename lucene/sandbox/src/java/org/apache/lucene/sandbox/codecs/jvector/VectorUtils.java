package org.apache.lucene.sandbox.codecs.jvector;

public final class VectorUtils {
    public static void validateNotZeroVector(byte[] vector, String context) {
        if (vector == null || vector.length == 0 || isZeroVector(vector)) {
            throw new IllegalArgumentException("Byte vector is null or empty");
        }
    }

    private static boolean isZeroVector(byte[] vector) {
        for (byte b : vector) if (b != 0) return false;
        return true;
    }
}
