/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.sandbox.codecs.ivfaster_evo;

import java.io.Closeable;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Batched {@code O_DIRECT} reads through Linux io_uring, issued with raw syscalls via the foreign
 * function API.
 *
 * <p>Used for the scattered fine-record reads of a rerank: one submission covers every record of
 * the batch, so the reads run in parallel on the device, and {@code O_DIRECT} keeps them out of the
 * page cache, where they would otherwise evict the RAM-resident coarse codes. Rings and destination
 * buffers are per thread and file-agnostic; each reader only owns its file descriptor.
 */
@SuppressWarnings("restricted")
final class UringDirectReader implements Closeable {
  private static final int ALIGNMENT = 4096, RING_ENTRIES = 1024;
  private static final long SYS_IO_URING_SETUP = 425, SYS_IO_URING_ENTER = 426;
  private static final long OFF_SQ_RING = 0L, OFF_CQ_RING = 0x8000000L, OFF_SQES = 0x10000000L;
  private static final int ENTER_GETEVENTS = 1, OP_READ = 22, FEAT_SINGLE_MMAP = 1;
  private static final int PROT_RW = 1 | 2, MAP_SHARED = 1, O_RDONLY = 0;
  private static final int SQE_SIZE = 64, CQE_SIZE = 16, PARAMS_SIZE = 120, EINTR = 4;
  private static final int O_DIRECT = directFlag();

  private static final MethodHandle SYSCALL, MMAP, OPEN, CLOSE;

  static {
    Linker linker = Linker.nativeLinker();
    SymbolLookup libc = linker.defaultLookup();
    ValueLayout.OfLong j = ValueLayout.JAVA_LONG;
    ValueLayout.OfInt i = ValueLayout.JAVA_INT;
    SYSCALL =
        linker.downcallHandle(
            libc.find("syscall").orElseThrow(),
            FunctionDescriptor.of(j, j, j, j, j, j, j, j),
            Linker.Option.firstVariadicArg(1));
    MMAP =
        linker.downcallHandle(
            libc.find("mmap").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, j, i, i, i, j));
    OPEN =
        linker.downcallHandle(
            libc.find("open").orElseThrow(),
            FunctionDescriptor.of(i, ValueLayout.ADDRESS, i),
            Linker.Option.firstVariadicArg(2));
    CLOSE = linker.downcallHandle(libc.find("close").orElseThrow(), FunctionDescriptor.of(i, i));
  }

  private static final ThreadLocal<Ring> RINGS = ThreadLocal.withInitial(Ring::create);
  private static final ThreadLocal<Buffer> BUFFERS = ThreadLocal.withInitial(Buffer::new);

  private final int fd;

  private UringDirectReader(int fd) {
    this.fd = fd;
  }

  /** Opens {@code path} for direct reads, failing if this platform or kernel lacks io_uring. */
  static UringDirectReader open(Path path) throws IOException {
    if (O_DIRECT == 0) throw new IOException("io_uring is unsupported on this platform");
    int fd;
    try (Arena arena = Arena.ofConfined()) {
      fd = (int) OPEN.invokeExact(arena.allocateFrom(path.toString()), O_RDONLY | O_DIRECT);
    } catch (Throwable t) {
      throw new IOException("open(O_DIRECT) failed for " + path, t);
    }
    if (fd < 0) throw new IOException("open(O_DIRECT) failed for " + path);
    try {
      RINGS.get();
    } catch (RuntimeException e) {
      closeFd(fd);
      throw new IOException("io_uring unavailable", e);
    }
    return new UringDirectReader(fd);
  }

  /**
   * Reads every {@code [positions[i], positions[i] + lengths[i])} range in one submission. The
   * returned segments alias a per-thread buffer and stay valid until this thread's next batch.
   */
  MemorySegment[] readBatch(long[] positions, int[] lengths) throws IOException {
    int n = positions.length;
    long[] start = new long[n], bufferOffset = new long[n];
    int[] alignedLength = new int[n], needed = new int[n];
    long total = 0;
    for (int k = 0; k < n; k++) {
      start[k] = positions[k] & -ALIGNMENT;
      long end = (positions[k] + lengths[k] + ALIGNMENT - 1) & -ALIGNMENT;
      alignedLength[k] = Math.toIntExact(end - start[k]);
      needed[k] = Math.toIntExact(positions[k] + lengths[k] - start[k]);
      bufferOffset[k] = total;
      total += alignedLength[k];
    }
    MemorySegment buffer = BUFFERS.get().ensure(total);
    Ring ring = RINGS.get();
    for (int done = 0; done < n; ) {
      int batch = Math.min(RING_ENTRIES, n - done);
      ring.read(fd, start, alignedLength, needed, buffer, bufferOffset, done, batch);
      done += batch;
    }
    MemorySegment[] out = new MemorySegment[n];
    for (int k = 0; k < n; k++) {
      out[k] = buffer.asSlice(bufferOffset[k] + positions[k] - start[k], lengths[k]);
    }
    return out;
  }

  @Override
  public void close() throws IOException {
    closeFd(fd);
  }

  private static void closeFd(int fd) throws IOException {
    try {
      int rc = (int) CLOSE.invokeExact(fd);
      if (rc != 0) throw new IOException("close failed");
    } catch (IOException e) {
      throw e;
    } catch (Throwable t) {
      throw new IOException("close failed", t);
    }
  }

  /** Returns this platform's {@code O_DIRECT} flag, or 0 where io_uring reads are unsupported. */
  private static int directFlag() {
    if (System.getProperty("os.name", "").toLowerCase(Locale.ROOT).startsWith("linux") == false) {
      return 0;
    }
    return switch (System.getProperty("os.arch", "")) {
      case "aarch64" -> 0x10000;
      case "amd64", "x86_64" -> 0x4000;
      default -> 0;
    };
  }

  private static long syscall(long n, long a, long b, long c, long d) throws Throwable {
    return (long) SYSCALL.invokeExact(n, a, b, c, d, 0L, 0L);
  }

  /** One thread's submission and completion rings. */
  private static final class Ring {
    final int ringFd, sqTail, sqArray, sqMask, cqHead, cqTail, cqCqes, cqMask;
    final MemorySegment sq, cq, sqes;

    private Ring(
        int ringFd, MemorySegment params, MemorySegment sq, MemorySegment cq, MemorySegment sqes) {
      this.ringFd = ringFd;
      this.sq = sq;
      this.cq = cq;
      this.sqes = sqes;
      sqTail = params.get(ValueLayout.JAVA_INT, 44);
      sqMask = sq.get(ValueLayout.JAVA_INT, params.get(ValueLayout.JAVA_INT, 48));
      sqArray = params.get(ValueLayout.JAVA_INT, 64);
      cqHead = params.get(ValueLayout.JAVA_INT, 80);
      cqTail = params.get(ValueLayout.JAVA_INT, 84);
      cqMask = cq.get(ValueLayout.JAVA_INT, params.get(ValueLayout.JAVA_INT, 88));
      cqCqes = params.get(ValueLayout.JAVA_INT, 100);
    }

    /** Sets up a ring and maps its queues; the ring lives as long as its thread. */
    static Ring create() {
      Arena arena = Arena.ofAuto();
      MemorySegment params = arena.allocate(PARAMS_SIZE);
      try {
        long ringFd = syscall(SYS_IO_URING_SETUP, RING_ENTRIES, params.address(), 0, 0);
        if (ringFd < 0) throw new IllegalStateException("io_uring_setup errno=" + -ringFd);
        int sqEntries = params.get(ValueLayout.JAVA_INT, 0);
        int cqEntries = params.get(ValueLayout.JAVA_INT, 4);
        boolean single = (params.get(ValueLayout.JAVA_INT, 20) & FEAT_SINGLE_MMAP) != 0;
        long sqSize = params.get(ValueLayout.JAVA_INT, 64) + (long) sqEntries * Integer.BYTES;
        long cqSize = params.get(ValueLayout.JAVA_INT, 100) + (long) cqEntries * CQE_SIZE;
        MemorySegment sq = map(single ? Math.max(sqSize, cqSize) : sqSize, ringFd, OFF_SQ_RING);
        MemorySegment cq = single ? sq : map(cqSize, ringFd, OFF_CQ_RING);
        MemorySegment sqes = map((long) sqEntries * SQE_SIZE, ringFd, OFF_SQES);
        return new Ring((int) ringFd, params, sq, cq, sqes);
      } catch (RuntimeException e) {
        throw e;
      } catch (Throwable t) {
        throw new IllegalStateException("io_uring setup failed", t);
      }
    }

    /** Submits {@code count} reads starting at entry {@code from} and waits for all of them. */
    void read(
        int fd,
        long[] start,
        int[] length,
        int[] needed,
        MemorySegment buffer,
        long[] bufferOffset,
        int from,
        int count)
        throws IOException {
      int tail = sq.get(ValueLayout.JAVA_INT, sqTail);
      for (int k = 0; k < count; k++) {
        int index = (tail + k) & sqMask;
        long sqe = (long) index * SQE_SIZE;
        sqes.asSlice(sqe, SQE_SIZE).fill((byte) 0);
        sqes.set(ValueLayout.JAVA_BYTE, sqe, (byte) OP_READ);
        sqes.set(ValueLayout.JAVA_INT, sqe + 4, fd);
        sqes.set(ValueLayout.JAVA_LONG, sqe + 8, start[from + k]);
        sqes.set(ValueLayout.JAVA_LONG, sqe + 16, buffer.address() + bufferOffset[from + k]);
        sqes.set(ValueLayout.JAVA_INT, sqe + 24, length[from + k]);
        sqes.set(ValueLayout.JAVA_LONG, sqe + 32, from + k);
        sq.set(ValueLayout.JAVA_INT, sqArray + (long) index * Integer.BYTES, index);
      }
      VarHandle.fullFence();
      sq.set(ValueLayout.JAVA_INT, sqTail, tail + count);
      VarHandle.fullFence();
      int submitted = 0, reaped = 0;
      IOException failure = null;
      while (reaped < count) {
        long ret;
        try {
          ret =
              syscall(
                  SYS_IO_URING_ENTER, ringFd, count - submitted, count - reaped, ENTER_GETEVENTS);
        } catch (Throwable t) {
          throw new IOException("io_uring_enter failed", t);
        }
        if (ret < 0 && ret != -EINTR) throw new IOException("io_uring_enter errno=" + -ret);
        if (ret > 0) submitted += (int) ret;
        VarHandle.fullFence();
        int head = cq.get(ValueLayout.JAVA_INT, cqHead), end = cq.get(ValueLayout.JAVA_INT, cqTail);
        for (; head != end; head++, reaped++) {
          long cqe = cqCqes + (long) (head & cqMask) * CQE_SIZE;
          int entry = (int) cq.get(ValueLayout.JAVA_LONG, cqe);
          int res = cq.get(ValueLayout.JAVA_INT, cqe + 8);
          if (res < needed[entry] && failure == null) {
            failure = new IOException("io_uring read res=" + res + " want=" + needed[entry]);
          }
        }
        VarHandle.fullFence();
        cq.set(ValueLayout.JAVA_INT, cqHead, head);
      }
      if (failure != null) throw failure;
    }

    private static MemorySegment map(long length, long fd, long offset) throws Throwable {
      MemorySegment p =
          (MemorySegment)
              MMAP.invokeExact(MemorySegment.NULL, length, PROT_RW, MAP_SHARED, (int) fd, offset);
      if (p.address() == -1L) throw new IllegalStateException("io_uring mmap failed");
      return p.reinterpret(length);
    }
  }

  /** A per-thread, 4096-aligned destination buffer that grows to the largest batch. */
  private static final class Buffer {
    MemorySegment segment = MemorySegment.NULL;

    MemorySegment ensure(long size) {
      if (segment.byteSize() < size) segment = Arena.ofAuto().allocate(size, ALIGNMENT);
      return segment;
    }
  }
}
