"""Reader/writer for the gpt2.bin checkpoint format (shared by the exporter and tests).

Layout (all integers little-endian uint32, all tensor data little-endian float32):

    magic        4 bytes  b"GPT2"
    version      u32      1
    n_layer      u32
    n_head       u32
    n_embd       u32
    n_ctx        u32
    vocab_size   u32
    n_tensors    u32
    n_tensors x {
        name_len u32, name bytes (utf-8),
        ndim     u32, dims u32[ndim],
        offset   u32 x 2  (u64 byte offset of data from file start, lo then hi)
    }
    ...zero padding...
    tensor data, each tensor starting on a 64-byte boundary

Weights keep Hugging Face's Conv1D layout: linear weights are [in, out], so y = x @ W + b.
"""
import struct

import numpy as np

MAGIC = b"GPT2"
VERSION = 1
ALIGN = 64


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def expected_shapes(n_layer, n_embd, n_ctx, vocab_size):
    """Canonical tensor order and shapes. The C++ loader checks against the same list."""
    C = n_embd
    shapes = [("wte.weight", (vocab_size, C)), ("wpe.weight", (n_ctx, C))]
    for i in range(n_layer):
        p = f"h.{i}."
        shapes += [
            (p + "ln_1.weight", (C,)),
            (p + "ln_1.bias", (C,)),
            (p + "attn.c_attn.weight", (C, 3 * C)),
            (p + "attn.c_attn.bias", (3 * C,)),
            (p + "attn.c_proj.weight", (C, C)),
            (p + "attn.c_proj.bias", (C,)),
            (p + "ln_2.weight", (C,)),
            (p + "ln_2.bias", (C,)),
            (p + "mlp.c_fc.weight", (C, 4 * C)),
            (p + "mlp.c_fc.bias", (4 * C,)),
            (p + "mlp.c_proj.weight", (4 * C, C)),
            (p + "mlp.c_proj.bias", (C,)),
        ]
    shapes += [("ln_f.weight", (C,)), ("ln_f.bias", (C,))]
    return shapes


def write(path, config, tensors):
    """config: dict with n_layer, n_head, n_embd, n_ctx, vocab_size.
    tensors: dict name -> np.ndarray. Must contain exactly the expected tensors."""
    order = expected_shapes(config["n_layer"], config["n_embd"], config["n_ctx"], config["vocab_size"])
    names = [n for n, _ in order]
    missing = set(names) - set(tensors)
    extra = set(tensors) - set(names)
    if missing or extra:
        raise ValueError(f"tensor set mismatch: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")

    arrays = []
    for name, shape in order:
        a = np.ascontiguousarray(tensors[name], dtype="<f4")
        if a.shape != tuple(shape):
            raise ValueError(f"{name}: shape {a.shape} != expected {shape}")
        arrays.append((name, a))

    header = bytearray()
    header += MAGIC
    header += struct.pack(
        "<7I", VERSION, config["n_layer"], config["n_head"], config["n_embd"],
        config["n_ctx"], config["vocab_size"], len(arrays),
    )
    # Compute metadata size first so offsets can be filled in.
    meta_size = len(header)
    for name, a in arrays:
        meta_size += 4 + len(name.encode()) + 4 + 4 * a.ndim + 8
    offset = _align(meta_size)
    offsets = []
    for _, a in arrays:
        offsets.append(offset)
        offset = _align(offset + a.nbytes)

    for (name, a), off in zip(arrays, offsets):
        nb = name.encode()
        header += struct.pack("<I", len(nb)) + nb
        header += struct.pack("<I", a.ndim) + struct.pack(f"<{a.ndim}I", *a.shape)
        header += struct.pack("<Q", off)
    assert len(header) == meta_size

    with open(path, "wb") as f:
        f.write(header)
        for (_, a), off in zip(arrays, offsets):
            f.write(b"\0" * (off - f.tell()))
            f.write(a.tobytes())


def read(path):
    """Returns (config, dict name -> np.ndarray). Used by tests to cross-check the C++ loader."""
    buf = open(path, "rb").read()
    if buf[:4] != MAGIC:
        raise ValueError("bad magic")
    version, n_layer, n_head, n_embd, n_ctx, vocab_size, n = struct.unpack_from("<7I", buf, 4)
    config = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_ctx=n_ctx, vocab_size=vocab_size)
    pos = 32
    tensors = {}
    for _ in range(n):
        (ln,) = struct.unpack_from("<I", buf, pos); pos += 4
        name = buf[pos:pos + ln].decode(); pos += ln
        (ndim,) = struct.unpack_from("<I", buf, pos); pos += 4
        dims = struct.unpack_from(f"<{ndim}I", buf, pos); pos += 4 * ndim
        (off,) = struct.unpack_from("<Q", buf, pos); pos += 8
        count = int(np.prod(dims))
        tensors[name] = np.frombuffer(buf, dtype="<f4", count=count, offset=off).reshape(dims)
    return config, tensors
