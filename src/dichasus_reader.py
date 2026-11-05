"""Minimal DICHASUS TFRecord reader that needs only numpy (no TensorFlow).

File layout (TFRecord):  uint64 length | uint32 crc | data[length] | uint32 crc
`data` is a serialized tf.train.Example. The fields csi / snr / pos-lidar are
tensors serialized with tf.io.serialize_tensor (a TensorProto).

For dichasus-005x every record has the same size, so record k starts at byte
k * record_size. That allows fetching a few records over HTTP Range requests
instead of downloading a whole 1.3-1.5 GB file.

CRC checks are skipped; shapes are checked instead.
"""
import struct
import urllib.error
import urllib.request

import numpy as np

DARUS_FILE_IDS = {  # DOI 10.18419/darus-2204
    "0053": 68398, "0054": 68399, "0055": 68400, "0056": 68401, "0057": 68402,
}
DARUS_URL = "https://darus.uni-stuttgart.de/api/access/datafile/{}"
_NP_DTYPES = {1: "<f4", 2: "<f8", 3: "<i4", 9: "<i8"}  # TensorFlow DataType enum


# ----------------------------------------------------------------- protobuf
def _varint(buf, i):
    result, shift = 0, 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if b < 0x80:
            return result, i
        shift += 7


def _fields(buf):
    """Yield (field_number, wire_type, value) of a protobuf message."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            v, i = buf[i:i + ln], i + ln
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
        yield fn, wt, v


def parse_tensor(buf):
    """TensorProto -> numpy array (only tensor_content encoding is needed here)."""
    dtype, shape, content = None, [], None
    for fn, _, v in _fields(buf):
        if fn == 1:
            dtype = v
        elif fn == 2:  # TensorShapeProto
            for fn2, _, dim in _fields(v):
                if fn2 == 2:
                    size = 0
                    for fn3, _, s in _fields(dim):
                        if fn3 == 1:
                            size = s
                    shape.append(size)
        elif fn == 4:
            content = v
    if dtype not in _NP_DTYPES or content is None:
        raise ValueError(f"unsupported tensor encoding (dtype={dtype})")
    return np.frombuffer(content, dtype=_NP_DTYPES[dtype]).reshape(shape)


def parse_example(buf):
    """tf.train.Example -> {name: bytes | float | int} (first value of each list)."""
    out = {}
    for fn, _, features in _fields(buf):
        if fn != 1:
            continue
        for fn2, _, entry in _fields(features):
            if fn2 != 1:
                continue
            key, val = None, None
            for fn3, _, x in _fields(entry):
                if fn3 == 1:
                    key = bytes(x).decode()
                elif fn3 == 2:
                    val = x
            for kind, wt_list, lst in _fields(val):
                vals = []
                for f5, wt5, b in _fields(lst):
                    if kind == 1:  # BytesList
                        vals.append(bytes(b))
                    elif kind == 2:  # FloatList (packed or not)
                        vals.extend(np.frombuffer(b, "<f4").tolist() if wt5 == 2
                                    else [struct.unpack("<f", b)[0]])
                    elif kind == 3:  # Int64List
                        if wt5 == 2:
                            j = 0
                            while j < len(b):
                                x, j = _varint(b, j)
                                vals.append(x)
                        else:
                            vals.append(b)
                out[key] = vals[0] if len(vals) == 1 else vals
    return out


def decode_record(data):
    """Decode one dichasus-005x record.

    csi is returned as complex64 (32, 1024) after fftshift along subcarriers,
    so DC sits at index 512 and the band edges at 0 / 1023 (as in the official
    DICHASUS tutorials).
    """
    ex = parse_example(memoryview(data))
    csi = parse_tensor(ex["csi"])
    if csi.shape != (32, 1024, 2):
        raise ValueError(f"unexpected csi shape {csi.shape}")
    csi = np.fft.fftshift(csi[..., 0] + 1j * csi[..., 1], axes=1).astype(np.complex64)
    snr = parse_tensor(ex["snr"]).astype(np.float32)
    pos = parse_tensor(ex["pos-lidar"]).astype(np.float64)
    # The files store "rot-lidar" (spec.json says "rot") and an extra per-antenna "cfo".
    rec = {"csi": csi, "snr": snr, "pos": pos,
           "time": float(ex.get("time", np.nan)),
           "rot": float(ex.get("rot-lidar", ex.get("rot", np.nan)))}
    if "cfo" in ex:
        rec["cfo"] = parse_tensor(ex["cfo"]).astype(np.float32)
    return rec


# --------------------------------------------------------------- TFRecord IO
def split_records(buf):
    """Split a byte string that starts at a record boundary into record payloads."""
    out, i = [], 0
    while i + 12 <= len(buf):
        (length,) = struct.unpack_from("<Q", buf, i)
        start = i + 12
        if start + length + 4 > len(buf):
            break
        out.append(buf[start:start + length])
        i = start + length + 4
    return out


def iter_file(path):
    """Iterate decoded records of a local .tfrecords file."""
    with open(path, "rb") as f:
        while True:
            hdr = f.read(12)
            if len(hdr) < 12:
                return
            (length,) = struct.unpack("<Q", hdr[:8])
            data = f.read(length)
            f.read(4)
            yield decode_record(data)


# -------------------------------------------------------- remote range access
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _get_range(url, start, end, timeout=120):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        total = r.headers.get("Content-Range", "").split("/")[-1]
        return r.read(), int(total) if total.isdigit() else None


def resolve_run(run):
    """Return (direct_url, file_size, record_size, n_records) for a run id like '0053'."""
    url = DARUS_URL.format(DARUS_FILE_IDS[run])
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        opener.open(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=60)
        direct = url
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise
        direct = e.headers["Location"]
    head, size = _get_range(direct, 0, 11)
    (length,) = struct.unpack("<Q", head[:8])
    rec_size = length + 16
    if size % rec_size:
        raise ValueError("records are not fixed-size; use sequential reading instead")
    return direct, size, rec_size, size // rec_size


def fetch_records(run, starts, chunk_len=1, retries=3):
    """Fetch `chunk_len` consecutive records at each start index of a run."""
    direct, _, rec_size, n_rec = resolve_run(run)
    out = []
    for s in starts:
        s = int(min(max(s, 0), n_rec - chunk_len))
        for attempt in range(retries):
            try:
                buf, _ = _get_range(direct, s * rec_size, (s + chunk_len) * rec_size - 1)
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == retries - 1:
                    raise
                direct, _, rec_size, n_rec = resolve_run(run)  # presigned URL may expire
        for k, data in enumerate(split_records(buf)):
            rec = decode_record(data)
            rec["run"], rec["index"] = run, s + k
            out.append(rec)
    return out, n_rec
