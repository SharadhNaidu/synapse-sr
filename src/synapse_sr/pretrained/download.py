import hashlib
import os
import pathlib
import time
import urllib.error
import urllib.request

HF = "https://huggingface.co"
ATTEMPTS = 4


def cache_dir():
    return pathlib.Path(os.environ.get("SYNAPSE_CACHE", pathlib.Path.home() / ".cache" / "synapse"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path):
    """SHA-256 of ``path``, remembered in a ``.sha256`` sidecar keyed on size and modification time, so an unchanged
    checkpoint is hashed once instead of on every load. Any change to the file invalidates the sidecar."""
    path = pathlib.Path(path)
    st = path.stat()
    key = f"{st.st_size} {st.st_mtime_ns}"
    side = path.with_name(path.name + ".sha256")
    try:
        digest, stamp = side.read_text().split(" ", 1)
        if stamp.strip() == key and len(digest) == 64:
            return digest
    except (OSError, ValueError):
        pass
    digest = sha256(path)
    try:
        side.write_text(f"{digest} {key}")
    except OSError:                                       # read-only location: just do not remember
        pass
    return digest


def resolve(url):
    """Point Hugging Face URLs at ``$HF_ENDPOINT`` (a mirror such as https://hf-mirror.com) when it is set."""
    mirror = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    return mirror + url[len(HF):] if mirror and url.startswith(HF + "/") else url


def offline():
    return any(os.environ.get(k, "0") not in ("", "0") for k in ("SYNAPSE_OFFLINE", "HF_HUB_OFFLINE"))


def _get(url, part, filename):
    """Append to ``part`` from where a previous attempt stopped; the server may ignore the range and restart."""
    from synapse_sr import ui
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:                        # range past the end: the part file is already complete
            return
        raise
    with r:
        if have and r.status != 206:
            have = 0
        size = int(r.headers.get("Content-Length") or 0)
        bar = ui.DownloadBar(filename, (have + size) or None)
        bar.update(have)
        got = 0
        try:
            with open(part, "ab" if have else "wb") as f:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    f.write(chunk)
                    got += len(chunk)
                    bar.update(len(chunk))
        finally:
            bar.close()
    if size and got < size:                               # the connection dropped without an error
        raise ConnectionResetError(f"download stopped after {have + got} of {have + size} bytes")


def fetch(url, expected_sha256, filename):
    """Download once into the cache (resuming and retrying on network errors) and verify SHA-256."""
    dst = cache_dir() / filename
    if not dst.exists():
        if offline():
            raise FileNotFoundError(f"{dst} is not cached and offline mode is on (SYNAPSE_OFFLINE / HF_HUB_OFFLINE); "
                                    "download it once with network access or pass weights='path/to/model.safetensors'")
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        src = resolve(url)
        for attempt in range(1, ATTEMPTS + 1):
            try:
                _get(src, part, filename)
                break
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                    raise
                if attempt == ATTEMPTS:
                    raise ConnectionError(f"could not download {src} after {ATTEMPTS} attempts ({e}); the partial file "
                                          f"{part} is kept and the next run resumes it. Behind a firewall, set "
                                          "HF_ENDPOINT to a mirror or pass weights=") from e
                time.sleep(2 ** attempt)
        os.replace(part, dst)
    got = file_sha256(dst)
    if got != expected_sha256:
        raise RuntimeError(f"checksum mismatch for {dst}: expected {expected_sha256}, got {got}; "
                           f"delete the file to re-download")
    return dst
