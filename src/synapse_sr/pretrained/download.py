import hashlib
import os
import pathlib
import tempfile
import urllib.request


def cache_dir():
    return pathlib.Path(os.environ.get("SYNAPSE_CACHE", pathlib.Path.home() / ".cache" / "synapse"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, expected_sha256, filename):
    """Download once into the cache and verify SHA-256 on every load."""
    dst = cache_dir() / filename
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        from synapse_sr import ui
        with tempfile.NamedTemporaryFile(delete=False, dir=dst.parent) as tmp:
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    bar = ui.DownloadBar(filename, int(r.headers.get("Content-Length") or 0) or None)
                    try:
                        for chunk in iter(lambda: r.read(1 << 20), b""):
                            tmp.write(chunk)
                            bar.update(len(chunk))
                    finally:
                        bar.close()
            except BaseException:
                tmp.close()
                os.unlink(tmp.name)
                raise
        os.replace(tmp.name, dst)
    got = sha256(dst)
    if got != expected_sha256:
        raise RuntimeError(f"checksum mismatch for {dst}: expected {expected_sha256}, got {got}; "
                           f"delete the file to re-download")
    return dst
