"""Downloading and verifying pinned artifacts that live outside the HuggingFace cache."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from ..logging_utils import get_logger
from .base import MetricError

__all__ = ["fetch_pinned", "sha256_file"]

logger = get_logger(__name__)

#: Seconds to wait on a stalled download before failing. Affects only how long a failure takes.
_DOWNLOAD_TIMEOUT_S = 60

_BLOCK_BYTES = 1 << 24


def sha256_file(path: Path) -> str:
    """Hex SHA-256 digest of a file, read in blocks.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_pinned(url: str, sha256: str, directory: Path, filename: str | None = None) -> Path:
    """Return a local copy of ``url``, downloading it once and verifying its digest every time.

    Args:
        url: Address of the artifact, pinned to a commit or by ``sha256``.
        sha256: Digest the artifact must match.
        directory: Cache directory; created if absent.
        filename: Name of the cached copy. None uses the last segment of ``url``, which suits
            addresses that end in a file name.

    Returns:
        Path to the verified file.

    Raises:
        MetricError: If the download fails or the digest does not match.
    """
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (filename or url.rsplit("/", 1)[-1])
    if target.is_file():
        if sha256_file(target) == sha256:
            return target
        logger.warning("cached %s does not match its pinned digest; downloading again", target)
    handle, temporary = tempfile.mkstemp(dir=directory, suffix=".part")
    os.close(handle)
    try:
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as response:
            Path(temporary).write_bytes(response.read())
        actual = sha256_file(Path(temporary))
        if actual != sha256:
            raise MetricError(f"{url} has sha256 {actual}, expected {sha256}")
        os.replace(temporary, target)
    except OSError as exc:
        raise MetricError(f"could not download {url}: {exc}") from exc
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    logger.info("downloaded and verified %s", target)
    return target
