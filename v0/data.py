"""IMDb dataset acquisition utilities (v0).

This module downloads (if needed), extracts, and validates the Stanford IMDb
dataset (aclImdb) into a local cache directory. The resulting directory is then
consumed by v0/v1/v2 without manual preparation.
"""

import os
import tarfile
import urllib.request
import shutil
from typing import Optional

STANFORD_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def _project_root_from_here() -> str:
    """Return the project root directory based on this file location.

    Returns:
        Absolute path to the project root (the parent directory of `v0/`).
    """
    # group/v0/data.py -> group/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _acl_dir_ok(acl_dir: str) -> bool:
    """Check whether an extracted `aclImdb` directory has the expected structure.

    Args:
        acl_dir: Path to the `aclImdb/` directory.

    Returns:
        True if the required train/test pos/neg subfolders exist, else False.
    """
    need = [
        os.path.join(acl_dir, "train", "pos"),
        os.path.join(acl_dir, "train", "neg"),
        os.path.join(acl_dir, "test", "pos"),
        os.path.join(acl_dir, "test", "neg"),
    ]
    return all(os.path.isdir(p) for p in need)


def _download(url: str, out_path: str) -> None:
    """Download a URL to a local file.

    Args:
        url: Source URL.
        out_path: Local destination path.

    Raises:
        RuntimeError: If the download fails.
    """
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def reporthook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, downloaded * 100.0 / total_size)
        print(
            f"\rDownloading... {pct:6.2f}% "
            f"({downloaded/1e6:.1f}/{total_size/1e6:.1f} MB)",
            end="",
        )

    print(f"Downloading IMDb dataset:\n  {url}\nTo:\n  {out_path}")
    urllib.request.urlretrieve(url, out_path, reporthook=reporthook)
    print("\nDownload complete.")


def _safe_extract_tar_gz(tar_path: str, dst_dir: str) -> None:
    """Safely extract a `.tar.gz` archive into a destination folder.

    This helper is intended to prevent path traversal by validating member paths
    before extraction.

    Args:
        tar_path: Path to the `.tar.gz` file.
        dst_dir: Destination directory for extraction.

    Raises:
        RuntimeError: If the archive appears unsafe or extraction fails.
    """

    def is_within_directory(directory: str, target: str) -> bool:
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        return os.path.commonpath([abs_directory]) == os.path.commonpath(
            [abs_directory, abs_target]
        )

    os.makedirs(dst_dir, exist_ok=True)
    print(f"Extracting:\n  {tar_path}\nTo:\n  {dst_dir}")

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = os.path.join(dst_dir, member.name)
            if not is_within_directory(dst_dir, member_path):
                raise RuntimeError(f"Unsafe path in tar: {member.name}")
        tar.extractall(dst_dir)

    print("Extraction complete.")


def ensure_aclImdb(
    cache_root: Optional[str] = None,
    url: str = STANFORD_URL,
    force_redownload: bool = False,
    keep_tar: bool = True,
) -> str:
    """Ensure the Stanford IMDb dataset (aclImdb) is present locally.

    If the dataset is not found in the cache directory, this function downloads the
    official archive, extracts it, and validates the resulting folder structure.

    Args:
        cache_root: Directory used for caching the downloaded archive and extracted data.
            If None, defaults to `<project_root>/data/cache`.
        url: Dataset URL (defaults to the official Stanford URL constant).
        force_redownload: If True, re-download and re-extract even if a cache exists.
        keep_tar: If False, delete the downloaded tarball after successful extraction.

    Returns:
        Absolute path to the extracted `aclImdb/` directory.

    Raises:
        RuntimeError: If the extracted directory structure is incomplete.
     
    """
    project_root = _project_root_from_here()

    if cache_root is None:
        cache_root = os.path.join(project_root, "data", "cache")
    cache_root = os.path.abspath(cache_root)

    acl_dir = os.path.join(cache_root, "aclImdb")
    tar_path = os.path.join(cache_root, "aclImdb_v1.tar.gz")

    if force_redownload and os.path.isdir(acl_dir):
        shutil.rmtree(acl_dir)

    if _acl_dir_ok(acl_dir):
        return acl_dir

    os.makedirs(cache_root, exist_ok=True)

    if force_redownload or not os.path.isfile(tar_path):
        _download(url, tar_path)

    _safe_extract_tar_gz(tar_path, cache_root)

    if not _acl_dir_ok(acl_dir):
        raise RuntimeError("aclImdb structure incomplete after extraction")

    if not keep_tar and os.path.isfile(tar_path):
        os.remove(tar_path)

    return acl_dir
