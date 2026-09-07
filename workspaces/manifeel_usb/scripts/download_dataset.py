#!/usr/bin/env python3
"""Download with resume support and safely extract the official ManiFeel USB dataset.

The archive is a Hugging Face Git-LFS object.  Its LFS oid is used as the
authoritative SHA256 checksum; a completed file is never trusted by name alone.
Existing destination datasets are never removed or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


HF_REPO_ID = "purdue-mars/manifeel"
HF_REVISION = "d2f3bd1fa7eb38ee807d4f3df2c2a3a3821371ea"
ARCHIVE_RELPATH = "data/usb_quan_Aug05.zip"
ARCHIVE_NAME = "usb_quan_Aug05.zip"
DATASET_DIRNAME = "usb_quan_Aug05"
ARCHIVE_SIZE = 3_991_771_935
ARCHIVE_SHA256 = "25e7912dec28282a2294adc34f819be59a01cebc73ec21501d938dc78f64cb00"
DOWNLOAD_URL = (
    f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/{HF_REVISION}/"
    f"{ARCHIVE_RELPATH}?download=true"
)

EXPECTED_ZARR = {
    "data/state": ([5976, 7], "<f4"),
    "data/action": ([5976, 6], "<f4"),
    "data/tactile_force_field_right": ([5976, 10, 14, 3], "<f4"),
    "data/wrist": ([5976, 256, 256, 3], "<f4"),
    "meta/episode_ends": ([50], "<i8"),
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _content_range_start(value: str | None) -> int | None:
    if not value:
        return None
    match = re.fullmatch(r"bytes\s+(\d+)-\d+/(?:\d+|\*)", value.strip())
    return int(match.group(1)) if match else None


def _verified_file(path: Path, expected_size: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and sha256_file(path).lower() == expected_sha256.lower()
    )


def download_with_resume(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    token: str | None = None,
    retries: int = 8,
    chunk_size: int = 8 * 1024 * 1024,
    timeout_s: float = 60.0,
) -> Path:
    """Download ``url`` into ``destination`` using an atomic ``.part`` file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _verified_file(destination, expected_size, expected_sha256):
            return destination
        raise FileExistsError(
            f"Refusing to overwrite existing archive with a checksum/size mismatch: {destination}"
        )

    part = destination.with_name(f"{destination.name}.part")
    if part.exists() and part.stat().st_size > expected_size:
        raise RuntimeError(f"Partial archive is larger than the expected source object: {part}")

    headers = {"User-Agent": "lerobot-manifeel-usb/1.0", "Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        if offset == expected_size:
            break
        request_headers = dict(headers)
        if offset:
            request_headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=request_headers)

        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                status_code = getattr(response, "status", response.getcode())
                mode = "ab"
                if offset:
                    if status_code == 206:
                        returned_start = _content_range_start(response.headers.get("Content-Range"))
                        if returned_start != offset:
                            raise RuntimeError(
                                f"Server returned an invalid Content-Range for resume: "
                                f"expected {offset}, got {response.headers.get('Content-Range')!r}"
                            )
                    elif status_code == 200:
                        # Range was ignored. Restart only the disposable partial file.
                        offset = 0
                        mode = "wb"
                    else:
                        raise RuntimeError(f"Unexpected HTTP status {status_code} while resuming")
                elif status_code not in (200, 206):
                    raise RuntimeError(f"Unexpected HTTP status {status_code}")

                with part.open(mode) as stream:
                    while True:
                        block = response.read(chunk_size)
                        if not block:
                            break
                        stream.write(block)
                        if stream.tell() > expected_size:
                            raise RuntimeError("Downloaded more bytes than the pinned Git-LFS object size")
                    stream.flush()
                    os.fsync(stream.fileno())
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and part.exists() and part.stat().st_size == expected_size:
                break
            last_error = exc
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = exc

        current = part.stat().st_size if part.exists() else 0
        if current == expected_size:
            break
        if attempt >= retries:
            raise RuntimeError(
                f"Download did not complete after {retries + 1} attempts "
                f"({current}/{expected_size} bytes)"
            ) from last_error
        time.sleep(min(2**attempt, 30))

    actual_size = part.stat().st_size if part.exists() else 0
    if actual_size != expected_size:
        raise RuntimeError(f"Archive size mismatch: expected {expected_size}, got {actual_size}")
    actual_sha256 = sha256_file(part)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Archive SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}. "
            f"The partial file was retained for diagnosis: {part}"
        )
    os.replace(part, destination)
    return destination


def _read_zarray_metadata(root: Path, key: str) -> dict[str, Any]:
    metadata_path = root.joinpath(*key.split("/"), ".zarray")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Zarr array metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_extracted_zarr(root: Path, *, canonical: bool = True) -> dict[str, Any]:
    root = Path(root)
    if not (root / ".zgroup").is_file():
        raise ValueError(f"Not a Zarr group: {root}")
    arrays: dict[str, Any] = {}
    for key, (expected_shape, expected_dtype) in EXPECTED_ZARR.items():
        metadata = _read_zarray_metadata(root, key)
        shape = list(metadata.get("shape", []))
        dtype = metadata.get("dtype")
        if canonical and (shape != expected_shape or dtype != expected_dtype):
            raise ValueError(
                f"Unexpected {key} schema: shape={shape}, dtype={dtype}; "
                f"expected shape={expected_shape}, dtype={expected_dtype}"
            )
        arrays[key] = {
            "shape": shape,
            "chunks": metadata.get("chunks"),
            "dtype": dtype,
            "compressor": metadata.get("compressor"),
        }
    return arrays


def _safe_member_target(staging_parent: Path, member_name: str) -> Path:
    posix_path = PurePosixPath(member_name)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise RuntimeError(f"Unsafe path in ZIP archive: {member_name!r}")
    target = staging_parent.joinpath(*posix_path.parts).resolve()
    parent = staging_parent.resolve()
    if os.path.commonpath((str(parent), str(target))) != str(parent):
        raise RuntimeError(f"ZIP member escapes extraction directory: {member_name!r}")
    return target


def safe_extract_archive(archive: Path, output_dir: Path) -> Path:
    """Extract with ZIP-slip checks and per-member atomic, resumable writes."""
    archive = Path(archive)
    output_dir = Path(output_dir)
    final_root = output_dir / DATASET_DIRNAME
    if final_root.exists():
        manifest_path = final_root / "download_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(
                f"Refusing to modify existing dataset without its download manifest: {final_root}"
            )
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("source", {}).get("sha256") != ARCHIVE_SHA256:
            raise FileExistsError(f"Existing dataset was not produced from the pinned archive: {final_root}")
        validate_extracted_zarr(final_root)
        return final_root

    staging_parent = output_dir / f".{DATASET_DIRNAME}.extracting"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staged_root = staging_parent / DATASET_DIRNAME

    with zipfile.ZipFile(archive, "r") as bundle:
        members = bundle.infolist()
        for info in members:
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(unix_mode):
                raise RuntimeError(f"Symbolic links are not accepted in the dataset ZIP: {info.filename}")
            target = _safe_member_target(staging_parent, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == info.file_size:
                # Completed members are atomically renamed below, so a same-size
                # file is a safe checkpoint from an earlier interrupted extraction.
                continue
            temporary = target.with_name(f".{target.name}.part")
            with bundle.open(info, "r") as source, temporary.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
                sink.flush()
                os.fsync(sink.fileno())
            if temporary.stat().st_size != info.file_size:
                raise RuntimeError(f"Extracted size mismatch for {info.filename}")
            os.replace(temporary, target)

    arrays = validate_extracted_zarr(staged_root)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "source": {
            "repository": HF_REPO_ID,
            "revision": HF_REVISION,
            "path": ARCHIVE_RELPATH,
            "url": DOWNLOAD_URL,
            "bytes": ARCHIVE_SIZE,
            "sha256": ARCHIVE_SHA256,
        },
        "archive": str(archive.resolve()),
        "zip_member_count": len(members),
        "dataset": {"episodes": 50, "frames": 5976, "zarr_arrays": arrays},
    }
    atomic_write_json(staged_root / "download_manifest.json", manifest)
    if final_root.exists():
        raise FileExistsError(f"Destination appeared during extraction; refusing to overwrite: {final_root}")
    os.replace(staged_root, final_root)
    try:
        staging_parent.rmdir()
    except OSError:
        pass
    return final_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will contain the pinned ZIP and extracted usb_quan_Aug05 Zarr group.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Prefer the HF_TOKEN environment variable.",
    )
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--no-extract", action="store_true", help="Only download and verify the ZIP.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.retries < 0:
        raise ValueError("--retries must be non-negative")
    archive = download_with_resume(
        DOWNLOAD_URL,
        args.output_dir / ARCHIVE_NAME,
        expected_size=ARCHIVE_SIZE,
        expected_sha256=ARCHIVE_SHA256,
        token=args.token,
        retries=args.retries,
        timeout_s=args.timeout_s,
    )
    print(f"Verified archive: {archive}")
    if not args.no_extract:
        dataset = safe_extract_archive(archive, args.output_dir)
        print(f"Verified Zarr dataset: {dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
