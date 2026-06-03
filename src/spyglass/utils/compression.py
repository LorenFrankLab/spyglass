"""NWB dataset-level compression utilities.

Provides HDF5 dataset-level compression for NWB files, allowing transparent
chunk-level decompression on read without needing to decompress entire files.

Functions
---------
repack_nwb : Repack an NWB file with dataset-level compression.
compressed_data : Wrap data in H5DataIO with compression kwargs.
batch_repack : Batch-repack files tracked by NwbRepackJob.
repack_progress : Report compression campaign progress.
"""

import logging
import os
import tempfile
from pathlib import Path

import h5py
from hdmf.backends.hdf5.h5_utils import H5DataIO

logger = logging.getLogger("spyglass")

# Datasets smaller than this are not worth compressing
_MIN_DATASET_BYTES = 1024  # 1 KB

# Datasets larger than this are copied in row-chunks to limit memory
_CHUNK_COPY_THRESHOLD = 500 * 1024 * 1024  # 500 MB

# Default row-chunk size for large dataset copies
_COPY_CHUNK_ROWS = 50_000


def compressed_data(data, compression="gzip", compression_opts=4):
    """Wrap data in H5DataIO with compression kwargs for NWB writes.

    Parameters
    ----------
    data : array-like
        Data to wrap (numpy array, DataChunkIterator, etc.)
    compression : str
        HDF5 compression filter name ('gzip' or 'lzf').
    compression_opts : int or None
        Compression level. For gzip: 1-9. For lzf: None.

    Returns
    -------
    H5DataIO
        Wrapped data object that pynwb will write with compression.
    """
    if compression == "lzf":
        compression_opts = None
    return H5DataIO(
        data=data,
        compression=compression,
        compression_opts=compression_opts,
    )


def repack_nwb(
    input_path,
    output_path=None,
    compression="gzip",
    compression_opts=4,
    min_dataset_bytes=_MIN_DATASET_BYTES,
    verify=True,
):
    """Repack an NWB/HDF5 file with dataset-level compression.

    Recursively copies all groups and datasets from input to output,
    applying compression to datasets above a size threshold. Uses
    atomic temp+rename to avoid partial writes. Skips datasets that
    are already compressed or are scalar.

    Parameters
    ----------
    input_path : str or Path
        Path to the input NWB file.
    output_path : str or Path, optional
        Path for the repacked file. Defaults to overwriting input_path.
    compression : str
        HDF5 compression filter ('gzip' or 'lzf').
    compression_opts : int or None
        Compression level. For gzip: 1-9. For lzf: None.
    min_dataset_bytes : int
        Skip datasets smaller than this (bytes).
    verify : bool
        If True, verify shapes and dtypes match after repacking.

    Returns
    -------
    dict
        Statistics: original_size, repacked_size, compression_ratio,
        datasets_compressed, datasets_skipped.

    Raises
    ------
    FileNotFoundError
        If input_path does not exist.
    ValueError
        If verification fails.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    overwrite = output_path is None
    if overwrite:
        output_path = input_path

    output_path = Path(output_path)
    original_size = input_path.stat().st_size

    if compression == "lzf":
        compression_opts = None

    # Write to a temp file, then atomically rename
    temp_fd, temp_path = tempfile.mkstemp(
        suffix=".nwb.tmp", dir=output_path.parent
    )
    os.close(temp_fd)
    temp_path = Path(temp_path)

    stats = {"datasets_compressed": 0, "datasets_skipped": 0}

    try:
        # Two-pass approach to preserve HDF5 object references:
        # 1. Byte-copy entire file (preserves all references)
        # 2. Rewrite uncompressed datasets with compression
        import shutil

        shutil.copy2(str(input_path), str(temp_path))

        # Now rewrite datasets with compression
        _recompress_datasets(
            temp_path,
            compression=compression,
            compression_opts=compression_opts,
            min_bytes=min_dataset_bytes,
            stats=stats,
        )

        if verify:
            _verify_repack(input_path, temp_path)

        # Atomic rename
        temp_path.rename(output_path)

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    repacked_size = output_path.stat().st_size
    ratio = original_size / repacked_size if repacked_size > 0 else 0.0

    return {
        "original_size": original_size,
        "repacked_size": repacked_size,
        "compression_ratio": ratio,
        "datasets_compressed": stats["datasets_compressed"],
        "datasets_skipped": stats["datasets_skipped"],
    }


def _recompress_datasets(
    file_path, compression, compression_opts, min_bytes, stats
):
    """Rewrite uncompressed datasets with compression in-place.

    Opens the file in r+ mode, and for each uncompressed dataset
    above the size threshold, reads data, deletes the dataset, and
    recreates it with compression. Skips datasets with reference
    dtypes or scalar datasets.
    """
    targets = []
    total_datasets = [0]

    with h5py.File(file_path, "r") as f:
        _find_compressible(f, targets, min_bytes, total_datasets)

    stats["datasets_skipped"] = total_datasets[0] - len(targets)

    if not targets:
        return

    with h5py.File(file_path, "a") as f:
        for ds_path in targets:
            ds = f[ds_path]

            # Read data and attributes
            data = ds[()]
            attrs = dict(ds.attrs)
            parent_path = "/".join(ds_path.split("/")[:-1]) or "/"
            name = ds_path.split("/")[-1]
            parent = f[parent_path]

            # Delete and recreate with compression
            del parent[name]

            kwargs = {
                "compression": compression,
                "compression_opts": compression_opts,
                "shuffle": True,
            }

            if data.nbytes > _CHUNK_COPY_THRESHOLD:
                new_ds = parent.create_dataset(
                    name,
                    shape=data.shape,
                    dtype=data.dtype,
                    chunks=True,
                    **kwargs,
                )
                n_rows = data.shape[0]
                for start in range(0, n_rows, _COPY_CHUNK_ROWS):
                    end = min(start + _COPY_CHUNK_ROWS, n_rows)
                    new_ds[start:end] = data[start:end]
            else:
                new_ds = parent.create_dataset(name, data=data, **kwargs)

            # Restore attributes
            for k, v in attrs.items():
                new_ds.attrs[k] = v

            stats["datasets_compressed"] += 1


def _find_compressible(group, targets, min_bytes, total_count, prefix=""):
    """Recursively find datasets eligible for compression."""
    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Group):
            _find_compressible(item, targets, min_bytes, total_count, path)
        elif isinstance(item, h5py.Dataset):
            total_count[0] += 1
            # Skip scalar
            if item.shape == ():
                continue
            # Skip already compressed
            if item.compression is not None:
                continue
            # Skip reference-type datasets
            if h5py.check_dtype(ref=item.dtype):
                continue
            # Skip small datasets
            nbytes = item.size * item.dtype.itemsize
            if nbytes < min_bytes:
                continue
            targets.append(path)


def _verify_repack(original_path, repacked_path):
    """Verify that repacked file has matching shapes and dtypes.

    Parameters
    ----------
    original_path : str or Path
        Path to the original file.
    repacked_path : str or Path
        Path to the repacked file.

    Raises
    ------
    ValueError
        If any dataset shape or dtype does not match.
    """
    mismatches = []

    with h5py.File(original_path, "r") as orig:
        with h5py.File(repacked_path, "r") as repack:
            _walk_verify(orig, repack, "", mismatches)

    if mismatches:
        msg = "Repack verification failed:\n" + "\n".join(mismatches)
        raise ValueError(msg)


def _walk_verify(orig_group, repack_group, path, mismatches):
    """Recursively walk and verify datasets match."""
    for name in orig_group:
        full_path = f"{path}/{name}"
        if name not in repack_group:
            mismatches.append(f"Missing in repacked: {full_path}")
            continue

        orig_item = orig_group[name]
        repack_item = repack_group[name]

        if isinstance(orig_item, h5py.Group):
            if not isinstance(repack_item, h5py.Group):
                mismatches.append(
                    f"Type mismatch at {full_path}: " "group vs dataset"
                )
                continue
            _walk_verify(orig_item, repack_item, full_path, mismatches)
        elif isinstance(orig_item, h5py.Dataset):
            if not isinstance(repack_item, h5py.Dataset):
                mismatches.append(
                    f"Type mismatch at {full_path}: " "dataset vs group"
                )
                continue
            if orig_item.shape != repack_item.shape:
                mismatches.append(
                    f"Shape mismatch at {full_path}: "
                    f"{orig_item.shape} vs {repack_item.shape}"
                )
            if orig_item.dtype != repack_item.dtype:
                mismatches.append(
                    f"Dtype mismatch at {full_path}: "
                    f"{orig_item.dtype} vs {repack_item.dtype}"
                )


def batch_repack(
    restriction=None,
    max_files=50,
    compression="gzip",
    compression_opts=4,
    dry_run=True,
    worker_id="",
):
    """Batch-repack NWB files, tracking progress in NwbRepackJob.

    Finds unrepacked files, inserts pending jobs, and processes them
    with status tracking. Largest files are processed first.

    Parameters
    ----------
    restriction : dict or str, optional
        DataJoint restriction on Nwbfile to limit scope.
    max_files : int
        Maximum number of files to process in this batch.
    compression : str
        HDF5 compression filter ('gzip' or 'lzf').
    compression_opts : int
        Compression level for gzip (1-9).
    dry_run : bool
        If True, report what would be repacked without doing it.
    worker_id : str
        Identifier for this worker (for multi-worker coordination).

    Returns
    -------
    list of dict
        Results for each file processed.
    """
    from datetime import datetime

    from spyglass.common.common_file_tracking import (
        CompressedNwbfile,
        NwbRepackJob,
    )
    from spyglass.common.common_nwbfile import Nwbfile

    # Find files not yet repacked
    base = Nwbfile()
    if restriction:
        base = base & restriction
    already = CompressedNwbfile().fetch("nwb_file_name")
    pending_in_job = NwbRepackJob().fetch("nwb_file_name")
    exclude = set(already) | set(pending_in_job)

    candidates = []
    for row in base.fetch(as_dict=True):
        name = row["nwb_file_name"]
        if name in exclude:
            continue
        path = Nwbfile.get_abs_path(name)
        if os.path.exists(path):
            size = os.path.getsize(path)
            candidates.append((name, path, size))

    # Sort by size descending (largest first)
    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:max_files]

    if dry_run:
        total_gb = sum(c[2] for c in candidates) / 1e9
        logger.info(
            f"[DRY RUN] Would repack {len(candidates)} files "
            f"({total_gb:.1f} GB total)"
        )
        return [
            {"nwb_file_name": c[0], "size_gb": c[2] / 1e9} for c in candidates
        ]

    # Insert pending jobs
    for name, path, size in candidates:
        NwbRepackJob.insert1(
            {
                "nwb_file_name": name,
                "status": "pending",
                "original_size_bytes": size,
                "worker_id": worker_id,
            },
            skip_duplicates=True,
        )

    results = []
    for name, path, size in candidates:
        key = {"nwb_file_name": name}

        # Claim the job
        (NwbRepackJob & key)._update("status", "in_progress")
        (NwbRepackJob & key)._update(
            "started_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        try:
            stats = repack_nwb(
                input_path=path,
                compression=compression,
                compression_opts=compression_opts,
                verify=True,
            )

            (NwbRepackJob & key)._update("status", "completed")
            (NwbRepackJob & key)._update(
                "repacked_size_bytes", stats["repacked_size"]
            )
            (NwbRepackJob & key)._update(
                "compression_ratio", stats["compression_ratio"]
            )
            (NwbRepackJob & key)._update(
                "completed_at",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            # Also register in CompressedNwbfile
            CompressedNwbfile().make(key, param_id=None)

            results.append(
                {"nwb_file_name": name, "status": "completed", **stats}
            )

        except Exception as e:
            (NwbRepackJob & key)._update("status", "failed")
            (NwbRepackJob & key)._update("error_message", str(e)[:512])
            logger.error(f"Failed to repack {name}: {e}")
            results.append(
                {"nwb_file_name": name, "status": "failed", "error": str(e)}
            )

    return results


def repack_progress():
    """Report compression campaign progress.

    Returns
    -------
    dict
        total, pending, in_progress, completed, failed, skipped,
        total_original_gb, total_repacked_gb, total_saved_gb.
    """
    from spyglass.common.common_file_tracking import NwbRepackJob

    if not NwbRepackJob():
        return {"total": 0}

    entries = NwbRepackJob.fetch(as_dict=True)
    by_status = {}
    for e in entries:
        by_status.setdefault(e["status"], []).append(e)

    completed = by_status.get("completed", [])
    total_orig = sum(e["original_size_bytes"] or 0 for e in completed)
    total_repack = sum(e["repacked_size_bytes"] or 0 for e in completed)

    return {
        "total": len(entries),
        "pending": len(by_status.get("pending", [])),
        "in_progress": len(by_status.get("in_progress", [])),
        "completed": len(completed),
        "failed": len(by_status.get("failed", [])),
        "skipped": len(by_status.get("skipped", [])),
        "total_original_gb": total_orig / 1e9,
        "total_repacked_gb": total_repack / 1e9,
        "total_saved_gb": (total_orig - total_repack) / 1e9,
    }
