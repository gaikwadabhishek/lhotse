import atexit
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from urllib3.exceptions import TimeoutError

# Get a logger instance for this module
logger = logging.getLogger(__name__)

from lhotse.array import Array, TemporalArray
from lhotse.audio.recording import Recording
from lhotse.cut import CutSet
from lhotse.features.base import Features
from lhotse.image import Image
from lhotse.serialization import get_aistore_client
from lhotse.utils import is_module_available, is_valid_url

# Mapping between Lhotse file storage types and in-memory equivalents.
FILE_TO_MEMORY_TYPE = {
    "numpy_files": "memory_raw",
    "lilcom_files": "memory_lilcom",
    "pillow_files": "memory_pillow",
}

ARCHIVE_EXTENSIONS = (".tar.gz", ".tar", ".tgz")


class AISBatchLoaderError(Exception):
    """Base exception for AISBatchLoader operations."""


class AISBatchLoader:
    """
    Loads all data referenced by a :class:`CutSet` in a single AIStore Get-Batch call.

    The loader optimizes I/O by aggregating all object URLs from a CutSet and requesting
    them together. It offloads archive extraction and data slicing to AIStore, avoiding
    redundant downloads and local decompression.

    Example:
        >>> loader = AISBatchLoader()
        >>> cuts_with_data = loader(cuts)
    """

    def __init__(
        self, collect_metrics: bool = False, metrics_save_interval: int = 100
    ) -> None:
        """Initialize the AISBatchLoader with an AIStore client and batch context.

        Args:
            collect_metrics: If True, collect per-batch latency metrics.
                Also auto-enabled when LHOTSE_METRICS_DIR env var is set.
            metrics_save_interval: When LHOTSE_METRICS_DIR is set, auto-save
                metrics to disk every this many batches. Defaults to 100.
                Set to 0 to disable periodic saving (atexit only).
        """
        if not is_module_available("aistore"):
            raise ImportError(
                "Please run 'pip install aistore>=1.17.0' to use AISBatchLoader."
            )
        self.client, _ = get_aistore_client()

        # Auto-enable metrics if LHOTSE_METRICS_DIR is set
        self._metrics_dir = os.environ.get("LHOTSE_METRICS_DIR")
        if self._metrics_dir or collect_metrics:
            self.collect_metrics = True
            self._batch_latencies: Optional[List[float]] = []
            self._per_object_latencies: Optional[List[float]] = []
        else:
            self.collect_metrics = False
            self._batch_latencies = None
            self._per_object_latencies = None

        self._metrics_save_interval = metrics_save_interval
        self._batch_count = 0

        # Auto-save on process exit if metrics dir is configured
        if self._metrics_dir and self.collect_metrics:
            atexit.register(self._auto_save_metrics)

    def _get_object_from_moss_in(self, moss_in: Any) -> bytes:
        """
        Fetch a single object from AIStore using the ObjectNames request info.

        This method is used as a fallback when batch operations fail or return empty content.
        It handles archive extraction if an archpath is specified.

        Args:
            moss_in: AIStore ObjectNames request containing bucket, object, and optional archpath.

        Returns:
            The object content as bytes.

        Raises:
            Exception: If the object cannot be fetched from AIStore.
        """
        from aistore.sdk.archive_config import ArchiveConfig

        config = None
        if hasattr(moss_in, "archpath") and moss_in.archpath:
            config = ArchiveConfig(archpath=moss_in.archpath)

        reader = (
            self.client.bucket(bck_name=moss_in.bck, provider=moss_in.provider)
            .object(moss_in.obj_name)
            .get_reader(archive_config=config)
        )
        return reader.read_all()

    def __call__(self, cuts: CutSet) -> CutSet:
        """
        Fetch all data referenced by a CutSet in one AIStore batch operation.

        Args:
            cuts: A non-lazy CutSet representing a single batch of data.

        Returns:
            The same CutSet object with all manifests updated to reference in-memory data.

        Raises:
            ValueError: If the input CutSet is lazy.
            AISBatchLoaderError: For invalid URLs or unsupported storage types.
        """
        if cuts.is_lazy:
            raise ValueError(
                "Lazy CutSets cannot be used with AISBatchLoader. "
                "Convert to eager via `cuts.to_eager()` before loading."
            )

        if self.collect_metrics:
            _t_start = time.perf_counter()

        # Try to use Colocation if available (aistore >= 1.20.0)
        # Colocation optimizes batch requests by grouping objects from the same storage target,
        # reducing network overhead and improving throughput for distributed data retrieval.
        try:
            from aistore.sdk.enums import Colocation

            batch = self.client.batch(colocation=Colocation.TARGET_AWARE)
        except (ImportError, TypeError):
            # Fall back to creating batch without colocation parameter for older versions
            batch = self.client.batch()
        # Collect all URLs for get-batch and track which manifests have URLs
        manifest_list = []
        for cut in cuts:
            for _, manifest in cut.iter_data():
                has_url = self._collect_manifest_urls(manifest, batch)
                manifest_list.append((manifest, has_url))

        # Execute batch request
        from aistore.sdk.errors import AISError

        # Save requests list before calling batch.get() - it may be cleared after execution
        saved_requests_list = list(batch.requests_list)

        try:
            batch_result = batch.get()
        except ValueError as e:
            # ValueError occurs when the batch request is invalid or empty
            logger.warning(
                f"ValueError during batch.get(): {e}. Returning unmodified cuts."
            )
            return cuts
        except AISError as e:
            logger.warning(
                f"AIStore batch.get() failed: {e}. Falling back to sequential GET requests."
            )
            # Fallback: make sequential GET requests for each object in the batch
            # Use a generator to maintain consistency with batch.get() which returns an iterator
            def sequential_get():
                for moss_in in saved_requests_list:
                    try:
                        content = self._get_object_from_moss_in(moss_in)
                        yield (moss_in, content)
                    except Exception as ex:
                        logger.error(
                            f"Failed to fetch object {moss_in.obj_name} from bucket "
                            f"{moss_in.provider}://{moss_in.bck}: {ex}"
                        )
                        raise AISBatchLoaderError(
                            f"Sequential GET fallback failed for {moss_in.obj_name}"
                        ) from ex

            batch_result = sequential_get()

        # Apply the received data back into each manifest that had a URL
        request_idx = 0
        for manifest, has_url in manifest_list:
            if has_url:
                info = None
                content = None

                try:
                    if self.collect_metrics:
                        _t_obj_start = time.perf_counter()
                    info, content = next(batch_result)
                    if self.collect_metrics:
                        self._per_object_latencies.append(
                            time.perf_counter() - _t_obj_start
                        )
                except StopIteration:
                    raise AISBatchLoaderError(
                        "Batch result iterator exhausted prematurely. "
                        f"Expected more objects for manifests with URLs."
                    )
                except TimeoutError as e:
                    # Timeout occurred - recover the request info from saved_requests_list
                    logger.warning(
                        f"Timeout while fetching batch result at index {request_idx}: {e}. "
                        f"Falling back to direct AIStore API call."
                    )

                    if request_idx < len(saved_requests_list):
                        info = saved_requests_list[request_idx]
                        content = b""  # Mark as empty to trigger retry
                    else:
                        raise AISBatchLoaderError(
                            f"Timeout at request index {request_idx}, but cannot recover: "
                            f"index out of range for saved_requests_list (len={len(saved_requests_list)})"
                        ) from e

                # Retry with direct API call if content is empty (from timeout or actual empty response)
                if content == b"":
                    logger.warning(
                        f"Object {info.obj_name}/{info.archpath} from bucket {info.provider}://{info.bck} "
                        f"returned empty content. Retrying with direct AIStore API call."
                    )
                    try:
                        content = self._get_object_from_moss_in(info)
                    except Exception as ex:
                        logger.error(
                            f"Failed to fetch object {info.obj_name} from bucket "
                            f"{info.provider}://{info.bck}: {ex}"
                        )
                        raise AISBatchLoaderError(
                            f"Direct API fallback failed for {info.obj_name}"
                        ) from ex

                self._inject_data_into_manifest(manifest, content)
                request_idx += 1

        if self.collect_metrics:
            self._batch_latencies.append(time.perf_counter() - _t_start)
            self._batch_count += 1

            # Periodically save metrics to disk so data survives worker termination
            if (
                self._metrics_dir
                and self._metrics_save_interval > 0
                and self._batch_count % self._metrics_save_interval == 0
            ):
                try:
                    self.save_metrics(self._metrics_dir)
                except Exception as e:
                    logger.warning(f"Failed to periodically save metrics: {e}")

        return cuts

    # ----------------------------- Metrics Methods -----------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Compute and return latency metrics from collected records.

        Returns a dict with keys: ``batch`` (P95, P99, avg latencies in seconds),
        ``per_object`` (same stats divided by object count), ``num_batches``,
        and ``total_objects``.

        Raises:
            RuntimeError: If metrics collection is not enabled or no data has
                been collected yet.
        """
        if not self.collect_metrics:
            raise RuntimeError(
                "Metrics collection is not enabled. "
                "Pass collect_metrics=True or set LHOTSE_METRICS_DIR."
            )
        if not self._batch_latencies:
            raise RuntimeError(
                "No metrics data collected yet. Call the loader on a CutSet first."
            )

        import numpy as np

        batch_arr = np.array(self._batch_latencies)
        obj_arr = (
            np.array(self._per_object_latencies)
            if self._per_object_latencies
            else np.array([0.0])
        )

        return {
            "batch": {
                "p50": float(np.median(batch_arr)),
                "p95": float(np.percentile(batch_arr, 95)),
                "p99": float(np.percentile(batch_arr, 99)),
                "avg": float(np.mean(batch_arr)),
            },
            "per_object": {
                "p50": float(np.median(obj_arr)),
                "p95": float(np.percentile(obj_arr, 95)),
                "p99": float(np.percentile(obj_arr, 99)),
                "avg": float(np.mean(obj_arr)),
            },
            "num_batches": len(self._batch_latencies),
            "total_objects": len(self._per_object_latencies),
        }

    def reset_metrics(self) -> None:
        """Clear all collected latency records."""
        if self.collect_metrics:
            self._batch_latencies = []
            self._per_object_latencies = []

    def save_metrics(self, path: str, rank: Optional[int] = None) -> None:
        """Save collected metrics to a JSON file.

        Args:
            path: Directory where the metrics file will be written.
            rank: Distributed rank. Defaults to ``int(os.environ.get("RANK", 0))``.
        """
        if rank is None:
            rank = int(os.environ.get("RANK", 0))

        metrics = self.get_metrics()
        metrics["rank"] = rank
        metrics["pid"] = os.getpid()

        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"rank_{rank}_pid_{os.getpid()}.json"
        final_path = out_dir / filename
        tmp_path = out_dir / f".{filename}.tmp"

        tmp_path.write_text(json.dumps(metrics, indent=2))
        tmp_path.rename(final_path)

    def _auto_save_metrics(self) -> None:
        """Save metrics on process exit (registered via atexit)."""
        try:
            if self._batch_latencies:
                self.save_metrics(self._metrics_dir)
        except Exception as e:
            logger.warning(f"Failed to auto-save metrics on exit: {e}")

    # ----------------------------- Internal Helpers -----------------------------

    def _collect_manifest_urls(self, manifest: Any, batch: Any) -> bool:
        """
        Add all URLs referenced in a manifest to the batch.

        Returns:
            True if URLs were added to the batch, False otherwise.
        """
        if isinstance(manifest, Recording):
            for source in manifest.sources:
                if source.type == "url":
                    self._add_url_to_batch(source.source, batch)
                    return True
            return False

        elif isinstance(manifest, TemporalArray):
            # TemporalArray wraps an Array, so we need to access the inner array
            inner_array = manifest.array
            if inner_array.storage_type not in FILE_TO_MEMORY_TYPE:
                raise AISBatchLoaderError(
                    f"Unsupported storage type '{inner_array.storage_type}'. "
                    f"Supported types: {list(FILE_TO_MEMORY_TYPE.keys())}"
                )

            obj_path = f"{inner_array.storage_path}/{inner_array.storage_key}"
            if is_valid_url(obj_path):
                self._add_url_to_batch(obj_path, batch)
                return True
            return False

        elif isinstance(manifest, (Array, Features, Image)):
            if manifest.storage_type not in FILE_TO_MEMORY_TYPE:
                raise AISBatchLoaderError(
                    f"Unsupported storage type '{manifest.storage_type}'. "
                    f"Supported types: {list(FILE_TO_MEMORY_TYPE.keys())}"
                )

            obj_path = f"{manifest.storage_path}/{manifest.storage_key}"
            if is_valid_url(obj_path):
                self._add_url_to_batch(obj_path, batch)
                return True
            return False

        return False

    def _add_url_to_batch(self, url: str, batch: Any) -> None:
        """Add a single AIStore URL to the batch request."""
        from aistore.sdk.utils import parse_url

        provider, bck_name, obj_name = parse_url(url)
        if not (provider and bck_name and obj_name):
            raise AISBatchLoaderError(f"Invalid object URL: '{url}'")

        arch_ext = self._get_archive_extension(obj_name)
        archpath = None
        if arch_ext and arch_ext in obj_name:
            prefix, _, suffix = obj_name.partition(f"{arch_ext}/")
            obj_name, archpath = prefix + arch_ext, suffix

        bucket = self.client.bucket(bck_name, provider)
        batch.add(bucket.object(obj_name), archpath=archpath)

    def _inject_data_into_manifest(self, manifest: Any, content: bytes) -> None:
        """Replace manifest storage references with in-memory content."""
        if isinstance(manifest, Recording):
            for source in manifest.sources:
                source.type = "memory"
                source.source = content

        elif isinstance(manifest, TemporalArray):
            # TemporalArray wraps an Array, so update the inner array
            inner_array = manifest.array
            inner_array.storage_type = FILE_TO_MEMORY_TYPE[inner_array.storage_type]
            inner_array.storage_path = ""
            inner_array.storage_key = content

        elif isinstance(manifest, (Array, Features, Image)):
            manifest.storage_type = FILE_TO_MEMORY_TYPE[manifest.storage_type]
            manifest.storage_path = ""
            manifest.storage_key = content

    @staticmethod
    def _get_archive_extension(obj_name: str) -> Optional[str]:
        """Return the supported archive extension if present in the object name."""
        for ext in ARCHIVE_EXTENSIONS:
            if ext in obj_name:
                return ext
        return None


def aggregate_metrics(path: str) -> Dict[str, Any]:
    """Aggregate metrics from all worker JSON files in a directory.

    Reads all ``rank_*_pid_*.json`` files produced by :meth:`AISBatchLoader.save_metrics`
    and computes global and per-rank statistics.

    Args:
        path: Directory containing the metrics JSON files.

    Returns:
        A dict with ``global`` and ``per_rank`` keys. The ``global`` dict contains
        aggregated batch/per_object P95/P99/avg, num_batches, total_objects,
        num_ranks, and num_workers. The ``per_rank`` dict is keyed by rank (int)
        and contains per-rank breakdowns.
    """
    import numpy as np

    metrics_dir = Path(path)
    files = sorted(metrics_dir.glob("rank_*_pid_*.json"))
    if not files:
        raise FileNotFoundError(f"No metrics files found in {path}")

    # Collect all per-file data grouped by rank
    per_rank_files: Dict[int, List[Dict]] = {}
    all_batch_latencies = []
    all_per_object_latencies = []
    total_batches = 0
    total_objects = 0

    for f in files:
        data = json.loads(f.read_text())
        rank = data["rank"]
        per_rank_files.setdefault(rank, []).append(data)

        # Reconstruct individual latencies from stats for global aggregation
        # We use the per-file stats directly
        total_batches += data["num_batches"]
        total_objects += data["total_objects"]
        all_batch_latencies.append(data["batch"])
        all_per_object_latencies.append(data["per_object"])

    # Build per-rank summary
    per_rank = {}
    for rank, file_data_list in sorted(per_rank_files.items()):
        rank_batch_avgs = [d["batch"]["avg"] for d in file_data_list]
        rank_batch_p95s = [d["batch"]["p95"] for d in file_data_list]
        rank_batch_p99s = [d["batch"]["p99"] for d in file_data_list]
        rank_po_avgs = [d["per_object"]["avg"] for d in file_data_list]
        rank_po_p95s = [d["per_object"]["p95"] for d in file_data_list]
        rank_po_p99s = [d["per_object"]["p99"] for d in file_data_list]
        rank_batches = sum(d["num_batches"] for d in file_data_list)
        rank_objects = sum(d["total_objects"] for d in file_data_list)

        per_rank[rank] = {
            "batch": {
                "p95": float(np.mean(rank_batch_p95s)),
                "p99": float(np.mean(rank_batch_p99s)),
                "avg": float(np.mean(rank_batch_avgs)),
            },
            "per_object": {
                "p95": float(np.mean(rank_po_p95s)),
                "p99": float(np.mean(rank_po_p99s)),
                "avg": float(np.mean(rank_po_avgs)),
            },
            "num_batches": rank_batches,
            "total_objects": rank_objects,
            "num_workers": len(file_data_list),
        }

    # Build global summary
    global_batch_avgs = [d["avg"] for d in all_batch_latencies]
    global_batch_p95s = [d["p95"] for d in all_batch_latencies]
    global_batch_p99s = [d["p99"] for d in all_batch_latencies]
    global_po_avgs = [d["avg"] for d in all_per_object_latencies]
    global_po_p95s = [d["p95"] for d in all_per_object_latencies]
    global_po_p99s = [d["p99"] for d in all_per_object_latencies]

    return {
        "global": {
            "batch": {
                "p95": float(np.mean(global_batch_p95s)),
                "p99": float(np.mean(global_batch_p99s)),
                "avg": float(np.mean(global_batch_avgs)),
            },
            "per_object": {
                "p95": float(np.mean(global_po_p95s)),
                "p99": float(np.mean(global_po_p99s)),
                "avg": float(np.mean(global_po_avgs)),
            },
            "num_batches": total_batches,
            "total_objects": total_objects,
            "num_ranks": len(per_rank_files),
            "num_workers": len(files),
        },
        "per_rank": per_rank,
    }
