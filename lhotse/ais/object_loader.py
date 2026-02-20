import atexit
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from lhotse.ais.common import (
    AISLoaderError,
    extract_manifest_url,
    inject_data_into_manifest,
    parse_ais_url,
)
from lhotse.cut import CutSet
from lhotse.serialization import get_aistore_client
from lhotse.utils import is_module_available


class AISObjectLoaderError(AISLoaderError):
    """Exception for AISObjectLoader operations."""


class AISObjectLoader:
    """
    Loads data referenced by a :class:`CutSet` via individual AIStore GET calls.

    Unlike :class:`AISBatchLoader` which uses the batch GET API, this loader
    makes individual GET requests per object. Concurrency is controlled via
    ``max_workers``.

    Example:
        >>> loader = AISObjectLoader(max_workers=4)
        >>> cuts_with_data = loader(cuts)
    """

    def __init__(
        self,
        max_workers: int = 1,
        collect_metrics: bool = False,
        metrics_save_interval: int = 100,
    ) -> None:
        """
        Initialize the AISObjectLoader.

        Args:
            max_workers: Number of concurrent fetch threads.
                1 (default) uses a simple sequential loop with no thread pool overhead.
                >1 uses a ThreadPoolExecutor for parallel fetches.
            collect_metrics: If True, collect per-object and per-batch latency
                metrics. Also auto-enabled when LHOTSE_METRICS_DIR env var is set.
            metrics_save_interval: When LHOTSE_METRICS_DIR is set, auto-save
                metrics to disk every this many batches. Defaults to 100.
                Set to 0 to disable periodic saving (atexit only).

        Raises:
            ImportError: If the ``aistore`` package is not installed.
            ValueError: If ``max_workers`` is less than 1.
        """
        if not is_module_available("aistore"):
            raise ImportError(
                "Please run 'pip install aistore>=1.17.0' to use AISObjectLoader."
            )
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self.max_workers = max_workers
        self.client, _ = get_aistore_client()

        # Auto-enable metrics if LHOTSE_METRICS_DIR is set
        self._metrics_dir = os.environ.get("LHOTSE_METRICS_DIR")
        if self._metrics_dir or collect_metrics:
            self.collect_metrics = True
            # Per-object: list of individual fetch times (seconds)
            self._object_latencies: Optional[List[float]] = []
            # Per-batch: list of (wall_clock_seconds, num_objects)
            self._batch_latencies: Optional[List[Tuple[float, int]]] = []
        else:
            self.collect_metrics = False
            self._object_latencies = None
            self._batch_latencies = None

        self._metrics_save_interval = metrics_save_interval
        self._batch_count = 0

        # Auto-save on process exit if metrics dir is configured
        if self._metrics_dir and self.collect_metrics:
            atexit.register(self._auto_save_metrics)

    def __call__(self, cuts: CutSet) -> CutSet:
        """
        Fetch all data referenced by a CutSet via individual GET requests.

        Args:
            cuts: A non-lazy CutSet representing a single batch of data.

        Returns:
            The same CutSet object with all manifests updated to reference in-memory data.

        Raises:
            ValueError: If the input CutSet is lazy.
            AISObjectLoaderError: For fetch failures or invalid URLs.
        """
        if cuts.is_lazy:
            raise ValueError(
                "Lazy CutSets cannot be used with AISObjectLoader. "
                "Convert to eager via `cuts.to_eager()` before loading."
            )

        if self.collect_metrics:
            _t_batch_start = time.perf_counter()

        # Collect (manifest, url) pairs
        fetch_tasks: List[Tuple[Any, str]] = []
        for cut in cuts:
            for _, manifest in cut.iter_data():
                url = extract_manifest_url(manifest)
                if url is not None:
                    fetch_tasks.append((manifest, url))

        if not fetch_tasks:
            return cuts

        if self.max_workers == 1:
            self._fetch_sequential(fetch_tasks)
        else:
            self._fetch_concurrent(fetch_tasks)

        if self.collect_metrics:
            batch_elapsed = time.perf_counter() - _t_batch_start
            self._batch_latencies.append((batch_elapsed, len(fetch_tasks)))
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

    def _fetch_object(self, url: str) -> bytes:
        """
        Fetch a single object from AIStore.

        Args:
            url: The AIStore URL to fetch.

        Returns:
            The object content as bytes.

        Raises:
            AISObjectLoaderError: If the fetch fails.
        """
        try:
            from aistore.sdk.archive_config import ArchiveConfig
        except ImportError:
            ArchiveConfig = None

        try:
            provider, bck_name, obj_name, archpath = parse_ais_url(url)
        except AISLoaderError as e:
            raise AISObjectLoaderError(str(e)) from e

        config = None
        if ArchiveConfig and archpath:
            config = ArchiveConfig(archpath=archpath)

        try:
            reader = (
                self.client.bucket(bck_name=bck_name, provider=provider)
                .object(obj_name)
                .get_reader(archive_config=config)
            )
            return reader.read_all()
        except Exception as e:
            raise AISObjectLoaderError(
                f"Failed to fetch object '{obj_name}' from "
                f"{provider}://{bck_name}: {e}"
            ) from e

    def _timed_fetch_object(self, url: str) -> bytes:
        """Fetch a single object and record its latency."""
        t_start = time.perf_counter()
        content = self._fetch_object(url)
        self._object_latencies.append(time.perf_counter() - t_start)
        return content

    def _fetch_sequential(self, fetch_tasks: List[Tuple[Any, str]]) -> None:
        """Fetch objects sequentially, fail-fast on error."""
        fetch_fn = (
            self._timed_fetch_object if self.collect_metrics else self._fetch_object
        )
        for manifest, url in fetch_tasks:
            content = fetch_fn(url)
            inject_data_into_manifest(manifest, content)

    def _fetch_concurrent(self, fetch_tasks: List[Tuple[Any, str]]) -> None:
        """Fetch objects concurrently using a thread pool."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        fetch_fn = (
            self._timed_fetch_object if self.collect_metrics else self._fetch_object
        )
        errors: List[Exception] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_manifest = {
                executor.submit(fetch_fn, url): manifest
                for manifest, url in fetch_tasks
            }

            for future in as_completed(future_to_manifest):
                manifest = future_to_manifest[future]
                try:
                    content = future.result()
                    inject_data_into_manifest(manifest, content)
                except Exception as e:
                    errors.append(e)

        if errors:
            msg = f"{len(errors)} object(s) failed to fetch:\n"
            msg += "\n".join(f"  - {e}" for e in errors)
            raise AISObjectLoaderError(msg) from errors[0]

    # ----------------------------- Metrics Methods -----------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Compute and return latency metrics from collected records.

        Returns a dict with keys:
        - ``per_object``: P50, P95, P99, avg latencies (seconds) for individual
          object fetches (the primary measurement).
        - ``batch``: P50, P95, P99, avg wall-clock latencies (seconds) per
          __call__ invocation, plus per-object derived stats (batch time / num_objects).
        - ``num_batches``, ``total_objects``.

        Raises:
            RuntimeError: If metrics collection is not enabled or no data has
                been collected yet.
        """
        if not self.collect_metrics:
            raise RuntimeError(
                "Metrics collection is not enabled. "
                "Pass collect_metrics=True or set LHOTSE_METRICS_DIR."
            )
        if not self._object_latencies and not self._batch_latencies:
            raise RuntimeError(
                "No metrics data collected yet. Call the loader on a CutSet first."
            )

        import numpy as np

        result: Dict[str, Any] = {}

        # Per-object: direct measurement of each GET call
        if self._object_latencies:
            obj_lat = np.array(self._object_latencies)
            result["per_object"] = {
                "p50": float(np.median(obj_lat)),
                "p95": float(np.percentile(obj_lat, 95)),
                "p99": float(np.percentile(obj_lat, 99)),
                "avg": float(np.mean(obj_lat)),
            }
        else:
            result["per_object"] = {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0}

        # Per-batch: wall-clock time for entire __call__
        if self._batch_latencies:
            batch_times = np.array([t for t, _ in self._batch_latencies])
            batch_per_object = np.array(
                [t / n if n > 0 else 0.0 for t, n in self._batch_latencies]
            )
            result["batch"] = {
                "p50": float(np.median(batch_times)),
                "p95": float(np.percentile(batch_times, 95)),
                "p99": float(np.percentile(batch_times, 99)),
                "avg": float(np.mean(batch_times)),
                "per_object_avg": float(np.mean(batch_per_object)),
            }
        else:
            result["batch"] = {
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "avg": 0.0,
                "per_object_avg": 0.0,
            }

        result["num_batches"] = len(self._batch_latencies)
        result["total_objects"] = len(self._object_latencies)

        return result

    def reset_metrics(self) -> None:
        """Clear all collected latency records."""
        if self.collect_metrics:
            self._object_latencies = []
            self._batch_latencies = []

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
            if self._object_latencies or self._batch_latencies:
                self.save_metrics(self._metrics_dir)
        except Exception as e:
            logger.warning(f"Failed to auto-save metrics on exit: {e}")
