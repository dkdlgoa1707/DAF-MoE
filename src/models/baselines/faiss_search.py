"""Exact FAISS search used by the production TabR implementation.

Adapted from yandex-research/tabular-dl-tabr `bin/tabr.py` at commit
17baa9082506f8e7a0f8d11bb1e08212926a1507 (MIT license).
"""

from contextlib import nullcontext
import time

import torch


FAISS_DEPENDENCY = "faiss-gpu-cu12==1.14.1.post1"


def require_faiss():
    try:
        import faiss
        import faiss.contrib.torch_utils  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Full TabR requires FAISS; install " + FAISS_DEPENDENCY
        ) from exc
    return faiss


class ExactFaissL2Search:
    """Reusable exact flat-L2 index, reset and rebuilt on every search."""

    def __init__(self, measure_performance=False):
        self.index = None
        self.resources = None
        self.dimension = None
        self.device_type = None
        self.refresh_count = 0
        self.total_index_refresh_seconds = 0.0
        self.total_search_seconds = 0.0
        self.total_query_count = 0
        self.measure_performance = bool(measure_performance)

    def _make_index(self, dimension, device):
        faiss = require_faiss()
        if device.type == "cuda":
            if not hasattr(faiss, "GpuIndexFlatL2"):
                raise RuntimeError(
                    "The installed FAISS build has no CUDA flat index; "
                    f"install {FAISS_DEPENDENCY}."
                )
            self.resources = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = (
                int(device.index)
                if device.index is not None
                else torch.cuda.current_device()
            )
            self.index = faiss.GpuIndexFlatL2(self.resources, dimension, config)
            self.device_type = "cuda"
        else:
            self.index = faiss.IndexFlatL2(dimension)
            self.device_type = "cpu"
        self.dimension = dimension

    @property
    def index_type(self):
        if self.index is None:
            return "GpuIndexFlatL2_or_IndexFlatL2"
        return type(self.index).__name__

    @property
    def dependency_version(self):
        return str(require_faiss().__version__)

    def search(
        self,
        query_keys,
        candidate_keys,
        k,
        query_row_ids=None,
        candidate_row_ids=None,
    ):
        if k <= 0:
            raise ValueError("FAISS retrieval context size must be positive.")
        if not len(candidate_keys):
            raise ValueError("FAISS retrieval candidates cannot be empty.")
        query_keys = query_keys.detach().float().contiguous()
        candidate_keys = candidate_keys.detach().float().contiguous()
        device_context = (
            torch.cuda.device(candidate_keys.device)
            if candidate_keys.device.type == "cuda"
            else nullcontext()
        )
        with device_context:
            if (
                self.index is None
                or self.dimension != candidate_keys.shape[1]
                or self.device_type != candidate_keys.device.type
            ):
                self._make_index(candidate_keys.shape[1], candidate_keys.device)
            if self.measure_performance:
                if candidate_keys.device.type == "cuda":
                    torch.cuda.synchronize(candidate_keys.device)
                refresh_started = time.perf_counter()
            self.index.reset()
            self.index.add(candidate_keys)
            if self.measure_performance:
                if candidate_keys.device.type == "cuda":
                    torch.cuda.synchronize(candidate_keys.device)
                self.total_index_refresh_seconds += (
                    time.perf_counter() - refresh_started
                )
            self.refresh_count += 1

            search_width = min(
                len(candidate_keys),
                int(k) + (1 if query_row_ids is not None else 0),
            )
            if self.measure_performance:
                search_started = time.perf_counter()
            distances, indices = self.index.search(query_keys, search_width)
            if self.measure_performance:
                if query_keys.device.type == "cuda":
                    torch.cuda.synchronize(query_keys.device)
                self.total_search_seconds += time.perf_counter() - search_started
            self.total_query_count += len(query_keys)
        if not isinstance(distances, torch.Tensor):
            distances = torch.from_numpy(distances).to(query_keys.device)
            indices = torch.from_numpy(indices).to(query_keys.device)
        indices = indices.long()

        if query_row_ids is not None:
            if candidate_row_ids is None:
                raise ValueError("Candidate row IDs are required for self-exclusion.")
            candidate_row_ids = candidate_row_ids.to(query_keys.device).long()
            selected_row_ids = candidate_row_ids[indices]
            distances = distances.masked_fill(
                selected_row_ids == query_row_ids.to(query_keys.device).long()[:, None],
                torch.inf,
            )
            order = distances.argsort(dim=1)
            distances = distances.gather(1, order)
            indices = indices.gather(1, order)

        width = min(int(k), search_width)
        distances = distances[:, :width]
        indices = indices[:, :width]
        if torch.isinf(distances).any():
            raise ValueError("No valid TabR candidate remains after self-exclusion.")
        return distances, indices

    @property
    def queries_per_second(self):
        if self.total_search_seconds <= 0:
            return None
        return self.total_query_count / self.total_search_seconds
