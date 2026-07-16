"""Bounded-memory primitives shared by TabR and ModernNCA."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.provenance import stable_hash


class PLREmbeddingsLite(nn.Module):
    """Official PLR-lite: periodic features, shared linear map, ReLU."""

    def __init__(
        self,
        n_features,
        n_frequencies,
        frequency_scale,
        d_embedding,
    ):
        super().__init__()
        if n_features <= 0:
            raise ValueError("PLR requires at least one numerical feature.")
        self.frequencies = nn.Parameter(
            torch.empty(n_features, n_frequencies)
        )
        bound = 3.0 * float(frequency_scale)
        nn.init.trunc_normal_(
            self.frequencies,
            mean=0.0,
            std=float(frequency_scale),
            a=-bound,
            b=bound,
        )
        self.linear = nn.Linear(2 * n_frequencies, d_embedding)
        self.activation = nn.ReLU()

    def forward(self, values):
        periodic = 2.0 * math.pi * self.frequencies.unsqueeze(0) * values.unsqueeze(-1)
        periodic = torch.cat([torch.cos(periodic), torch.sin(periodic)], dim=-1)
        return self.activation(self.linear(periodic))


class OneHotWithUnknownIgnored(nn.Module):
    """One-hot train categories; the reserved unknown ID maps to all zeros."""

    def __init__(self, train_cardinalities):
        super().__init__()
        self.train_cardinalities = tuple(int(value) for value in train_cardinalities)

    @property
    def output_dim(self):
        return sum(self.train_cardinalities)

    def forward(self, values):
        encoded = []
        for index, cardinality in enumerate(self.train_cardinalities):
            feature = values[:, index]
            encoded.append(
                F.one_hot(feature.clamp_max(cardinality), cardinality + 1)[..., :-1]
            )
        if not encoded:
            return values.new_empty(values.shape[0], 0, dtype=torch.float32)
        return torch.cat(encoded, dim=-1).to(torch.float32)


class CandidateStore:
    """Train-only candidate tensors retained on CPU and transferred by chunk."""

    def __init__(self, inputs, targets):
        if not targets.numel():
            raise ValueError("Retrieval candidate store cannot be empty.")
        row_ids = inputs.get("row_ids")
        if row_ids is None:
            row_ids = torch.arange(len(targets), dtype=torch.long)
        self.inputs = {
            name: value.detach().to("cpu").contiguous()
            for name, value in inputs.items()
            if name != "row_ids"
        }
        self.targets = targets.detach().to("cpu").contiguous().reshape(-1)
        self.row_ids = row_ids.detach().to("cpu").long().contiguous().reshape(-1)
        if len(self.targets) != len(self.row_ids):
            raise ValueError("Candidate targets and row IDs must have equal length.")
        if any(len(value) != len(self.targets) for value in self.inputs.values()):
            raise ValueError("All candidate input tensors must have equal length.")
        if self.row_ids.unique().numel() != len(self.row_ids):
            raise ValueError("Train candidate row IDs must be unique.")

    def __len__(self):
        return len(self.targets)

    @property
    def provenance(self):
        return {
            "candidate_count": len(self),
            "row_id_hash": stable_hash(self.row_ids.numpy()),
            "storage_device": "cpu",
        }

    def iter_chunks(self, chunk_size, indices=None):
        indices = (
            torch.arange(len(self), dtype=torch.long)
            if indices is None
            else indices.detach().to("cpu").long()
        )
        for positions in indices.split(int(chunk_size)):
            yield (
                positions,
                {name: value[positions] for name, value in self.inputs.items()},
                self.targets[positions],
                self.row_ids[positions],
            )

    def gather(self, indices, device):
        shape = indices.shape
        flat = indices.detach().to("cpu").long().reshape(-1)
        inputs = {
            name: value[flat].to(device).reshape(*shape, *value.shape[1:])
            for name, value in self.inputs.items()
        }
        targets = self.targets[flat].to(device).reshape(shape)
        row_ids = self.row_ids[flat].to(device).reshape(shape)
        return inputs, targets, row_ids


def streaming_topk(
    query_keys,
    query_row_ids,
    store,
    encode_chunk,
    k,
    chunk_size,
    candidate_indices=None,
    squared=True,
):
    """Exact brute-force top-k while only one candidate chunk is on device."""
    if k <= 0:
        raise ValueError("Retrieval context size must be positive.")
    device = query_keys.device
    best_distances = None
    best_indices = None
    query_ids = None if query_row_ids is None else query_row_ids.to(device).long()

    with torch.no_grad():
        for positions, cpu_inputs, _, cpu_row_ids in store.iter_chunks(
            chunk_size, candidate_indices
        ):
            chunk_inputs = {name: value.to(device) for name, value in cpu_inputs.items()}
            candidate_keys = encode_chunk(chunk_inputs)
            distances = torch.cdist(query_keys, candidate_keys, p=2)
            if squared:
                distances = distances.square()
            if query_ids is not None:
                candidate_ids = cpu_row_ids.to(device)
                distances = distances.masked_fill(
                    query_ids[:, None] == candidate_ids[None, :], torch.inf
                )
            chunk_k = min(k, distances.shape[1])
            chunk_distances, local_indices = distances.topk(
                chunk_k, dim=1, largest=False
            )
            global_indices = positions.to(device)[local_indices]
            if best_distances is None:
                best_distances, best_indices = chunk_distances, global_indices
            else:
                merged_distances = torch.cat([best_distances, chunk_distances], dim=1)
                merged_indices = torch.cat([best_indices, global_indices], dim=1)
                keep = min(k, merged_distances.shape[1])
                best_distances, order = merged_distances.topk(
                    keep, dim=1, largest=False
                )
                best_indices = merged_indices.gather(1, order)

    if best_distances is None or torch.isinf(best_distances).all(dim=1).any():
        raise ValueError("No valid retrieval candidate remains after row-ID exclusion.")
    valid_width = int((~torch.isinf(best_distances)).sum(dim=1).min().item())
    return best_distances[:, :valid_width], best_indices[:, :valid_width]
