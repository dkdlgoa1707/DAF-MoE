# Phase 2 Retrieval Baselines

## Upstream References

- TabR follows `yandex-research/tabular-dl-tabr` as mirrored by TALENT commit
  `08301d670a7c854bcf3a73298763484ba58eecdb`.
- ModernNCA follows the model and method implementation in the same TALENT
  commit.
- PLR-lite follows the shared-linear periodic embedding used by TabR and
  ModernNCA in that source.

The local implementation replaces FAISS-owned candidate residency with an
exact chunked search so train candidates can remain on CPU. This changes the
search backend, not the squared-L2 neighbor ranking used by TabR.

## Candidate Boundary

Only the train loader is passed to `set_candidates` or `set_train_context`.
Features, targets, and stable row IDs are detached and stored by
`CandidateStore` on CPU. Validation and test labels are never accepted by the
store. Candidate count and the train row-ID hash are added to the run manifest.

TabR scans candidate keys in bounded chunks and transfers only selected
contexts back through the encoder with gradients. ModernNCA applies uniform
stochastic neighbor sampling during training and aggregates Euclidean soft-NCA
weights chunk by chunk. The configured chunk size bounds candidate tensors on
the accelerator; the full preprocessed candidate table remains CPU-backed.

## Self-Exclusion

Self-exclusion compares stable query and candidate row IDs. Feature equality or
zero distance is never used. Consequently, a duplicate-feature train row with a
different ID remains eligible, and a validation/test row identical to a train
row is not removed.

## Verification

`tests/test_phase2_retrieval_models.py` compares exact chunked retrieval with a
brute-force reference and covers duplicate rows, held-out boundaries,
classification/regression labels, uniform SNS, backward passes, and a mocked
large CPU candidate store. `analysis/verify_baselines.py` includes the same
row-ID and CPU-residency invariants in its baseline smoke check.
