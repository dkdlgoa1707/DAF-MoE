# Phase 2 Retrieval and TabM Upstream Audit

Audit date: 2026-07-16

## Pinned sources

| Model | Canonical source | Commit | License | Local reference |
|---|---|---|---|---|
| TabR | `yandex-research/tabular-dl-tabr` | `17baa9082506f8e7a0f8d11bb1e08212926a1507` | MIT | `bin/tabr.py` |
| ModernNCA | `LAMDA-Tabular/TALENT` | `08301d670a7c854bcf3a73298763484ba58eecdb` | MIT | `TALENT/model/models/modernNCA.py`, `TALENT/model/methods/modernNCA.py` |
| TabM | `yandex-research/tabm` | `28e47ae301c92ec37787dde1ce923a0793f405b4` | Apache-2.0 | `tabm.py:init_scaling_`, `TabM.__init__` |

The local code keeps the repository's existing adapters and HPO ranges. Only
the execution semantics and initialization listed below are changed.

## Difference audit

| Concern | Local before Prompt 2 | Pinned upstream | Required correction |
|---|---|---|---|
| TabR feature encoder | PLR-lite, missing mask, train-fitted one-hot | PLR-style numerical module and one-hot, then shared encoder | Keep Phase 2 adapter/encoder contract |
| TabR retrieval | CPU candidate store with chunked `torch.cdist` top-k | exact FAISS `IndexFlatL2`/`GpuIndexFlatL2` rebuilt every forward | use exact FAISS in production; retain brute force only as a test oracle |
| TabR index refresh | candidate keys recomputed per chunk each forward | candidate keys and index refreshed every forward | preserve every-forward refresh and record it |
| TabR self-exclusion | row-ID mask inside every brute-force chunk | search `k+1`, mask the query's own index, retain `k` | implement row-ID-safe `k+1` removal |
| TabR gradient flow | search keys no-grad; selected keys re-encoded with grad | memory-efficient mode searches no-grad and re-encodes selected contexts with grad | preserve selected-context gradients |
| TabR candidate scope | fixed train-only candidate store | training candidates only; current batch removed/re-added for diagonal exclusion | keep train-only row provenance and explicit self-exclusion |
| TabR aggregation | squared-L2 similarity softmax, label/value mixer, predictor | same | no architectural change |
| ModernNCA feature encoder | PLR-lite, one-hot, linear and residual BN blocks | same high-level encoder and post-encoder BN blocks | keep Phase 2 adapter/encoder contract |
| ModernNCA SNS | uniform sample from all train rows, then row-ID mask | remove query batch, uniformly sample remaining train rows, re-add query batch | match official query-batch inclusion and diagonal exclusion |
| ModernNCA normalization | candidate representations encoded independently by chunks while training | sampled candidates form one logical BN batch; evaluation uses all train candidates | encode each logical candidate set once before distance streaming |
| ModernNCA evaluation | exact streaming softmax over all train rows | exact full softmax over all train rows | retain exact math with stable streaming aggregation, no subsampling |
| ModernNCA gradient flow | chunk graphs accumulated during training | gradients flow through query and full sampled candidate representation | preserve full sampled-candidate graph without detach |
| ModernNCA prediction | Euclidean softmax; one-hot class or scalar target average | same | no mathematical change |
| TabM initialization | independent scalar for every final encoded dimension | `init_scaling_(..., chunks=d_features)` shares one draw inside each original feature representation | provide explicit original-feature chunks |
| TabM plain chunks | value and missing indicator independently initialized | repository-specific requirement groups both as one source feature | group value plus missing indicator per numerical feature |
| TabM dagger chunks | every PLE dimension and missing indicator independent | official chunk initialization applied to feature embedding width | group all PLE dimensions plus missing indicator per numerical feature |
| TabM categorical chunks | every one-hot dimension independent | one chunk per categorical feature cardinality | group each feature's full encoded representation |
| TabM member aggregation | member-wise loss; probability/mean prediction aggregation | same | retain |

## Canonical execution decisions

- TabR production search is exact FAISS flat L2. CUDA uses
  `GpuIndexFlatL2`; CPU uses `IndexFlatL2`. Missing FAISS is an error, never a
  fallback.
- ModernNCA uses uniform SNS only while training. Validation and test use the
  complete train candidate set. Distance aggregation may be streamed, but the
  candidate representation is computed as one logical batch so chunk size
  cannot alter BatchNorm behavior.
- TabM's small initialization helper is locally adapted from the pinned
  Apache-2.0 `init_scaling_` chunk semantics. The repository-specific missing
  indicators are included in their source numerical feature chunks.
