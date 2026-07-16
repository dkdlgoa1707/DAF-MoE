"""Machine-readable constants for the Phase 2 experiment protocol."""


PROTOCOL_VERSION = "phase2-v1"
HPO_SEED = 42
FINAL_EVALUATION_SEEDS = tuple(range(43, 58))
N_HPO_COMPLETE_TRIALS = 50

DATASETS = (
    "california",
    "adult",
    "higgs_small",
    "covertype",
    "allstate",
    "bnp",
    "nhanes",
    "mimic3",
    "mimic4",
)

MAIN_METHODS = (
    "daf_moe_v2",
    "xgboost",
    "catboost",
    "mlp",
    "resnet",
    "ft_transformer",
    "tabr",
    "tabm",
    "modernnca",
    "realmlp",
    "tabicl",
)

SECONDARY_METHODS = ("tabm_ple",)
MAIN_RANK_INCLUDED = {
    **{method: True for method in MAIN_METHODS},
    "tabm_ple": False,
}
