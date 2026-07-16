"""Machine-readable constants for the Phase 2 experiment protocol."""


PROTOCOL_VERSION = "phase2-v2"
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

CUSTOM_NEURAL_METHODS = frozenset(
    {
        "daf_moe_v2",
        "mlp",
        "resnet",
        "ft_transformer",
        "tabr",
        "tabm",
        "tabm_ple",
        "modernnca",
    }
)
NATIVE_METHODS = frozenset({"xgboost", "catboost", "realmlp", "tabicl"})

MODEL_IMPLEMENTATION_VERSIONS = {
    "daf_moe_v2": "daf-moe-v2-local-v1",
    "xgboost": "xgboost-2.1.4-native-categorical",
    "catboost": "catboost-1.2.10-native",
    "mlp": "rtdl-mlp-phase2-v1",
    "resnet": "rtdl-resnet-phase2-v1",
    "ft_transformer": "rtdl-ft-transformer-phase2-v1",
    "tabr": "tabr-official-17baa908-faiss-1.14.1",
    "tabm": "tabm-mini-chunk-init-28e47ae-v1",
    "modernnca": "modernnca-talent-08301d6-v1",
    "realmlp": "pytabkit-realmlp-1.7.3",
    "tabicl": "tabicl-2.1.1-v2",
    "tabm_ple": "tabm-mini-ple-chunk-init-28e47ae-v1",
}

TARGET_POLICY_CLASS_MAPPING = "class_mapping"
TARGET_POLICY_STANDARDIZE = "standardize"
TARGET_POLICY_NATIVE = "native"


def _normalize_model_name(model_name):
    normalized = str(model_name).lower()
    return "daf_moe_v2" if normalized.startswith("daf_moe_v2") else normalized


def model_implementation_version(model_name):
    normalized = _normalize_model_name(model_name)
    try:
        return MODEL_IMPLEMENTATION_VERSIONS[normalized]
    except KeyError as exc:
        raise ValueError(f"No Phase 2 implementation version for {model_name}.") from exc


def resolve_target_policy(model_name, task_type):
    if task_type == "classification":
        return TARGET_POLICY_CLASS_MAPPING
    if task_type != "regression":
        raise ValueError(f"Unsupported task_type: {task_type}")
    normalized = _normalize_model_name(model_name)
    if normalized in CUSTOM_NEURAL_METHODS:
        return TARGET_POLICY_STANDARDIZE
    if normalized in NATIVE_METHODS:
        return TARGET_POLICY_NATIVE
    raise ValueError(f"No Phase 2 target policy for {model_name}.")
