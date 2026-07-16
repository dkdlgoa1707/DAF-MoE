"""Fail-fast validation for Phase 2 model-specific baseline configs."""


DAF_ONLY_FIELDS = {
    "n_experts",
    "top_k",
    "d_ff",
    "router_noise_std",
    "mu_init_strategy",
    "lambda_spec",
    "lambda_repel",
    "lambda_bal",
    "use_raw_path",
    "use_deep_path",
    "use_dist_token",
    "use_loss_free_balancing",
    "use_film_gating",
    "use_lightweight_preservation",
    "use_ple_embedding",
    "bias_update_rate",
}

PHASE2_NONRETRIEVAL = {
    "mlp",
    "resnet",
    "ft_transformer",
    "tabm",
    "tabm_ple",
    "tabr",
    "modernnca",
}


def _explicit_fields(config):
    return set(getattr(config, "explicit_fields", ()) or ())


def validate_model_config(config):
    model_name = config.model_name.lower()
    if model_name not in PHASE2_NONRETRIEVAL:
        return

    explicit = _explicit_fields(config)
    incompatible = sorted(explicit.intersection(DAF_ONLY_FIELDS))
    if model_name != "tabm_ple" and "ple_n_bins" in explicit:
        incompatible.append("ple_n_bins")
    if model_name == "tabm" and "ple_boundaries" in explicit:
        incompatible.append("ple_boundaries")
    if incompatible:
        raise ValueError(
            f"Fields incompatible with {model_name}: {sorted(set(incompatible))}. "
            "Use a model-specific Phase 2 config instead of a DAF best config."
        )

    if int(config.n_layers) <= 0:
        raise ValueError(f"{model_name} n_layers must be positive.")
    for name in ("dropout", "attention_dropout", "ffn_dropout", "residual_dropout"):
        value = float(getattr(config, name))
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1), got {value}.")

    if model_name == "ft_transformer" and int(config.n_heads) != 8:
        raise ValueError("Phase 2 FT-Transformer fixes n_heads=8.")
    if model_name in {"tabm", "tabm_ple"} and int(config.k) != 32:
        raise ValueError("Phase 2 TabM-mini fixes k=32.")
    if model_name == "tabm" and getattr(config, "ple_boundaries", None) is not None:
        raise ValueError("Plain TabM must not receive PLE boundaries.")
    if model_name == "tabr":
        if int(config.tabr_n_candidates) != 96:
            raise ValueError("Phase 2 full TabR fixes context size at 96.")
        if float(config.tabr_dropout1) != 0.0:
            raise ValueError("Phase 2 full TabR fixes dropout1=0.")
    if model_name == "modernnca":
        if float(config.nca_temperature) != 1.0:
            raise ValueError("Phase 2 ModernNCA fixes temperature=1.0.")
        if not 0.0 < float(config.nca_sample_rate) <= 1.0:
            raise ValueError("ModernNCA sample_rate must be in (0, 1].")
