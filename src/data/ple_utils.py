import numpy as np
import yaml

from pathlib import Path


def _ensure_strictly_increasing(boundaries, eps=1e-6):
    for idx in range(1, len(boundaries)):
        if boundaries[idx] <= boundaries[idx - 1]:
            boundaries[idx] = boundaries[idx - 1] + eps
    return boundaries


def compute_ple_boundaries(x_num_scaled: np.ndarray, n_bins: int) -> list:
    """
    Compute feature-wise quantile boundaries from z-scored training values.
    """
    if x_num_scaled.ndim != 2:
        raise ValueError("x_num_scaled must have shape [n_samples, n_numerical].")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    if x_num_scaled.shape[1] == 0:
        return []

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    boundaries = []
    for feature_idx in range(x_num_scaled.shape[1]):
        feature_bounds = np.quantile(x_num_scaled[:, feature_idx], quantiles).astype(float).tolist()
        boundaries.append(_ensure_strictly_increasing(feature_bounds))

    return boundaries


def inject_ple_boundaries_into_yaml(base_yaml_path: str, boundaries: list, out_path: str):
    """Inject PLE boundaries into a YAML config and write a runnable copy."""
    with open(base_yaml_path, 'r', encoding='utf-8') as source:
        config = yaml.safe_load(source)
    config['ple_boundaries'] = boundaries

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as target:
        yaml.safe_dump(config, target, sort_keys=False)
