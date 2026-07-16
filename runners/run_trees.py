"""Deprecated Phase 2 tree entrypoint.

The former implementation performed full-data ordinal encoding and numerical
imputation. Those operations violate the Phase 2 native estimator protocol.
"""


def main():
    raise SystemExit(
        "runners/run_trees.py is retired. Use runners/run_phase2_native.py "
        "with --model xgboost or --model catboost."
    )


if __name__ == "__main__":
    main()
