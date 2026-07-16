"""Deprecated pre-Phase-2 HPO entrypoint.

The former runner counted failed trials toward its target and converted
exceptions to worst-score COMPLETE trials. Both behaviors violate Phase 2.
"""


def main():
    raise SystemExit(
        "tune.py is retired for Phase 2. Use `python runners/run_phase2.py hpo ...`; "
        "the new engine counts only finite COMPLETE trials and seals test data."
    )


if __name__ == "__main__":
    main()
