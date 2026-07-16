"""Deprecated TabICL entrypoint.

The former implementation rebuilt DataFrames from DAF-preprocessed tensors.
TabICLv2 now runs only through the raw-frame native protocol.
"""


def main():
    raise SystemExit(
        "runners/run_tabicl.py is retired. Use runners/run_phase2_native.py "
        "with --model tabicl --mode final."
    )


if __name__ == "__main__":
    main()
