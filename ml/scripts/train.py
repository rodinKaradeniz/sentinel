"""Train the baseline LightGBM and save the artifact.

Thin entrypoint — all logic lives in ml/src/. Run from the repo root:
    python -m ml.scripts.train
"""
from ml.src import models


def main() -> None:
    pipe, threshold, val_result = models.train_baseline()
    model_path, meta_path = models.save_baseline(pipe, threshold)

    print(val_result.summary())
    print(f"\nCost-optimal threshold (from val): {threshold:.3f}")
    print(f"Saved model:    {model_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
