"""Print saved model evaluation results."""

import json
from pathlib import Path


def main() -> None:
    path = Path("models/metrics.json")
    if not path.exists():
        raise FileNotFoundError("Run python -m src.train before evaluating")
    report = json.loads(path.read_text(encoding="utf-8"))
    print(f"Best model: {report['best_model']}")
    for name, metrics in report["models"].items():
        print(name, " ".join(f"{key}={value:.3f}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()