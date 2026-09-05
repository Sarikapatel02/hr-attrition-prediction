"""Load, validate, and generate HR attrition datasets."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

TARGET = "Attrition"
REQUIRED_COLUMNS = {TARGET, "Age", "JobRole", "MonthlyIncome", "OverTime", "JobSatisfaction", "YearsAtCompany"}


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the minimum schema and normalize the binary target."""
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    result = data.copy()
    result[TARGET] = result[TARGET].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
    if result[TARGET].isna().any():
        raise ValueError("Attrition must contain only Yes/No or 1/0 values")
    if result[TARGET].nunique() < 2:
        raise ValueError("Attrition must contain both classes")
    return result


def load_data(path: str | Path) -> pd.DataFrame:
    """Read a CSV and validate its target and core feature columns."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return validate_data(pd.read_csv(csv_path))


def make_demo_data(rows: int = 400, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic dataset for local development and smoke tests."""
    rng = np.random.default_rng(seed)
    roles = np.array(["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager"])
    overtime = rng.choice(["Yes", "No"], rows, p=[0.3, 0.7])
    age = rng.integers(18, 61, rows)
    income = np.maximum(1000, rng.normal(5500, 2200, rows).astype(int))
    satisfaction = rng.integers(1, 5, rows)
    years = rng.integers(0, 21, rows)
    role = rng.choice(roles, rows)
    score = -1.4 + 1.1 * (overtime == "Yes") - 0.18 * satisfaction - 0.06 * years + 0.00004 * (7000 - income)
    probability = 1 / (1 + np.exp(-score))
    attrition = np.where(rng.random(rows) < probability, "Yes", "No")
    return pd.DataFrame({"Age": age, "JobRole": role, "MonthlyIncome": income, "OverTime": overtime,
                         "JobSatisfaction": satisfaction, "YearsAtCompany": years, TARGET: attrition})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-demo", action="store_true")
    parser.add_argument("--output", default="data/raw/demo_hr_attrition.csv")
    args = parser.parse_args()
    if args.generate_demo:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = make_demo_data()
        data.to_csv(output, index=False)
        print(f"Wrote {len(data)} demo rows to {output}")


if __name__ == "__main__":
    main()