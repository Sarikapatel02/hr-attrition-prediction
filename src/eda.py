"""Create compact EDA summaries and figures."""

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from .data_loader import load_data


def create_eda_report(data_path: str, output_dir: str = "reports/figures") -> None:
    data = load_data(data_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = data.describe(include="all").transpose()
    summary.to_csv(output / "eda_summary.csv")
    rate = data.groupby("OverTime", observed=True)["Attrition"].mean().sort_values()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=rate.index, y=rate.values, color="#d97706")
    plt.ylabel("Attrition rate")
    plt.xlabel("Overtime")
    plt.tight_layout()
    plt.savefig(output / "attrition_by_overtime.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    args = parser.parse_args()
    create_eda_report(args.data)


if __name__ == "__main__":
    main()