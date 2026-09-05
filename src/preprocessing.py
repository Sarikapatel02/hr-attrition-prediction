"""Feature selection and preprocessing for HR attrition models."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import TARGET, validate_data

DEFAULT_DROP_COLUMNS = {TARGET, "EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"}


def build_preprocessor(data: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    clean = validate_data(data)
    features = clean.drop(columns=[column for column in DEFAULT_DROP_COLUMNS if column in clean])
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                 ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    transformer = ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)])
    return transformer, numeric, categorical


def save_processed(data: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "validated_data.csv", index=False)


def save_preprocessor(preprocessor: ColumnTransformer, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)