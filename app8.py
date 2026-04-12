import io
import streamlit as st
import pandas as pd
import numpy as np
import io
import hashlib
import gc
import re
import warnings
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.formula.api as smf
import streamlit as st
from scipy import stats
from sklearn import datasets, metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@st.cache_data(show_spinner=False, ttl=3600, max_entries=2)
def load_large_file(file_bytes: bytes, file_name: str):
    ext = file_name.lower().split(".")[-1]

    bio = io.BytesIO(file_bytes)

    if ext == "csv":
        return pd.read_csv(bio, low_memory=False)

    elif ext in ["xlsx", "xls"]:
        return pd.read_excel(bio, engine="openpyxl")

    elif ext == "parquet":
        return pd.read_parquet(bio)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

# Optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF = None
    FPDF_AVAILABLE = False

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

st.set_page_config(page_title="Mr Ready", page_icon="📊", layout="wide")

APP_CSS = """
<style>
:root {
  --mr-green: #1f7a4d;
  --mr-green-soft: #e8f5ee;
  --mr-blue: #1e6fb8;
}
.stApp {
  background: linear-gradient(180deg, #f8fbf9 0%, #ffffff 100%);
}
.block-container {
  padding-top: 0.85rem;
  padding-bottom: 2rem;
}
.mr-top {
  padding: 1rem 1.2rem;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(31,122,77,.12), rgba(30,111,184,.08));
  border: 1px solid rgba(31,122,77,.18);
  margin-bottom: 1rem;
}
.mr-pill {
  display: inline-block;
  margin-right: .35rem;
  margin-top: .25rem;
  padding: .25rem .6rem;
  border-radius: 999px;
  background: #eef7f1;
  color: var(--mr-green);
  font-size: .88rem;
  font-weight: 600;
}
.small-label {
  font-size: 0.85rem;
  font-weight: 600;
  color: #1f7a4d;
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


# =========================================================
# DATA MODELS
# =========================================================
@dataclass
class DataProfile:
    rows: int
    cols: int
    numeric_cols: list
    categorical_cols: list
    datetime_cols: list
    missing_summary: pd.DataFrame
    duplicates: int
    memory_mb: float


# =========================================================
# BASIC HELPERS
# =========================================================
def safe_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.copy()
    s = s.where(~s.isna(), np.nan)
    s = s.astype(str).str.strip()
    s = s.replace(
        {
            r",": "",
            r"\$": "",
            r"%": "",
            r"£": "",
            r"€": "",
            r"\(": "-",
            r"\)": "",
        },
        regex=True,
    )
    s = s.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "none": np.nan,
            "NaN": np.nan,
            "null": np.nan,
            "NULL": np.nan,
        }
    )
    return pd.to_numeric(s, errors="coerce")


def split_columns(df: pd.DataFrame):
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in datetime_cols]
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]
    return numeric_cols, categorical_cols, datetime_cols


def infer_task_type(series: pd.Series) -> str:
    s = series.dropna()

    if s.empty:
        return "classification"

    if (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_categorical_dtype(s)
        or pd.api.types.is_bool_dtype(s)
    ):
        return "classification"

    if pd.api.types.is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce").dropna()
        if s_num.empty:
            return "classification"

        unique_count = s_num.nunique(dropna=True)
        integer_like = np.all(np.isclose(s_num, np.round(s_num)))
        if integer_like and unique_count <= 20:
            return "classification"
        return "regression"

    return "classification"


def classify_target_kind(series: pd.Series) -> str:
    task = infer_task_type(series)
    if task == "regression":
        return "numerical"

    non_null = series.dropna()
    unique_count = non_null.nunique()
    return "binary" if unique_count == 2 else "categorical"


def profile_data(df: pd.DataFrame) -> DataProfile:
    numeric_cols, categorical_cols, datetime_cols = split_columns(df)

    miss = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean() * 100).round(2).values,
            "dtype": df.dtypes.astype(str).values,
            "unique_values": [df[c].nunique(dropna=True) for c in df.columns],
        }
    ).sort_values(["missing_pct", "missing_count"], ascending=False)

    return DataProfile(
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        missing_summary=miss.reset_index(drop=True),
        duplicates=int(df.duplicated().sum()),
        memory_mb=float(df.memory_usage(deep=True).sum() / (1024 ** 2)),
    )


def normalize_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            s = out[col].copy()
            s = s.where(~s.isna(), np.nan)
            stripped = s.astype(str).str.strip()
            stripped = stripped.replace(
                {
                    "": np.nan,
                    "nan": np.nan,
                    "None": np.nan,
                    "none": np.nan,
                    "NaN": np.nan,
                    "null": np.nan,
                    "NULL": np.nan,
                }
            )

            parsed_num = safe_numeric(stripped)
            if parsed_num.notna().mean() >= 0.7:
                out[col] = parsed_num
                continue

            parsed_dt = pd.to_datetime(stripped, errors="coerce", utc=False)
            if parsed_dt.notna().mean() >= 0.6:
                out[col] = parsed_dt
                continue

            out[col] = stripped

    return out


# =========================================================
# DATA LOADING
# =========================================================
def get_preloaded_dataset(name: str) -> pd.DataFrame:
    if name == "Iris":
        data = datasets.load_iris(as_frame=True)
        df = data.frame.copy()
        df["species"] = df["target"].map(dict(enumerate(data.target_names)))
        return df.drop(columns=["target"])

    if name == "Wine":
        return datasets.load_wine(as_frame=True).frame.copy()

    if name == "Breast Cancer":
        data = datasets.load_breast_cancer(as_frame=True)
        df = data.frame.copy()
        df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
        return df.drop(columns=["target"])

    if name == "Diabetes":
        return datasets.load_diabetes(as_frame=True).frame.copy()

    return pd.DataFrame()


def robust_csv_read(raw: bytes) -> pd.DataFrame:
    if not raw:
        raise ValueError("The uploaded CSV file is empty.")

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    seps = [None, ",", ";", "\t", "|"]

    last_error = None

    for enc in encodings:
        for sep in seps:
            try:
                buffer = io.BytesIO(raw)
                if sep is None:
                    return pd.read_csv(buffer, encoding=enc, sep=None, engine="python")
                return pd.read_csv(buffer, encoding=enc, sep=sep)
            except Exception as e:
                last_error = e
                continue

    raise ValueError(f"CSV file could not be parsed. {last_error}")


def robust_excel_read(raw: bytes, file_name: str) -> pd.DataFrame:
    ext = file_name.rsplit(".", 1)[-1].lower()

    engine_map = {
        "xlsx": ["openpyxl", None],
        "xlsm": ["openpyxl", None],
        "xls": ["xlrd", None],
        "xlsb": ["pyxlsb", None],
    }

    if ext not in engine_map:
        raise ValueError("Unsupported Excel extension.")

    if not raw:
        raise ValueError("The uploaded Excel file is empty.")

    last_error = None

    for engine in engine_map[ext]:
        try:
            buffer = io.BytesIO(raw)
            if engine is not None:
                return pd.read_excel(buffer, engine=engine)
            return pd.read_excel(buffer)
        except Exception as e:
            last_error = e
            continue

    raise ValueError(f"Excel file could not be parsed. {last_error}")


@st.cache_data(show_spinner=False, ttl=3600, max_entries=2)
def load_uploaded_file_from_bytes(file_bytes: bytes, file_name: str):
    if file_bytes is None:
        return None

    file_name = file_name.lower().strip()

    if file_name.endswith(".csv"):
        df = robust_csv_read(file_bytes)

    elif file_name.endswith((".xlsx", ".xls", ".xlsm", ".xlsb")):
        df = robust_excel_read(file_bytes, file_name)

    else:
        raise ValueError("Unsupported file format. Please upload CSV, XLS, XLSX, XLSM, or XLSB.")

    if df is None or df.empty:
        raise ValueError("The uploaded file was read, but it contains no usable rows.")

    df.columns = [str(c).strip() for c in df.columns]
    df = normalize_uploaded_data(df)

    return df


def load_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None

    try:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if st.session_state.get("uploaded_file_hash") != file_hash:
            df = load_uploaded_file_from_bytes(file_bytes, uploaded_file.name)
            st.session_state["uploaded_file_hash"] = file_hash
            st.session_state["df"] = df
            gc.collect()

        return st.session_state.get("df")

    except Exception as e:
        st.error(
            f"File could not be read: {e}. "
            "For .xls files install xlrd. For .xlsb files install pyxlsb."
        )
        return None

@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def prepare_base_dataset(df: pd.DataFrame):
    base_df = df.copy()
    clean_df, plan_df = impute_dataframe(base_df)
    return base_df, clean_df, plan_df
    
# =========================================================
# CLEANING
# =========================================================
def impute_dataframe(df: pd.DataFrame):
    out = df.copy()
    plan = []
    numeric_cols, categorical_cols, datetime_cols = split_columns(out)

    for col in numeric_cols:
        missing_count = out[col].isna().sum()
        if missing_count == 0:
            continue

        missing_pct = round(out[col].isna().mean() * 100, 2)
        non_null = out[col].dropna()

        if non_null.empty:
            fill_value = 0
            strategy = "constant_0"
        else:
            skew_val = non_null.skew() if len(non_null) > 2 else 0.0
            strategy = "median" if abs(skew_val) > 0.5 else "mean"
            fill_value = non_null.median() if strategy == "median" else non_null.mean()

        out[col] = out[col].fillna(fill_value)
        plan.append(
            {
                "column": col,
                "type": "numerical",
                "missing_pct": missing_pct,
                "strategy": strategy,
            }
        )

    for col in categorical_cols:
        missing_count = out[col].isna().sum()
        if missing_count == 0:
            continue

        missing_pct = round(out[col].isna().mean() * 100, 2)
        mode = out[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"

        out[col] = out[col].fillna(fill_value)
        plan.append(
            {
                "column": col,
                "type": "categorical",
                "missing_pct": missing_pct,
                "strategy": "mode" if not mode.empty else "constant_unknown",
            }
        )

    for col in datetime_cols:
        missing_count = out[col].isna().sum()
        if missing_count == 0:
            continue

        missing_pct = round(out[col].isna().mean() * 100, 2)
        non_null = out[col].dropna()

        if non_null.empty:
            fill_value = pd.Timestamp("2000-01-01")
            strategy = "constant_default_date"
        else:
            fill_value = non_null.min()
            strategy = "min_date"

        out[col] = out[col].fillna(fill_value)
        plan.append(
            {
                "column": col,
                "type": "datetime",
                "missing_pct": missing_pct,
                "strategy": strategy,
            }
        )

    return out, pd.DataFrame(plan)


def leakage_check(df: pd.DataFrame, target: str) -> list[str]:
    warnings_list = []
    target_series = df[target]

    target_lower = target.lower().strip()
    target_non_null = target_series.dropna()

    for col in df.columns:
        if col == target:
            continue

        col_series = df[col]
        col_lower = str(col).lower().strip()

        if (
            target_lower in col_lower
            or any(flag in col_lower for flag in ["target", "label", "outcome", "predicted", "prediction"])
        ):
            warnings_list.append(f"Possible leakage pattern in column name: '{col}'.")

        unique_ratio = col_series.nunique(dropna=True) / max(len(col_series), 1)
        if col_series.nunique(dropna=True) == len(col_series.dropna()) and unique_ratio > 0.9:
            warnings_list.append(f"'{col}' looks like a unique ID column.")

        try:
            aligned = pd.DataFrame({"x": col_series, "y": target_series}).dropna()

            if not aligned.empty:
                same_ratio = (aligned["x"].astype(str) == aligned["y"].astype(str)).mean()
                if same_ratio >= 0.95:
                    warnings_list.append(f"'{col}' is almost identical to the target.")
        except Exception:
            pass

        try:
            if pd.api.types.is_numeric_dtype(col_series) and pd.api.types.is_numeric_dtype(target_series):
                aligned_num = df[[col, target]].dropna()
                if len(aligned_num) >= 3:
                    corr = aligned_num.corr(numeric_only=True).iloc[0, 1]
                    if pd.notna(corr) and abs(corr) > 0.98:
                        warnings_list.append(f"'{col}' has extremely high correlation with target ({corr:.3f}).")
        except Exception:
            pass

        try:
            if target_non_null.nunique() == 2:
                grouped = pd.DataFrame({"x": col_series, "y": target_series}).dropna()
                if not grouped.empty and grouped["x"].nunique() == grouped["y"].nunique():
                    mapping_check = grouped.groupby("x")["y"].nunique().max()
                    if mapping_check == 1 and grouped["x"].nunique() <= max(20, int(0.2 * len(grouped))):
                        warnings_list.append(f"'{col}' may encode the target too directly.")
        except Exception:
            pass

    return list(dict.fromkeys(warnings_list))


# =========================================================
# INTERPRETATION HELPERS (UPDATED)
# =========================================================
def numeric_distribution_interpretation(df: pd.DataFrame, col: str) -> str:
    s = safe_numeric(df[col]).dropna()

    if s.empty:
        return "There is not enough valid data to interpret this distribution."

    mean_val = float(s.mean())
    median_val = float(s.median())
    std_val = float(s.std()) if len(s) > 1 else 0.0
    min_val = float(s.min())
    max_val = float(s.max())
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1

    skew_val = float(stats.skew(s, bias=False)) if len(s) > 2 else 0.0
    kurt_val = float(stats.kurtosis(s, bias=False)) if len(s) > 3 else 0.0

    # Normality style interpretation
    if abs(skew_val) < 0.5 and abs(mean_val - median_val) < max(0.1 * abs(mean_val), 0.15) and abs(kurt_val) < 1.5:
        shape_text = "The distribution looks fairly close to normal."
    elif skew_val > 0.5:
        shape_text = "The distribution is positively skewed, so values stretch more to the right."
    elif skew_val < -0.5:
        shape_text = "The distribution is negatively skewed, so values stretch more to the left."
    else:
        shape_text = "The distribution is not perfectly normal and shows some uneven shape."

    # Center interpretation
    if abs(mean_val - median_val) < max(0.1 * abs(mean_val), 0.15):
        center_text = "The mean and median are close, which supports a more balanced shape."
    elif mean_val > median_val:
        center_text = "The mean is above the median, which supports right skew."
    else:
        center_text = "The mean is below the median, which supports left skew."

    # Spread interpretation
    if std_val < max(abs(mean_val) * 0.15, 0.25):
        spread_text = "The spread looks fairly tight."
    elif std_val < max(abs(mean_val) * 0.35, 0.75):
        spread_text = "The spread is moderate."
    else:
        spread_text = "The spread is wide, so the values vary a lot."

    # Outlier check
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((s < lower_bound) | (s > upper_bound)).sum())

    if outlier_count == 0:
        outlier_text = "No strong outlier signal is visible."
    elif outlier_count <= max(3, int(0.02 * len(s))):
        outlier_text = "A few possible outliers are present."
    else:
        outlier_text = "Several possible outliers are present."

    return (
        f"For '{col}', values range from {min_val:.2f} to {max_val:.2f}. "
        f"The mean is {mean_val:.2f} and the median is {median_val:.2f}. "
        f"{shape_text} {center_text} {spread_text} {outlier_text}"
    )


def categorical_chart_insight(df: pd.DataFrame, col: str) -> str:
    s = df[col].dropna()

    if s.empty:
        return "There is not enough data to interpret this chart."

    counts = s.value_counts()
    top_label = counts.index[0]
    top_count = counts.iloc[0]
    total = counts.sum()
    top_pct = (top_count / total) * 100

    return (
        f"In '{col}', '{top_label}' is most common "
        f"with {top_count} rows ({top_pct:.1f}%)."
    )


def business_chart_story(df, chart_type, x_col, y_col, target):
    if df is None or df.empty or x_col not in df.columns:
        return "Not enough data to interpret this chart."

    # Pie / donut style charts
    if chart_type in ["Pie Chart", "Donut Chart"]:
        value_col = y_col if y_col in df.columns else None

        if value_col is None:
            counts = df[x_col].dropna().value_counts()
            if counts.empty:
                return f"Not enough data to interpret the distribution of '{x_col}'."

            top_cat = counts.idxmax()
            low_cat = counts.idxmin()
            return (
                f"'{top_cat}' has the highest share in '{x_col}', "
                f"while '{low_cat}' has the lowest share."
            )

        work = df[[x_col, value_col]].copy()
        work[value_col] = safe_numeric(work[value_col])
        work = work.dropna(subset=[x_col, value_col])

        if work.empty:
            return f"Not enough data to interpret '{value_col}' across '{x_col}'."

        grouped = (
            work.groupby(x_col, dropna=False)[value_col]
            .sum()
            .reset_index()
            .sort_values(value_col, ascending=False)
        )

        if grouped.empty:
            return f"Not enough data to interpret '{value_col}' across '{x_col}'."

        top_row = grouped.iloc[0]
        low_row = grouped.iloc[-1]

        return (
            f"'{top_row[x_col]}' has the highest {value_col} at {top_row[value_col]:.2f}, "
            f"while '{low_row[x_col]}' has the lowest {value_col} at {low_row[value_col]:.2f}."
        )

    # Bar / column / line / area / stacked / combo
    if y_col and y_col in df.columns:
        work = df[[x_col, y_col]].copy()
        work[y_col] = safe_numeric(work[y_col])
        work = work.dropna(subset=[x_col, y_col])

        if work.empty:
            return f"Not enough data to interpret '{y_col}' across '{x_col}'."

        grouped = (
            work.groupby(x_col, dropna=False)[y_col]
            .mean()
            .reset_index()
            .sort_values(y_col, ascending=False)
        )

        if grouped.empty:
            return f"Not enough data to interpret '{y_col}' across '{x_col}'."

        top_row = grouped.iloc[0]
        low_row = grouped.iloc[-1]

        if chart_type == "Line Chart":
            return (
                f"'{top_row[x_col]}' shows the highest {y_col} at {top_row[y_col]:.2f}, "
                f"while '{low_row[x_col]}' shows the lowest {y_col} at {low_row[y_col]:.2f}."
            )

        return (
            f"'{top_row[x_col]}' is higher in {y_col} at {top_row[y_col]:.2f}, "
            f"while '{low_row[x_col]}' is lower in {y_col} at {low_row[y_col]:.2f}."
        )

    counts = df[x_col].dropna().value_counts()
    if counts.empty:
        return f"Not enough data to interpret '{x_col}'."

    top_cat = counts.idxmax()
    low_cat = counts.idxmin()

    return (
        f"'{top_cat}' appears most often in '{x_col}', "
        f"while '{low_cat}' appears least often."
    )


def business_interpretation_paragraph(df, target, chart_name, focus_col=None):
    s = df[target].dropna()

    if s.empty:
        return "Not enough data to interpret."

    if chart_name == "target_distribution":
        vc = s.value_counts(normalize=True) * 100
        return f"The largest group in '{target}' is '{vc.index[0]}' at {vc.iloc[0]:.1f}%."

    if chart_name == "numeric_by_target" and focus_col:
        try:
            grouped = df[[target, focus_col]].dropna()
            medians = grouped.groupby(target)[focus_col].median().sort_values(ascending=False)
            return (
                f"'{focus_col}' differs across '{target}'. "
                f"Highest median: '{medians.index[0]}', lowest: '{medians.index[-1]}'."
            )
        except Exception:
            return "Could not compute group comparison."

    if chart_name == "trend" and focus_col:
        return f"'{focus_col}' over time shows the trend direction."

    return "This chart highlights the main pattern."


def residual_interpretation(y_true, y_pred):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    residuals = y_true - y_pred

    if residuals.empty:
        return "Not enough data for residual analysis."

    mean_resid = residuals.mean()
    abs_mean = residuals.abs().mean()

    if abs(mean_resid) < max(abs_mean * 0.1, 1e-6):
        bias_text = "No strong bias."
    elif mean_resid > 0:
        bias_text = "Slight underprediction."
    else:
        bias_text = "Slight overprediction."

    denom = max(np.abs(y_pred).mean(), 1e-6)
    spread_ratio = residuals.std() / denom

    if spread_ratio < 0.2:
        spread_text = "Errors are tight."
    elif spread_ratio < 0.5:
        spread_text = "Errors are moderate."
    else:
        spread_text = "Errors are wide."

    return f"{bias_text} {spread_text}"


def roc_interpretation(roc_auc):
    if roc_auc >= 0.9:
        return "Very strong ROC."
    if roc_auc >= 0.8:
        return "Strong ROC."
    if roc_auc >= 0.7:
        return "Decent ROC."
    return "Weak ROC."


def pr_interpretation(pr_auc):
    if pr_auc >= 0.9:
        return "Very strong PR."
    if pr_auc >= 0.8:
        return "Strong PR."
    if pr_auc >= 0.7:
        return "Fair PR."
    return "Weak PR."


def forecast_interpretation(history_df: pd.DataFrame, forecast_df: pd.DataFrame, value_col: str):
    if history_df.empty or forecast_df.empty or value_col not in history_df:
        return "Forecast interpretation not available."

    hist = history_df[value_col].dropna()
    fc = forecast_df[value_col].dropna()

    if hist.empty or fc.empty:
        return "Forecast interpretation not available."

    hist_start, hist_end = hist.iloc[0], hist.iloc[-1]
    fc_start, fc_end = fc.iloc[0], fc.iloc[-1]

    if fc_end > fc_start:
        trend = "upward"
    elif fc_end < fc_start:
        trend = "downward"
    else:
        trend = "stable"

    return f"The forecast suggests a {trend} trend for '{value_col}'."


def build_key_insights(
    df: pd.DataFrame,
    target: str | None = None,
    best_model_name: str | None = None
) -> list[str]:
    insights = generate_key_insights(df, target)

    if best_model_name:
        insights.append(f"Current best model: '{best_model_name}'.")

    return insights[:4]


def business_summary(df: pd.DataFrame, target: str | None = None) -> list[str]:
    numeric_cols, categorical_cols, datetime_cols = split_columns(df)

    bullets = [
        f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}",
        f"{len(numeric_cols)} numeric, {len(categorical_cols)} categorical columns",
    ]

    if target and target in df.columns:
        task = infer_task_type(df[target])
        target_kind = classify_target_kind(df[target])
        bullets.insert(1, f"Target: '{target}' ({target_kind})")

        if task == "regression":
            val = safe_numeric(df[target]).dropna()
            if not val.empty:
                bullets.append(f"Average '{target}': {val.mean():.2f}")
        else:
            vc = df[target].dropna().value_counts(normalize=True) * 100
            if not vc.empty:
                bullets.append(f"Top class: '{vc.index[0]}' ({vc.iloc[0]:.1f}%)")
    else:
        missing_pct = df.isna().mean().mean() * 100
        bullets.append(f"Average missingness: {missing_pct:.1f}%")

        if numeric_cols:
            bullets.append(f"Main numeric field count: {len(numeric_cols)}")
        if categorical_cols:
            bullets.append(f"Main category field count: {len(categorical_cols)}")

    if datetime_cols:
        bullets.append(f"Date column detected: '{datetime_cols[0]}'")
    else:
        bullets.append("No date column detected")

    return bullets


# =========================================================
# EXECUTIVE INSIGHTS + STORY MODE
# =========================================================

def is_identifier_like_column(df: pd.DataFrame, col: str) -> bool:
    col_lower = str(col).lower().strip()
    s = df[col].dropna()

    id_keywords = [
        "id", "code", "stockcode", "invoice", "invoice_no", "orderno",
        "orderid", "customerid", "sku", "serial", "uuid", "transaction"
    ]

    if any(k in col_lower for k in id_keywords):
        return True

    if s.empty:
        return False

    unique_ratio = s.nunique(dropna=True) / max(len(s), 1)

    # numeric columns should not be treated as IDs only because they are highly unique
    if pd.api.types.is_numeric_dtype(s):
        integer_like = np.all(np.isclose(s, np.round(s))) if len(s) > 0 else False

        if integer_like and unique_ratio > 0.95:
            return True

        return False

    if unique_ratio > 0.98:
        return True

    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        avg_len = s.astype(str).str.len().mean()
        if unique_ratio > 0.85 and avg_len > 5:
            return True

    return False


def get_meaningful_predictors(df: pd.DataFrame, target: str):
    numeric_cols, categorical_cols, datetime_cols = split_columns(df)

    numeric_predictors = [
        c for c in numeric_cols
        if c != target and not is_identifier_like_column(df, c)
    ]

    categorical_predictors = [
        c for c in categorical_cols
        if c != target
        and not is_identifier_like_column(df, c)
        and 2 <= df[c].nunique(dropna=True) <= max(20, int(0.15 * len(df)))
    ]

    return numeric_predictors, categorical_predictors, datetime_cols


def detect_transaction_like_dataset(df: pd.DataFrame) -> bool:
    cols = [str(c).lower().strip() for c in df.columns]
    transaction_keywords = [
        "invoice", "order", "stockcode", "quantity", "unitprice",
        "price", "customerid", "description"
    ]
    matches = sum(any(k in c for k in transaction_keywords) for c in cols)
    return matches >= 3


def numeric_target_quality_note(df: pd.DataFrame, target: str) -> str:
    y = safe_numeric(df[target]).dropna()

    if y.empty:
        return f"'{target}' does not have enough valid numeric values for interpretation."

    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_pct = ((y < lower) | (y > upper)).mean() * 100

    if outlier_pct > 5:
        return (
            f"The average {target} is around {y.mean():.2f}, but extreme low and high values are present, "
            f"so averages should be interpreted carefully."
        )

    return (
        f"The average {target} is around {y.mean():.2f}, and the values look reasonably stable for summary analysis."
    )


def classification_target_quality_note(df: pd.DataFrame, target: str) -> str:
    vc = df[target].dropna().value_counts(normalize=True) * 100

    if vc.empty:
        return f"'{target}' does not have enough valid values for interpretation."

    if vc.iloc[0] > 70:
        return (
            f"The largest group in '{target}' is '{vc.index[0]}' at {vc.iloc[0]:.1f}%, "
            f"so results should also pay attention to smaller groups."
        )

    return (
        f"The groups in '{target}' are reasonably distributed, which supports comparison across segments."
    )


def best_numeric_driver(df: pd.DataFrame, target: str, numeric_predictors: list[str]) -> str | None:
    if not numeric_predictors:
        return None

    try:
        temp = df[[target] + numeric_predictors].copy()
        for c in [target] + numeric_predictors:
            temp[c] = safe_numeric(temp[c])

        corr = (
            temp.corr(numeric_only=True)[target]
            .drop(target, errors="ignore")
            .abs()
            .sort_values(ascending=False)
        )

        if not corr.empty and pd.notna(corr.iloc[0]):
            return corr.index[0]
    except Exception:
        return None

    return None


def best_business_category(df: pd.DataFrame, target: str, categorical_predictors: list[str]) -> str | None:
    if not categorical_predictors:
        return None

    task = infer_task_type(df[target])
    best_col = None
    best_signal = -np.inf

    for col in categorical_predictors:
        try:
            if task == "regression":
                temp = df[[col, target]].copy()
                temp[target] = safe_numeric(temp[target])
                temp = temp.dropna()

                if temp.empty or temp[col].nunique() < 2:
                    continue

                grp = temp.groupby(col)[target].mean()
                signal = grp.max() - grp.min() if len(grp) >= 2 else -np.inf
            else:
                temp = df[[col, target]].dropna()

                if temp.empty or temp[col].nunique() < 2:
                    continue

                grp = temp.groupby(col)[target].nunique()
                signal = grp.max()

            if signal > best_signal:
                best_signal = signal
                best_col = col
        except Exception:
            continue

    return best_col
    

def generate_key_insights(df: pd.DataFrame, target: str | None = None) -> list[str]:
    insights = []
    numeric_cols, categorical_cols, datetime_cols = split_columns(df)
    transaction_like = detect_transaction_like_dataset(df)

    def top_numeric_summary(col: str) -> str:
        s = safe_numeric(df[col]).dropna()
        if s.empty:
            return f"'{col}' is available, but it does not have enough clean numeric values yet."
        return (
            f"'{col}' stands out as a high-signal numeric field with average {s.mean():.2f} "
            f"and spread {s.std():.2f}."
        )

    def top_category_summary(col: str) -> str:
        vc = df[col].dropna().value_counts()
        if vc.empty:
            return f"'{col}' is available, but it does not have enough clean values yet."
        return (
            f"'{col}' is a strong grouping field, and '{vc.index[0]}' is currently the largest segment "
            f"at {(vc.iloc[0] / vc.sum()) * 100:.1f}%."
        )

    # =====================================================
    # TARGET-BASED PATH
    # =====================================================
    if target and target in df.columns:
        task = infer_task_type(df[target])
        numeric_predictors, categorical_predictors, _ = get_meaningful_predictors(df, target)

        # Insight 1: target story
        if task == "regression":
            y = safe_numeric(df[target]).dropna()
            if not y.empty:
                q1 = y.quantile(0.25)
                q3 = y.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_pct = ((y < lower) | (y > upper)).mean() * 100

                if outlier_pct > 5:
                    insights.append(
                        f"'{target}' averages {y.mean():.2f}, but it contains noticeable extreme values, "
                        f"so trend and model interpretation should be treated carefully."
                    )
                else:
                    insights.append(
                        f"'{target}' averages {y.mean():.2f} and looks fairly stable overall, "
                        f"which makes it suitable for comparison and modeling."
                    )
            else:
                insights.append(f"'{target}' does not yet have enough usable numeric values for a solid insight.")

            driver = best_numeric_driver(df, target, numeric_predictors)
            if driver:
                aligned = df[[target, driver]].copy()
                aligned[target] = safe_numeric(aligned[target])
                aligned[driver] = safe_numeric(aligned[driver])
                aligned = aligned.dropna()

                if not aligned.empty:
                    corr_val = aligned[target].corr(aligned[driver])
                    if pd.notna(corr_val):
                        direction = "positive" if corr_val > 0 else "negative"
                        insights.append(
                            f"'{driver}' is the strongest numeric driver of '{target}' with a {direction} relationship "
                            f"(correlation {corr_val:.2f})."
                        )

        else:
            vc = df[target].dropna().value_counts(normalize=True) * 100
            if not vc.empty:
                top_class = vc.index[0]
                top_pct = vc.iloc[0]

                if len(vc) > 1:
                    second_class = vc.index[1]
                    second_pct = vc.iloc[1]
                    insights.append(
                        f"'{target}' is led by '{top_class}' at {top_pct:.1f}%, followed by "
                        f"'{second_class}' at {second_pct:.1f}%."
                    )
                else:
                    insights.append(
                        f"'{target}' currently has one visible class, '{top_class}', covering {top_pct:.1f}% of rows."
                    )
            else:
                insights.append(f"'{target}' does not yet have enough usable values for a solid class insight.")

            if categorical_predictors:
                category = best_business_category(df, target, categorical_predictors)
                if category:
                    grp = (
                        df[[category, target]]
                        .dropna()
                        .groupby(category)[target]
                        .agg(lambda x: x.astype(str).nunique())
                    )
                    if not grp.empty:
                        insights.append(
                            f"'{category}' is a useful business grouping field for separating patterns in '{target}'."
                        )

        # Insight 3: transaction or structure note
        if transaction_like:
            id_like_cols = [c for c in df.columns if is_identifier_like_column(df, c)]
            if id_like_cols:
                display_cols = ", ".join(id_like_cols[:3])
                insights.append(
                    f"Identifier-like fields such as {display_cols} are present and should be excluded from business interpretation."
                )
            else:
                insights.append(
                    "This looks like transaction-level data, so grouped summaries will be more meaningful than row-level records."
                )
        else:
            usable_numeric = [c for c in numeric_cols if c != target and not is_identifier_like_column(df, c)]
            usable_cats = [
                c for c in categorical_cols
                if c != target and not is_identifier_like_column(df, c) and 2 <= df[c].nunique(dropna=True) <= max(20, int(0.15 * len(df)))
            ]

            if usable_numeric:
                insights.append(top_numeric_summary(usable_numeric[0]))
            elif usable_cats:
                insights.append(top_category_summary(usable_cats[0]))

        # Insight 4: time or data quality
        if datetime_cols:
            insights.append(
                f"'{datetime_cols[0]}' enables time-based tracking, so trend analysis and forecasting are possible."
            )
        else:
            missing_pct = df.isna().mean().mean() * 100
            duplicate_rows = int(df.duplicated().sum())
            insights.append(
                f"Data quality looks usable overall with {missing_pct:.1f}% average missingness and {duplicate_rows} duplicate rows."
            )

        return list(dict.fromkeys(insights))[:4]

    # =====================================================
    # DATASET-LEVEL PATH FOR BUSINESS MODE
    # =====================================================
    if transaction_like:
        insights.append(
            "This dataset looks transaction-level, so grouped business summaries will be more useful than row-level records."
        )

    id_like_cols = [c for c in df.columns if is_identifier_like_column(df, c)]
    if id_like_cols:
        display_cols = ", ".join(id_like_cols[:3])
        insights.append(
            f"Identifier-like columns such as {display_cols} are present and should not be treated as business drivers."
        )

    usable_numeric = [c for c in numeric_cols if not is_identifier_like_column(df, c)]
    usable_cats = [
        c for c in categorical_cols
        if not is_identifier_like_column(df, c) and 2 <= df[c].nunique(dropna=True) <= max(20, int(0.15 * len(df)))
    ]

    if usable_numeric:
        insights.append(top_numeric_summary(usable_numeric[0]))

    if usable_cats:
        insights.append(top_category_summary(usable_cats[0]))

    if datetime_cols:
        insights.append(
            f"'{datetime_cols[0]}' gives this dataset a clear time dimension, which is useful for trend and forecast views."
        )
    else:
        missing_pct = df.isna().mean().mean() * 100
        duplicate_rows = int(df.duplicated().sum())
        insights.append(
            f"Overall data quality is solid with {missing_pct:.1f}% average missingness and {duplicate_rows} duplicate rows."
        )

    if not insights:
        insights.append("The dataset loaded correctly, but no strong automated insight could be produced yet.")

    return list(dict.fromkeys(insights))[:4]


def generate_business_recommendations(df: pd.DataFrame, target: str | None = None) -> list[str]:
    recs = []
    numeric_cols, categorical_cols, datetime_cols = split_columns(df)
    transaction_like = detect_transaction_like_dataset(df)

    def add_unique(text: str):
        if text and text not in recs:
            recs.append(text)

    def best_numeric_summary_col(exclude_cols=None):
        exclude_cols = exclude_cols or []
        candidates = []
        for col in numeric_cols:
            if col in exclude_cols or is_identifier_like_column(df, col):
                continue
            s = safe_numeric(df[col]).dropna()
            if len(s) > 0:
                candidates.append((col, len(s), float(s.std()) if len(s) > 1 else 0.0))
        if not candidates:
            return None
        candidates = sorted(candidates, key=lambda x: (x[1], x[2]), reverse=True)
        return candidates[0][0]

    def best_segment_col(exclude_cols=None):
        exclude_cols = exclude_cols or []
        usable = [
            c for c in categorical_cols
            if c not in exclude_cols
            and not is_identifier_like_column(df, c)
            and 2 <= df[c].nunique(dropna=True) <= max(20, int(0.15 * len(df)))
        ]
        return usable[0] if usable else None

    # =====================================================
    # TARGET-BASED PATH
    # =====================================================
    if target and target in df.columns:
        task = infer_task_type(df[target])
        numeric_predictors, categorical_predictors, _ = get_meaningful_predictors(df, target)

        if task == "regression":
            y = safe_numeric(df[target]).dropna()
            if not y.empty:
                q1 = y.quantile(0.25)
                q3 = y.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_pct = ((y < lower) | (y > upper)).mean() * 100

                if outlier_pct > 5:
                    add_unique(
                        f"Review extreme values in '{target}'. Why: Around {outlier_pct:.1f}% of values look unusual and may distort averages or model fit. Expected impact: More stable reporting and better predictions."
                    )

            driver = best_numeric_driver(df, target, numeric_predictors)
            if driver:
                aligned = df[[target, driver]].copy()
                aligned[target] = safe_numeric(aligned[target])
                aligned[driver] = safe_numeric(aligned[driver])
                aligned = aligned.dropna()

                if not aligned.empty:
                    corr_val = aligned[target].corr(aligned[driver])
                    if pd.notna(corr_val):
                        direction = "increase" if corr_val > 0 else "decrease"
                        add_unique(
                            f"Monitor '{driver}' closely. Why: It is the strongest numeric driver found for '{target}', and higher values tend to {direction} the target. Expected impact: Better control over target movement."
                        )

            category = best_business_category(df, target, categorical_predictors)
            if category:
                grp = df[[category, target]].copy()
                grp[target] = safe_numeric(grp[target])
                grp = grp.dropna()

                if not grp.empty:
                    grouped = grp.groupby(category)[target].mean().sort_values(ascending=False)
                    if len(grouped) >= 2:
                        add_unique(
                            f"Compare '{target}' across '{category}'. Why: '{grouped.index[0]}' currently leads while '{grouped.index[-1]}' trails on average. Expected impact: Clearer segment-level action."
                        )

        else:
            vc = df[target].dropna().value_counts(normalize=True) * 100
            if not vc.empty:
                top_class = vc.index[0]
                top_pct = vc.iloc[0]

                if top_pct > 70:
                    add_unique(
                        f"Pay extra attention to minority groups in '{target}'. Why: '{top_class}' dominates at {top_pct:.1f}%, which can hide weaker segments. Expected impact: More balanced analysis and better class handling."
                    )
                else:
                    add_unique(
                        f"Use '{target}' for segment comparison. Why: The class mix is usable and not overly dominated by one group. Expected impact: Better business interpretation across categories."
                    )

            category = best_business_category(df, target, categorical_predictors)
            if category:
                temp = df[[category, target]].dropna()
                if not temp.empty:
                    cross = pd.crosstab(temp[category], temp[target])
                    if not cross.empty:
                        dominant_seg = cross.sum(axis=1).sort_values(ascending=False).index[0]
                        add_unique(
                            f"Break down '{target}' by '{category}'. Why: This grouping shows clear class separation and '{dominant_seg}' is one of the most visible segments. Expected impact: Stronger targeting decisions."
                        )

            strong_num = best_numeric_summary_col(exclude_cols=[target])
            if strong_num:
                add_unique(
                    f"Profile '{strong_num}' across target groups. Why: It looks like one of the strongest measurable fields in the dataset. Expected impact: Better understanding of what separates the classes."
                )

        if transaction_like:
            add_unique(
                "Aggregate the data before interpreting performance. Why: Transaction-level rows can hide the real pattern. Expected impact: Clearer business actions."
            )

        if datetime_cols:
            add_unique(
                f"Track '{target}' over time using '{datetime_cols[0]}'. Why: A usable time field is available. Expected impact: Better timing and forecasting decisions."
            )

        missing_pct = df.isna().mean().mean() * 100
        if missing_pct > 10:
            add_unique(
                f"Improve data quality before major decisions. Why: Missing data is around {missing_pct:.1f}%. Expected impact: More reliable analysis."
            )

        if not recs:
            fallback_num = best_numeric_summary_col(exclude_cols=[target])
            fallback_cat = best_segment_col(exclude_cols=[target])

            if fallback_num:
                add_unique(
                    f"Start with '{fallback_num}' as a key driver review. Why: No stronger automatic recommendation was triggered, and this field looks consistently informative. Expected impact: Better direction for deeper analysis."
                )
            elif fallback_cat:
                add_unique(
                    f"Start with grouped analysis by '{fallback_cat}'. Why: No stronger automatic recommendation was triggered, and this field is a practical segmenting variable. Expected impact: Faster business interpretation."
                )
            else:
                add_unique(
                    f"Review the strongest predictors of '{target}' and compare their patterns across the dataset. Why: No strong automatic recommendation was triggered. Expected impact: Better direction for deeper analysis."
                )

        return recs[:4]

    # =====================================================
    # NO-TARGET BUSINESS MODE
    # =====================================================
    if transaction_like:
        add_unique(
            "Summarize the data by meaningful business groups before interpreting it. Why: Transaction-level rows rarely tell the full story. Expected impact: Clearer insight."
        )

    segment_col = best_segment_col()
    if segment_col:
        top_levels = df[segment_col].dropna().value_counts()
        if not top_levels.empty:
            add_unique(
                f"Start with grouped analysis using '{segment_col}'. Why: '{top_levels.index[0]}' is the largest visible segment and this field is well suited for comparison. Expected impact: Better decision support."
            )

    numeric_focus = best_numeric_summary_col()
    if numeric_focus:
        s = safe_numeric(df[numeric_focus]).dropna()
        if not s.empty:
            add_unique(
                f"Review the spread and trend of '{numeric_focus}'. Why: It looks like one of the strongest measurable fields in the dataset. Expected impact: Faster understanding of performance patterns."
            )

    if datetime_cols:
        add_unique(
            f"Use '{datetime_cols[0]}' for trend and forecasting views. Why: A usable time field exists. Expected impact: Better visibility into change over time."
        )

    id_like_cols = [c for c in df.columns if is_identifier_like_column(df, c)]
    if id_like_cols:
        display_cols = ", ".join(id_like_cols[:2])
        add_unique(
            f"Keep identifier-like fields such as {display_cols} out of business interpretation. Why: They describe records rather than drivers. Expected impact: Cleaner summaries and better charts."
        )

    missing_pct = df.isna().mean().mean() * 100
    if missing_pct > 10:
        add_unique(
            f"Improve missing-data handling first. Why: Missingness is around {missing_pct:.1f}%. Expected impact: More reliable summaries."
        )

    if not recs:
        add_unique(
            "Begin with data cleaning, grouped summaries, and a review of the most complete numeric and category columns. Why: That gives the strongest first-pass understanding. Expected impact: Better next-step analysis."
        )

    return recs[:4]


def executive_story(df: pd.DataFrame, target: str | None = None) -> str:
    insights = generate_key_insights(df, target)
    recommendations = generate_business_recommendations(df, target)

    numeric_cols, categorical_cols, datetime_cols = split_columns(df)
    transaction_like = detect_transaction_like_dataset(df)

    if target and target in df.columns:
        task = infer_task_type(df[target])

        opening = (
            f"We analyzed the dataset to understand the behavior of '{target}' and identify the main business patterns affecting it."
        )

        key_findings = " ".join([x.rstrip(".") + "." for x in insights[:3]]) if insights else f"'{target}' shows patterns that support further analysis."

        if transaction_like:
            pattern_text = (
                "The dataset behaves like transaction-level data, so row-level product or transaction fields should not be treated as business drivers. "
                "Instead, the strongest meaning usually comes from grouped patterns and aggregated summaries."
            )
        elif task == "classification":
            pattern_text = (
                f"The results suggest that the groups within '{target}' do not behave in exactly the same way, which creates an opportunity for better targeting and decision-making."
            )
        else:
            pattern_text = (
                f"The results suggest that '{target}' is influenced by a mix of measurable factors and business segments, which creates room for performance improvement."
            )

        if datetime_cols:
            timing_text = (
                f"A usable time field is also available, so these patterns can be tracked over time to understand when performance improves or declines."
            )
        else:
            timing_text = (
                "Overall, the dataset is most useful when interpreted through meaningful business groups and clean variables rather than highly unique record-level fields."
            )

        if recommendations:
            actions = "Recommended next steps: " + "; ".join(
                [r.split("Why:")[0].strip().rstrip(".") for r in recommendations[:3]]
            ) + "."
        else:
            actions = "Recommended next steps: focus on data quality, meaningful groupings, and the main drivers of the target."

        return "\n\n".join([opening, key_findings, pattern_text, timing_text, actions])

    # =====================================================
    # DATASET-LEVEL STORY FOR BUSINESS MODE
    # =====================================================
    opening = (
        "We reviewed the dataset to identify its structure, the most useful business fields, and the areas that are most likely to produce meaningful analysis."
    )

    key_findings = " ".join([x.rstrip(".") + "." for x in insights[:3]]) if insights else (
        "The dataset contains usable information, but the strongest patterns will depend on grouping, cleaning, and trend analysis."
    )

    if transaction_like:
        pattern_text = (
            "The dataset appears to be transaction-level, which means the clearest business meaning will usually come from grouped summaries rather than row-level records."
        )
    else:
        pattern_text = (
            "The dataset appears suitable for general business analysis, especially through numeric summaries, segmentation, and comparisons across key categories."
        )

    if datetime_cols:
        timing_text = (
            f"A usable time field is available through '{datetime_cols[0]}', so trend analysis and forecasting can be added naturally."
        )
    else:
        timing_text = (
            "The dataset is currently strongest for summary and segmentation analysis rather than timeline analysis."
        )

    if recommendations:
        actions = "Recommended next steps: " + "; ".join(
            [r.split("Why:")[0].strip().rstrip(".") for r in recommendations[:3]]
        ) + "."
    else:
        actions = "Recommended next steps: clean the data, review meaningful categories, and identify the most useful numeric measures."

    return "\n\n".join([opening, key_findings, pattern_text, timing_text, actions])

    
# =========================================================
# CHART HELPERS
# =========================================================
def plot_matplotlib(fig):
    st.pyplot(fig)
    plt.close(fig)


def plot_plotly(fig):
    st.plotly_chart(fig, width="stretch")


def make_business_chart(df, chart_type, x_col=None, y_col=None, color_col=None, title=None):
    title = title or "Business Chart"

    if df is None or df.empty:
        return px.bar(title=f"{title} (no data)")

    if x_col is not None and x_col not in df.columns:
        return px.bar(title=f"{title} (invalid x column)")

    if y_col is not None and y_col not in df.columns:
        y_col = None

    if color_col is not None and color_col not in df.columns:
        color_col = None

    if chart_type == "Column Chart":
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title)

    elif chart_type == "Bar Chart":
        fig = px.bar(df, x=y_col, y=x_col, color=color_col, orientation="h", title=title)

    elif chart_type == "Line Chart":
        fig = px.line(df, x=x_col, y=y_col, color=color_col, markers=True, title=title)

    elif chart_type == "Area Chart":
        fig = px.area(df, x=x_col, y=y_col, color=color_col, title=title)

    elif chart_type == "Pie Chart":
        if x_col is None or x_col not in df.columns:
            return px.bar(title=f"{title} (invalid category column)")
        if y_col is not None and y_col in df.columns:
            fig = px.pie(df, names=x_col, values=y_col, title=title)
        else:
            counts = df[x_col].dropna().value_counts().reset_index()
            counts.columns = [x_col, "count"]
            fig = px.pie(counts, names=x_col, values="count", title=title)

    elif chart_type == "Donut Chart":
        if x_col is None or x_col not in df.columns:
            return px.bar(title=f"{title} (invalid category column)")
        if y_col is not None and y_col in df.columns:
            fig = px.pie(df, names=x_col, values=y_col, title=title, hole=0.45)
        else:
            counts = df[x_col].dropna().value_counts().reset_index()
            counts.columns = [x_col, "count"]
            fig = px.pie(counts, names=x_col, values="count", title=title, hole=0.45)

    elif chart_type == "Stacked Bar Chart":
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, barmode="stack", title=title)

    elif chart_type == "Combo Chart":
        fig = go.Figure()
        if x_col is not None and y_col is not None and x_col in df.columns and y_col in df.columns:
            fig.add_trace(go.Bar(x=df[x_col], y=df[y_col], name=str(y_col)))

            if color_col and color_col in df.columns and pd.api.types.is_numeric_dtype(df[color_col]):
                line_source = color_col
            else:
                line_source = y_col

            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[line_source],
                    mode="lines+markers",
                    name=f"{line_source} trend",
                    yaxis="y2",
                )
            )
            fig.update_layout(
                title=title,
                yaxis=dict(title=str(y_col)),
                yaxis2=dict(title=f"{line_source} trend", overlaying="y", side="right", showgrid=False),
            )
        else:
            fig = px.bar(title=f"{title} (insufficient columns)")

    else:
        fig = px.bar(df, x=x_col, y=y_col, title=title)

    fig.update_layout(
        xaxis_title=str(x_col) if x_col else "",
        yaxis_title=str(y_col) if y_col else "",
        legend_title_text=str(color_col) if color_col else "",
    )
    return fig


def render_count_or_share_charts(df, col, target=None, target_task=None):
    if col not in df.columns:
        st.warning(f"Column '{col}' not found.")
        return

    source = df[[col]].copy() if target is None or target not in df.columns else df[[col, target]].copy()
    source = source.dropna(subset=[col])

    if source.empty:
        st.warning("Not enough data to draw this chart.")
        return

    # Limit very high-cardinality categories
    top_levels = source[col].value_counts().head(15).index
    source = source[source[col].isin(top_levels)]

    if target_task == "classification" and target is not None and target in source.columns:
        cross = source.groupby([col, target]).size().reset_index(name="count")

        if cross.empty:
            st.warning("Not enough grouped data to draw this chart.")
            return

        plot_plotly(
            px.bar(
                cross,
                x=col,
                y="count",
                color=target,
                title=f"{target} by {col}",
                barmode="group",
            )
        )
        plot_plotly(
            px.bar(
                cross,
                x=col,
                y="count",
                color=target,
                title=f"Stacked {target} by {col}",
                barmode="stack",
            )
        )
    else:
        pie_df = source[col].value_counts().reset_index()
        pie_df.columns = [col, "count"]

        if pie_df.empty:
            st.warning("Not enough data to draw this chart.")
            return

        plot_plotly(make_business_chart(pie_df, "Pie Chart", x_col=col, y_col="count", title=f"{col} share"))
        plot_plotly(make_business_chart(pie_df, "Donut Chart", x_col=col, y_col="count", title=f"{col} share donut"))

        st.markdown("**Interpretation**")
        st.info(categorical_chart_insight(source, col))


def render_numeric_target_charts(df, col, target):
    if col not in df.columns or target not in df.columns:
        st.warning("Selected columns are not available.")
        return

    work = df[[col, target]].copy()
    work[col] = safe_numeric(work[col])

    target_task = infer_task_type(df[target])

    if target_task == "classification":
        work = work.dropna(subset=[col, target])

        if work.empty:
            st.warning("Not enough data to compare numeric values across target groups.")
            return

        grouped = work.groupby(target, dropna=False)[col].mean().reset_index()

        fig = px.bar(
            grouped,
            x=target,
            y=col,
            color=target,
            text=col,
            title=f"{target} vs {col}",
        )
        plot_plotly(fig)

        st.info(
            f"This chart compares the average '{col}' across the groups in '{target}'. "
            f"It helps show which group tends to have higher or lower values."
        )

    else:
        work[target] = safe_numeric(work[target])
        work = work.dropna(subset=[col, target]).sort_values(col)

        if work.empty:
            st.warning("Not enough numeric data to compare target against this column.")
            return

        if work.shape[0] > 200:
            work = work.sample(200, random_state=42).sort_values(col)

        fig = px.scatter(
            work,
            x=col,
            y=target,
            trendline="ols",
            title=f"{target} vs {col}",
        )
        plot_plotly(fig)

        st.info(
            f"This scatter chart shows how '{target}' changes as '{col}' changes. "
            f"It helps reveal whether higher or lower values of '{col}' are linked with higher or lower '{target}'."
        )

# =========================================================
# RECOMMENDATION HELPERS
# =========================================================
def recommend_business_columns(df: pd.DataFrame, target: str):
    numeric_cols, categorical_cols, dt_cols = split_columns(df)

    good_y = [c for c in numeric_cols if c != target]
    good_categories = [c for c in categorical_cols if c != target]

    return {
        "good_y": good_y,
        "good_categories": good_categories,
        "date_cols": dt_cols,
    }


def recommend_forecast_columns(df: pd.DataFrame):
    numeric_cols, _, dt_cols = split_columns(df)

    value_candidates = []
    for col in numeric_cols:
        s = safe_numeric(df[col]).dropna()
        if s.empty:
            continue

        score = 1
        lower = col.lower()

        for kw in ["sales", "revenue", "income", "amount", "price", "cost", "profit", "expense", "value"]:
            if kw in lower:
                score += 10

        if s.nunique() > 10:
            score += 2

        if len(s) >= max(10, int(0.5 * len(df))):
            score += 2

        value_candidates.append((col, score))

    value_candidates = sorted(value_candidates, key=lambda x: x[1], reverse=True)

    return {
        "date_cols": dt_cols,
        "value_cols": [x[0] for x in value_candidates],
    }


def build_pivot_tables(df: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    numeric_cols, categorical_cols, _ = split_columns(df)
    pivots = {}

    usable_cats = [c for c in categorical_cols if df[c].nunique(dropna=True) <= 20]

    for col in usable_cats[:4]:
        counts = df[col].dropna().value_counts().rename_axis(col).reset_index(name="count")
        if not counts.empty:
            pivots[f"{col}_counts"] = counts

    if target in numeric_cols:
        for col in usable_cats[:3]:
            grouped = (
                df[[col, target]]
                .dropna(subset=[col])
                .groupby(col, dropna=False)[target]
                .agg(["count", "mean", "sum", "median"])
                .reset_index()
                .sort_values("sum", ascending=False)
            )
            if not grouped.empty:
                pivots[f"{col}_by_{target}"] = grouped

    return pivots


def detect_pivot_column_types(df: pd.DataFrame, cat_unique_threshold: int = 20):
    categorical_cols = []
    numeric_cols = []
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    for col in df.columns:
        if col in datetime_cols:
            categorical_cols.append(col)
            continue

        s = df[col]

        if pd.api.types.is_bool_dtype(s):
            categorical_cols.append(col)

        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_string_dtype(s):
            categorical_cols.append(col)

        elif pd.api.types.is_numeric_dtype(s):
            non_null = s.dropna()
            n_unique = non_null.nunique()

            if len(non_null) == 0:
                numeric_cols.append(col)
            elif n_unique <= cat_unique_threshold and np.all(np.isclose(non_null, np.round(non_null))):
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)

        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, datetime_cols

def create_custom_pivot(df, index_cols, column_cols, value_col, aggfunc):
    if not index_cols and not column_cols:
        return None

    temp = df.copy()
    calc_fields = build_calculated_fields(temp)
    numeric_cols, categorical_cols, _ = detect_pivot_column_types(temp)

    if value_col == "__row_count__":
        pivot = pd.pivot_table(
            temp,
            index=index_cols if index_cols else None,
            columns=column_cols if column_cols else None,
            aggfunc="size",
            fill_value=0
        )
        return pivot.reset_index()

    if value_col in calc_fields:
        temp[value_col] = calc_fields[value_col]
        actual = value_col
        value_is_numeric = True
        value_is_categorical = False
    else:
        actual = value_col
        value_is_numeric = actual in numeric_cols
        value_is_categorical = actual in categorical_cols

    if actual not in temp.columns:
        return None

    if value_is_categorical:
        if aggfunc not in ["count", "nunique"]:
            aggfunc = "count"

        if aggfunc == "nunique":
            pivot = pd.pivot_table(
                temp,
                index=index_cols if index_cols else None,
                columns=column_cols if column_cols else None,
                values=actual,
                aggfunc=pd.Series.nunique,
                fill_value=0
            )
        else:
            pivot = pd.pivot_table(
                temp,
                index=index_cols if index_cols else None,
                columns=column_cols if column_cols else None,
                values=actual,
                aggfunc="count",
                fill_value=0
            )

        return pivot.reset_index()

    if value_is_numeric:
        agg_map = {
            "sum": "sum",
            "mean": "mean",
            "count": "count",
            "median": "median",
            "min": "min",
            "max": "max",
            "std": "std",
            "nunique": pd.Series.nunique,
        }

        if aggfunc not in agg_map:
            aggfunc = "sum"

        pivot = pd.pivot_table(
            temp,
            index=index_cols if index_cols else None,
            columns=column_cols if column_cols else None,
            values=actual,
            aggfunc=agg_map[aggfunc],
            fill_value=0
        )
        return pivot.reset_index()

    return None

# =========================================================
# CALCULATED FIELDS
# =========================================================
def build_calculated_fields(df: pd.DataFrame) -> dict[str, pd.Series]:
    calc_fields = {}

    def find_col(keywords):
        for col in df.columns:
            if any(k in col.lower() for k in keywords):
                return col
        return None

    sales = find_col(["sales", "revenue", "amount"])
    profit = find_col(["profit"])
    quantity = find_col(["quantity", "qty"])
    discount = find_col(["discount"])
    cost = find_col(["cost"])

    if sales:
        calc_fields["Revenue"] = safe_numeric(df[sales])

    if cost:
        calc_fields["Cost"] = safe_numeric(df[cost])

    if sales and profit:
        s = safe_numeric(df[sales])
        p = safe_numeric(df[profit])
        calc_fields["Profit"] = p
        calc_fields["Estimated Cost"] = s - p
        calc_fields["Profit Ratio"] = np.where(np.abs(s) > 1e-9, p / s, np.nan)

    if sales and discount:
        s = safe_numeric(df[sales])
        d = safe_numeric(df[discount])
        calc_fields["Discount Amount"] = s * d
        calc_fields["Net Sales"] = s * (1 - d)

    if sales and quantity:
        s = safe_numeric(df[sales])
        q = safe_numeric(df[quantity])
        calc_fields["Unit Price"] = np.where(np.abs(q) > 1e-9, s / q, np.nan)

    return calc_fields


# =========================================================
# MODELING
# =========================================================
def prepare_modeling_dataset(df: pd.DataFrame, target: str):
    work = df.copy()

    if target not in work.columns:
        raise ValueError("Target column not found.")

    work = work.dropna(subset=[target]).copy()

    if work.empty:
        raise ValueError("No usable rows after removing missing target.")

    # remove exact duplicate rows first
    work = work.drop_duplicates().copy()

    target_series = work[target]
    X = work.drop(columns=[target]).copy()

    # drop identifier-like columns
    id_like_cols = [c for c in X.columns if is_identifier_like_column(work, c)]
    if id_like_cols:
        X = X.drop(columns=id_like_cols, errors="ignore")

    # drop columns that directly leak the target by name
    target_lower = str(target).lower().strip()
    leakage_name_cols = []
    for col in X.columns:
        col_lower = str(col).lower().strip()
        if (
            target_lower in col_lower
            or any(flag in col_lower for flag in ["target", "label", "outcome", "predicted", "prediction"])
        ):
            leakage_name_cols.append(col)

    if leakage_name_cols:
        X = X.drop(columns=leakage_name_cols, errors="ignore")

    # drop constant columns
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols, errors="ignore")

    # drop columns that are almost identical to target
    target_kind = classify_target_kind(target_series)
    near_target_cols = []

    for col in X.columns:
        try:
            aligned = pd.DataFrame({"x": X[col], "y": target_series}).dropna()
            if aligned.empty:
                continue

            same_ratio = (aligned["x"].astype(str) == aligned["y"].astype(str)).mean()
            if same_ratio >= 0.98:
                near_target_cols.append(col)
                continue

            if target_kind != "numerical":
                if aligned["x"].nunique() <= max(20, int(0.2 * len(aligned))):
                    mapping_check = aligned.groupby("x")["y"].nunique().max()
                    if mapping_check == 1 and aligned["x"].nunique() == aligned["y"].nunique():
                        near_target_cols.append(col)
                        continue

            if pd.api.types.is_numeric_dtype(X[col]) and pd.api.types.is_numeric_dtype(target_series):
                aligned_num = pd.DataFrame(
                    {
                        "x": safe_numeric(X[col]),
                        "y": safe_numeric(target_series),
                    }
                ).dropna()

                if len(aligned_num) >= 3:
                    corr = aligned_num.corr(numeric_only=True).iloc[0, 1]
                    if pd.notna(corr) and abs(corr) >= 0.995:
                        near_target_cols.append(col)
        except Exception:
            continue

    if near_target_cols:
        X = X.drop(columns=list(dict.fromkeys(near_target_cols)), errors="ignore")

    if X.empty or X.shape[1] == 0:
        raise ValueError("No usable predictors remain after removing leakage and identifier-like columns.")

    cleaned = pd.concat([X, target_series], axis=1)

    removed_info = {
        "identifier_like_cols": id_like_cols,
        "leakage_name_cols": leakage_name_cols,
        "near_target_cols": list(dict.fromkeys(near_target_cols)),
        "constant_cols": constant_cols,
    }

    return cleaned, removed_info

def build_preprocessor(X: pd.DataFrame):
    X = X.copy()

    datetime_features = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]

    for col in datetime_features:
        X[f"{col}_year"] = X[col].dt.year
        X[f"{col}_month"] = X[col].dt.month
        X[f"{col}_day"] = X[col].dt.day
        X[f"{col}_dayofweek"] = X[col].dt.dayofweek

    X = X.drop(columns=datetime_features, errors="ignore")

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )

    return preprocessor, numeric_features, categorical_features, X


def get_feature_names(preprocessor, numeric_features, categorical_features):
    names = list(numeric_features)

    if categorical_features:
        try:
            cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            names.extend(cat_encoder.get_feature_names_out(categorical_features).tolist())
        except Exception:
            names.extend(categorical_features)

    return names


def recommended_model_label(task_type: str, model_name: str) -> str:
    mapping = {
        "Linear Regression": "Good simple baseline for numeric prediction.",
        "Ridge": "Useful when many inputs are related to each other.",
        "Lasso": "Useful when you want a smaller set of important drivers.",
        "ElasticNet": "Balances shrinkage and feature selection.",
        "Decision Tree Regressor": "Useful when relationships are not straight-line.",
        "Random Forest Regressor": "Strong all-round numeric prediction model.",
        "XGBoost Regressor": "Strong boosted model for complex numeric prediction.",
        "KNN Regressor": "Useful when similar rows should give similar values.",

        "Logistic Regression": "Good simple baseline for classification.",
        "Decision Tree Classifier": "Easy rule-based classification model.",
        "Random Forest Classifier": "Strong all-round classification model.",
        "XGBoost Classifier": "Strong boosted model for complex classification.",
        "KNN Classifier": "Useful when similar rows should belong to similar groups.",
        "SVM": "Useful when class boundaries are more complex.",
    }

    return mapping.get(model_name, f"Recommended {task_type} model")


def evaluate_model(task_type: str, y_true, y_pred):
    if task_type == "regression":
        return {
            "R2": float(metrics.r2_score(y_true, y_pred)),
            "MAE": float(metrics.mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(metrics.mean_squared_error(y_true, y_pred))),
        }

    return {
        "Accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "Precision": float(metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1": float(metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def model_bank(task_type: str):
    models = {}

    if task_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "Lasso": Lasso(random_state=42),
            "ElasticNet": ElasticNet(random_state=42),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "Random Forest Regressor": RandomForestRegressor(random_state=42, n_estimators=200),
            "KNN Regressor": KNeighborsRegressor(),
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost Regressor"] = XGBRegressor(
                random_state=42, n_estimators=200, max_depth=5, learning_rate=0.05
            )

    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
            "Random Forest Classifier": RandomForestClassifier(random_state=42, n_estimators=200),
            "KNN Classifier": KNeighborsClassifier(),
            "SVM": SVC(probability=True),
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost Classifier"] = XGBClassifier(
                random_state=42, n_estimators=200, max_depth=5, learning_rate=0.05, eval_metric="mlogloss"
            )

    return models


def train_models(df: pd.DataFrame, target: str, selected_models=None, enable_feature_selection=True):
    data = df.dropna(subset=[target]).copy()

    if data.empty:
        raise ValueError("No usable rows after removing missing target.")

    if len(data) < 10:
        raise ValueError("Dataset too small for modeling.")

    X_raw = data.drop(columns=[target])
    y = data[target]

    task_type = infer_task_type(y)

    preprocessor, numeric_features, categorical_features, X = build_preprocessor(X_raw)

    if X.shape[1] == 0:
        raise ValueError("No usable predictors.")

    models = model_bank(task_type)

    if selected_models:
        models = {k: v for k, v in models.items() if k in selected_models}

    if not models:
        raise ValueError("No models selected.")

    # Clean target for split
    y_clean = y.dropna()

    # Stratify safely
    stratify = None
    if task_type == "classification":
        vc = y_clean.value_counts()
        if vc.min() >= 2 and len(vc) > 1:
            stratify = y_clean

    X = X.loc[y_clean.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clean, test_size=0.2, random_state=42, stratify=stratify
    )

    if X_train.empty or X_test.empty:
        raise ValueError("Split failed.")

    preprocessor.fit(X_train)

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Avoid huge dense conversion
    if hasattr(X_train_t, "toarray") and X_train_t.shape[1] < 5000:
        X_train_t = X_train_t.toarray()
        X_test_t = X_test_t.toarray()

    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)

    rows = []
    fitted = {}

    for name, model in models.items():
        try:
            model.fit(X_train_t, y_train)
            pred = model.predict(X_test_t)

            metric_row = evaluate_model(task_type, y_test, pred)

            rows.append({
                "Model": name,
                "Recommended use": recommended_model_label(task_type, name),
                **metric_row
            })

            fitted[name] = {
                "model": model,
                "pred": pred,
                "y_test": y_test,
                "feature_names": feature_names,
                "task_type": task_type,
            }

        except Exception as e:
            rows.append({
                "Model": name,
                "Recommended use": f"Failed: {str(e)}"
            })

    results_df = pd.DataFrame(rows)

    metric_cols = ["R2", "MAE", "RMSE"] if task_type == "regression" else ["Accuracy", "Precision", "Recall", "F1"]
    success_df = results_df.dropna(subset=[c for c in metric_cols if c in results_df.columns], how="all")

    if success_df.empty:
        raise ValueError("All models failed.")

    if task_type == "regression":
        success_df["rank_score"] = (
            success_df["R2"].rank(ascending=False)
            + success_df["MAE"].rank(ascending=True)
            + success_df["RMSE"].rank(ascending=True)
        )
    else:
        success_df["rank_score"] = (
            success_df["F1"].rank(ascending=False)
            + success_df["Accuracy"].rank(ascending=False)
        )

    success_df = success_df.sort_values("rank_score")

    best_name = success_df.iloc[0]["Model"]

    return task_type, results_df, best_name, fitted[best_name]


def model_improving_tips(
    task_type,
    best_model_name,
    results_df=None,
    y_true=None,
    y_pred=None,
):
    if task_type == "classification":
        return [
            f"The current best model is {best_model_name}. Compare it with the next best model to check whether the ranking is stable.",
            "Use the confusion matrix to identify which classes are getting mixed up most often.",
            "If one class matters more, tune the model toward higher recall or higher precision for that class.",
            "If class probabilities are available, adjust the classification threshold instead of relying only on the default cutoff.",
        ]

    tips = []

    r2_val = None
    mae_val = None
    rmse_val = None

    if results_df is not None and not results_df.empty and "Model" in results_df.columns:
        row = results_df[results_df["Model"] == best_model_name]
        if not row.empty:
            row = row.iloc[0]
            r2_val = row["R2"] if "R2" in row and pd.notna(row["R2"]) else None
            mae_val = row["MAE"] if "MAE" in row and pd.notna(row["MAE"]) else None
            rmse_val = row["RMSE"] if "RMSE" in row and pd.notna(row["RMSE"]) else None

    residuals = None
    outlier_ratio = None
    residual_spread_ratio = None
    bias = None

    if y_true is not None and y_pred is not None:
        y_true = pd.Series(y_true).reset_index(drop=True)
        y_pred = pd.Series(y_pred).reset_index(drop=True)

        if len(y_true) > 0 and len(y_true) == len(y_pred):
            residuals = y_true - y_pred
            abs_scale = max(float(np.abs(y_true).median()), 1e-6)
            residual_spread_ratio = float(residuals.std() / abs_scale) if len(residuals) > 1 else 0.0
            bias = float(residuals.mean())

            q1 = residuals.quantile(0.25)
            q3 = residuals.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            if iqr > 0:
                outlier_ratio = float(((residuals < lower) | (residuals > upper)).mean())

    # 1. Data problem
    if r2_val is not None:
        if r2_val < 0.10:
            tips.append(
                f"The model shows very low explanatory power with R² = {r2_val:.3f}, which means it is not capturing meaningful patterns in the data. "
                f"The current features may be weak, noisy, or not business-relevant, so start by cleaning the data and removing irrelevant or identifier-like columns."
            )
        elif r2_val < 0.30:
            tips.append(
                f"The model explains only a limited share of the variation with R² = {r2_val:.3f}. "
                f"There is some signal, but stronger feature engineering is likely needed."
            )
        else:
            tips.append(
                f"The model captures a reasonable amount of signal with R² = {r2_val:.3f}, but there is still room to improve feature quality and model fit."
            )

    # 2. Outliers
    if outlier_ratio is not None:
        if outlier_ratio > 0.05:
            tips.append(
                f"The residual plot shows wide and uneven errors with several extreme outliers. "
                f"About {outlier_ratio * 100:.1f}% of residuals look unusual, so removing, capping, or separately handling extreme values could improve model stability."
            )
        elif residual_spread_ratio is not None and residual_spread_ratio > 0.50:
            tips.append(
                "The residual errors are widely spread even without a large outlier share. "
                "This suggests unstable predictions and weak fit, so target cleaning and better feature construction should come before more model tuning."
            )

    # 3. Feature issue
    if r2_val is not None and r2_val < 0.20:
        tips.append(
            "The model may be relying on weak or noisy features. "
            "Focus on meaningful business variables and aggregated metrics, such as grouped totals, country-level summaries, time-based summaries, or quantity-driven features, instead of raw row-level fields."
        )

    # 4. Model limitation
    if results_df is not None and not results_df.empty and "R2" in results_df.columns:
        valid_r2 = pd.to_numeric(results_df["R2"], errors="coerce").dropna()
        if not valid_r2.empty and valid_r2.max() < 0.20:
            tips.append(
                "Since even the better models are performing poorly, the main problem is likely feature quality or preprocessing rather than model choice alone. "
                "This looks more like an underfitting or data-preparation issue than a simple tuning issue."
            )
        elif valid_r2.nunique() > 1:
            tips.append(
                f"The current best model is {best_model_name}. "
                f"Compare it with the next best model and then tune only after confirming the data and feature set are strong enough."
            )
    else:
        tips.append(
            f"The current best model is {best_model_name}. "
            f"Use it as the baseline, but focus first on data quality, outliers, and better feature engineering."
        )

    # fallback
    if not tips:
        tips = [
            f"The current best model is {best_model_name}. Review fit quality before tuning.",
            "Check whether the model is using meaningful predictors or mostly noisy row-level fields.",
            "Review residual spread and outliers before trying more complex models.",
            "If simple and tree-based models both struggle, improve features first.",
        ]

    return tips[:4]

# =========================================================
# STATS AND TESTS
# =========================================================
def statsmodels_summary(df: pd.DataFrame, target: str, task_type: str):
    work = df.copy()

    work = work.loc[:, ~work.columns.duplicated()].copy()

    if target not in work.columns:
        return None, "Statistical summary skipped because the target column was not found."

    target_series = work[target]
    if isinstance(target_series, pd.DataFrame):
        target_series = target_series.iloc[:, 0]

    work = work.dropna(subset=[target]).copy()

    for col in list(work.columns):
        if pd.api.types.is_datetime64_any_dtype(work[col]):
            work = work.drop(columns=[col])

    if work.shape[0] < 10:
        return None, "Not enough data for statistical summary."

    target_series = work[target]
    if isinstance(target_series, pd.DataFrame):
        target_series = target_series.iloc[:, 0]

    numeric_cols, categorical_cols, _ = split_columns(work)

    numeric_predictors = [c for c in numeric_cols if c != target]
    categorical_predictors = [
        c for c in categorical_cols
        if c != target and work[c].nunique(dropna=True) <= 10
    ]

    predictor_cols = numeric_predictors + categorical_predictors

    if not predictor_cols:
        return None, "No usable predictors for statistical summary."

    model_df = pd.concat(
        [target_series.rename(target), work[predictor_cols]],
        axis=1
    ).dropna()

    if model_df.empty:
        return None, "No usable rows remain for statistical summary."

    if task_type == "classification":
        unique_classes = pd.Series(model_df[target]).nunique(dropna=True)

        try:
            model_df[target] = model_df[target].astype("category")
            model_df["_target_code_"] = model_df[target].cat.codes

            formula = "Q('_target_code_') ~ " + " + ".join([f"Q('{c}')" for c in predictor_cols])

            if unique_classes == 2:
                model = smf.logit(formula=formula, data=model_df).fit(disp=False)
                out = model.summary2().tables[1].reset_index().rename(columns={"index": "feature"})
                return out, None

            # multiclass classification
            model = smf.mnlogit(formula=formula, data=model_df).fit(disp=False)
            out = model.summary2().tables[1].reset_index().rename(columns={"index": "feature"})
            return out, None

        except Exception as e:
            return None, f"Classification statistical summary unavailable: {e}"

    y = pd.to_numeric(model_df[target], errors="coerce")
    model_df = model_df.copy()
    model_df[target] = y
    model_df = model_df.dropna(subset=[target])

    if model_df.empty:
        return None, "Statistical summary is skipped because the regression target is not numeric."

    try:
        formula = f"Q('{target}') ~ " + " + ".join([f"Q('{c}')" for c in predictor_cols])
        model = smf.ols(formula=formula, data=model_df).fit()
        out = model.summary2().tables[1].reset_index().rename(columns={"index": "feature"})
        return out, None
    except Exception as e:
        return None, f"Regression model failed: {e}"
        
    # --------------------------
    # Classification summary
    # --------------------------
    if task_type == "classification":
        unique_classes = pd.Series(model_df[target]).nunique(dropna=True)

        if unique_classes != 2:
            return None, "Statistical summary is skipped because this classification target is not binary."

        try:
            model_df[target] = model_df[target].astype("category").cat.codes
            formula = f"Q('{target}') ~ " + " + ".join([f"Q('{c}')" for c in predictor_cols])
            model = smf.logit(formula=formula, data=model_df).fit(disp=False)
            out = model.summary2().tables[1].reset_index().rename(columns={"index": "feature"})
            return out, None
        except Exception as e:
            return None, f"Binary logistic summary unavailable: {e}"

    # --------------------------
    # Regression summary
    # --------------------------
    y = pd.to_numeric(model_df[target], errors="coerce")
    model_df = model_df.copy()
    model_df[target] = y
    model_df = model_df.dropna(subset=[target])

    if model_df.empty:
        return None, "Statistical summary is skipped because the regression target is not numeric."

    try:
        formula = f"Q('{target}') ~ " + " + ".join([f"Q('{c}')" for c in predictor_cols])
        model = smf.ols(formula=formula, data=model_df).fit()
        out = model.summary2().tables[1].reset_index().rename(columns={"index": "feature"})
        return out, None
    except Exception as e:
        return None, f"Regression model failed: {e}"

def cramers_v(x, y):
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()

    valid = x.index.intersection(y.index)
    x, y = x.loc[valid], y.loc[valid]

    if len(x) == 0:
        return np.nan

    table = pd.crosstab(x, y)

    if table.size == 0:
        return np.nan

    chi2 = stats.chi2_contingency(table)[0]
    n = table.sum().sum()
    r, k = table.shape

    return np.sqrt((chi2 / n) / max(min(k - 1, r - 1), 1))


def correlation_ratio(categories, measurements):
    categories = pd.Series(categories)
    measurements = safe_numeric(measurements)

    valid = categories.notna() & measurements.notna()
    categories = categories[valid]
    measurements = measurements[valid]

    if len(measurements) == 0:
        return np.nan

    grand_mean = measurements.mean()
    groups = [measurements[categories == cat] for cat in categories.unique()]

    numerator = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups if len(g) > 0)
    denominator = sum((measurements - grand_mean) ** 2)

    return 0.0 if denominator == 0 else np.sqrt(numerator / denominator)


def recommend_tests(df):
    numeric_cols, categorical_cols, _ = split_columns(df)

    recs = []

    if len(numeric_cols) >= 2:
        recs.append("Use correlation to measure relationship between numeric variables.")

    if len(categorical_cols) >= 2:
        recs.append("Use chi square or Cramer's V for categorical association.")

    if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
        recs.append("Use T test or ANOVA to compare numeric values across groups.")

    return recs


def run_custom_test_builder(df: pd.DataFrame):
    st.markdown("### Custom test builder")
    st.caption("Tests are shown based on the variable type you choose.")

    numeric_cols, categorical_cols, _ = split_columns(df)

    for rec in recommend_tests(df):
        st.markdown(f"- {rec}")

    var_type = st.radio(
        "Variable type",
        ["Numerical variables", "Categorical variables"],
        horizontal=True,
        key="test_builder_var_type",
    )

    with st.form("custom_test_form"):
        if var_type == "Numerical variables":
            test_name = st.selectbox(
                "Choose test",
                ["Correlation", "T test", "ANOVA", "F test for equal variance"],
                key="test_builder_num_test",
            )

            if test_name == "Correlation":
                if len(numeric_cols) < 2:
                    st.info("At least two numeric columns are required.")
                    submitted = st.form_submit_button("Run test")
                    return

                c1 = st.selectbox("First numeric column", numeric_cols, key="corr1")
                c2_options = [c for c in numeric_cols if c != c1]
                c2 = st.selectbox("Second numeric column", c2_options, key="corr2")

                submitted = st.form_submit_button("Run test")
                if submitted:
                    sub = df[[c1, c2]].copy()
                    sub[c1] = safe_numeric(sub[c1])
                    sub[c2] = safe_numeric(sub[c2])
                    sub = sub.dropna()

                    if len(sub) < 3:
                        st.info("Not enough valid data for correlation.")
                        return

                    corr, pval = stats.pearsonr(sub[c1], sub[c2])
                    st.write(
                        {
                            "correlation": round(float(corr), 4),
                            "p_value": round(float(pval), 6),
                        }
                    )

            elif test_name == "T test":
                if len(numeric_cols) < 1 or len(categorical_cols) < 1:
                    st.info("You need at least one numeric and one categorical column.")
                    submitted = st.form_submit_button("Run test")
                    return

                num_col = st.selectbox("Numeric column", numeric_cols, key="tt_num")
                cat_col = st.selectbox("Categorical grouping column", categorical_cols, key="tt_cat")

                temp_groups = (
                    df[cat_col]
                    .dropna()
                    .astype(str)
                    .value_counts()
                    .index
                    .tolist()
                )

                if len(temp_groups) < 2:
                    st.info("The selected categorical column needs at least 2 groups.")
                    submitted = st.form_submit_button("Run test")
                    return

                g1_label = st.selectbox("First group", temp_groups, key="tt_g1")
                g2_options = [g for g in temp_groups if g != g1_label]
                g2_label = st.selectbox("Second group", g2_options, key="tt_g2")

                submitted = st.form_submit_button("Run test")
                if submitted:
                    sub = df[[num_col, cat_col]].copy()
                    sub[num_col] = safe_numeric(sub[num_col])
                    sub[cat_col] = sub[cat_col].astype(str)
                    sub = sub.dropna(subset=[num_col, cat_col])

                    sub = sub[sub[cat_col].isin([g1_label, g2_label])]

                    g1 = sub.loc[sub[cat_col] == g1_label, num_col].dropna()
                    g2 = sub.loc[sub[cat_col] == g2_label, num_col].dropna()

                    if len(g1) < 2 or len(g2) < 2:
                        st.info("Each selected group needs at least 2 valid values.")
                        return

                    stat, pval = stats.ttest_ind(g1, g2, equal_var=False)

                    st.write(
                        {
                            "numeric_column": num_col,
                            "grouping_column": cat_col,
                            "group_1": str(g1_label),
                            "group_2": str(g2_label),
                            "mean_group_1": round(float(g1.mean()), 4),
                            "mean_group_2": round(float(g2.mean()), 4),
                            "t_statistic": round(float(stat), 4),
                            "p_value": round(float(pval), 6),
                        }
                    )

            
            elif test_name == "ANOVA":
                if len(numeric_cols) < 1 or len(categorical_cols) < 1:
                    st.info("You need at least one numeric and one categorical column.")
                    submitted = st.form_submit_button("Run test")
                    return

                num_col = st.selectbox("Numeric column", numeric_cols, key="anova_num")
                cat_col = st.selectbox("Categorical grouping column", categorical_cols, key="anova_cat")

                submitted = st.form_submit_button("Run test")
                if submitted:
                    sub = df[[num_col, cat_col]].copy()
                    sub[num_col] = safe_numeric(sub[num_col])
                    sub = sub.dropna(subset=[num_col, cat_col])

                    unique_groups = sub[cat_col].dropna().unique().tolist()

                    groups = []
                    used_group_names = []

                    for grp in unique_groups:
                        vals = sub.loc[sub[cat_col] == grp, num_col].dropna()
                        if len(vals) > 0:
                            groups.append(vals)
                            used_group_names.append(grp)

                    if len(groups) < 3:
                        st.info("ANOVA needs at least 3 groups with valid data.")
                        return

                    stat, pval = stats.f_oneway(*groups)

                    st.write(
                        {
                            "numeric_column": num_col,
                            "grouping_column": cat_col,
                            "groups_used": [str(g) for g in used_group_names],
                            "anova_f_statistic": round(float(stat), 4),
                            "p_value": round(float(pval), 6),
                        }
                    )

            elif test_name == "F test for equal variance":
                if len(numeric_cols) < 1 or len(categorical_cols) < 1:
                    st.info("You need at least one numeric and one categorical column.")
                    submitted = st.form_submit_button("Run test")
                    return

                num_col = st.selectbox("Numeric column", numeric_cols, key="f_num")
                cat_col = st.selectbox("Categorical grouping column", categorical_cols, key="f_cat")

                temp_groups = (
                    df[cat_col]
                    .dropna()
                    .astype(str)
                    .value_counts()
                    .index
                    .tolist()
                )

                if len(temp_groups) < 2:
                    st.info("The selected categorical column needs at least 2 groups.")
                    submitted = st.form_submit_button("Run test")
                    return

                g1_label = st.selectbox("First group", temp_groups, key="f_g1")
                g2_options = [g for g in temp_groups if g != g1_label]
                g2_label = st.selectbox("Second group", g2_options, key="f_g2")

                submitted = st.form_submit_button("Run test")
                if submitted:
                    sub = df[[num_col, cat_col]].copy()
                    sub[num_col] = safe_numeric(sub[num_col])
                    sub[cat_col] = sub[cat_col].astype(str)
                    sub = sub.dropna(subset=[num_col, cat_col])

                    sub = sub[sub[cat_col].isin([g1_label, g2_label])]

                    g1 = sub.loc[sub[cat_col] == g1_label, num_col].dropna()
                    g2 = sub.loc[sub[cat_col] == g2_label, num_col].dropna()

                    if len(g1) < 2 or len(g2) < 2:
                        st.info("Each selected group needs at least 2 valid numeric values.")
                        return

                    var1 = np.var(g1, ddof=1)
                    var2 = np.var(g2, ddof=1)

                    if var2 <= 1e-12:
                        st.info("The second selected group has near-zero variance, so the F test cannot be computed reliably.")
                        return

                    f_stat = var1 / var2
                    pval = 2 * min(
                        stats.f.cdf(f_stat, len(g1) - 1, len(g2) - 1),
                        stats.f.sf(f_stat, len(g1) - 1, len(g2) - 1),
                    )

                    st.write(
                        {
                            "numeric_column": num_col,
                            "grouping_column": cat_col,
                            "group_1": str(g1_label),
                            "group_2": str(g2_label),
                            "variance_group_1": round(float(var1), 4),
                            "variance_group_2": round(float(var2), 4),
                            "f_statistic": round(float(f_stat), 4),
                            "p_value": round(float(pval), 6),
                        }
                    )
        else:
            test_name = st.selectbox(
                "Choose test",
                ["Association (Chi square)", "Cramers V", "Association between category and number"],
                key="test_builder_cat_test",
            )

            if test_name == "Association (Chi square)":
                if len(categorical_cols) < 2:
                    st.info("At least two categorical columns are required.")
                    submitted = st.form_submit_button("Run test")
                    return

                c1 = st.selectbox("First categorical column", categorical_cols, key="chi1")
                c2_options = [c for c in categorical_cols if c != c1]
                c2 = st.selectbox("Second categorical column", c2_options, key="chi2")

                submitted = st.form_submit_button("Run test")
                if submitted:
                    sub = df[[c1, c2]].dropna()

                    if sub.empty:
                        st.info("Not enough valid data for chi square test.")
                        return

                    table = pd.crosstab(sub[c1], sub[c2])

                    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
                        st.info("Chi square test needs at least a 2 by 2 contingency table.")
                        return

                    chi2, pval, dof, _ = stats.chi2_contingency(table)

                    st.write(
                        {
                            "chi_square": round(float(chi2), 4),
                            "p_value": round(float(pval), 6),
                            "degrees_of_freedom": int(dof),
                        }
                    )

            elif test_name == "Cramers V":
                if len(categorical_cols) < 2:
                    st.info("At least two categorical columns are required.")
                    submitted = st.form_submit_button("Run test")
                    return

                c1 = st.selectbox("First categorical column", categorical_cols, key="cv1")
                c2_options = [c for c in categorical_cols if c != c1]
                c2 = st.selectbox("Second categorical column", c2_options, key="cv2")

                submitted = st.form_submit_button("Run test")
                if submitted:
                    sub = df[[c1, c2]].dropna()

                    if sub.empty:
                        st.info("Not enough valid data for Cramer's V.")
                        return

                    val = cramers_v(sub[c1], sub[c2])
                    st.write({"cramers_v": round(float(val), 4)})

            elif test_name == "Association between category and number":
                if len(categorical_cols) < 1 or len(numeric_cols) < 1:
                    st.info("You need one categorical and one numeric column.")
                    submitted = st.form_submit_button("Run test")
                    return

                c1 = st.selectbox("Categorical column", categorical_cols, key="eta_cat")
                c2 = st.selectbox("Numeric column", numeric_cols, key="eta_num")

                submitted = st.form_submit_button("Run test")
                if submitted:
                    sub = df[[c1, c2]].copy()
                    sub[c2] = safe_numeric(sub[c2])
                    sub = sub.dropna(subset=[c1, c2])

                    if sub.empty:
                        st.info("Not enough valid data for this association test.")
                        return

                    eta = correlation_ratio(sub[c1], sub[c2])
                    st.write({"association_strength": round(float(eta), 4)})

                    

# =========================================================
# SHAP AND IMPORTANCE
# =========================================================
def show_permutation_importance(best_bundle):
    st.markdown("### Advanced feature importance")

    try:
        model = best_bundle.get("model")
        X_test = best_bundle.get("X_test")
        y_test = best_bundle.get("y_test")
        feature_names = best_bundle.get("feature_names", [])

        if model is None or X_test is None or y_test is None:
            st.info("Permutation importance is unavailable because model outputs are incomplete.")
            return

        if len(y_test) == 0 or X_test.shape[0] == 0:
            st.info("No test data available for permutation importance.")
            return

        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()

        X_test = np.asarray(X_test)

        if X_test.ndim != 2:
            st.info("Permutation importance is unavailable because the test matrix is not two dimensional.")
            return

        if len(feature_names) == 0 or len(feature_names) != X_test.shape[1]:
            feature_names = [f"Feature {i+1}" for i in range(X_test.shape[1])]

        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=5,
            random_state=42,
        )

        imp_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": result.importances_mean,
            }
        ).sort_values("importance", ascending=False)

        if imp_df.empty:
            st.info("Permutation importance could not be computed.")
            return

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=imp_df.head(15), x="importance", y="feature", ax=ax)
        ax.set_title("Permutation Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        plot_matplotlib(fig)

    except Exception as e:
        st.info(f"Permutation importance unavailable: {e}")

def show_shap(best_bundle, best_model_name: str):
    st.markdown("### SHAP explainability")

    if not SHAP_AVAILABLE:
        st.info("SHAP is not installed in this environment.")
        return

    if not any(x in best_model_name for x in ["XGBoost", "Random Forest", "Decision Tree"]):
        st.info("SHAP visuals are shown here only for tree based models.")
        return

    model = best_bundle.get("model")
    X_test = best_bundle.get("X_test")
    feature_names = best_bundle.get("feature_names", [])
    task_type = best_bundle.get("task_type", "regression")

    if model is None or X_test is None:
        st.info("SHAP is unavailable because model outputs are incomplete.")
        return

    if X_test.shape[0] == 0:
        st.info("No test samples available for SHAP.")
        return

    try:
        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()

        X_test = np.asarray(X_test)

        if X_test.ndim != 2:
            st.info("SHAP is unavailable because the test matrix is not two dimensional.")
            return

        if len(feature_names) != X_test.shape[1]:
            feature_names = [f"Feature {i+1}" for i in range(X_test.shape[1])]

        X_sample = X_test[: min(120, X_test.shape[0])]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        explanation = None

        if isinstance(shap_values, list):
            class_options = list(range(len(shap_values)))
            class_idx = st.selectbox(
                "Choose class index for class specific SHAP",
                class_options,
                key="class_shap_idx",
            )

            selected_values = np.asarray(shap_values[class_idx])

            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)) and np.ndim(expected_value) > 0:
                selected_base = expected_value[class_idx]
            else:
                selected_base = expected_value

            explanation = shap.Explanation(
                values=selected_values,
                base_values=np.repeat(selected_base, selected_values.shape[0]),
                data=X_sample,
                feature_names=feature_names,
            )

        else:
            shap_values_arr = np.asarray(shap_values)

            if shap_values_arr.ndim == 3:
                class_options = list(range(shap_values_arr.shape[2]))
                class_idx = st.selectbox(
                    "Choose class index for class specific SHAP",
                    class_options,
                    key="class_shap_idx",
                )

                selected_values = shap_values_arr[:, :, class_idx]

                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)) and np.ndim(expected_value) > 0:
                    selected_base = expected_value[class_idx]
                else:
                    selected_base = expected_value

                explanation = shap.Explanation(
                    values=selected_values,
                    base_values=np.repeat(selected_base, selected_values.shape[0]),
                    data=X_sample,
                    feature_names=feature_names,
                )

            elif shap_values_arr.ndim == 2:
                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)) and np.ndim(expected_value) > 0:
                    selected_base = expected_value[0]
                else:
                    selected_base = expected_value

                explanation = shap.Explanation(
                    values=shap_values_arr,
                    base_values=np.repeat(selected_base, shap_values_arr.shape[0]),
                    data=X_sample,
                    feature_names=feature_names,
                )

            else:
                st.info("SHAP explanation is unavailable because the SHAP output shape is unsupported.")
                return

        st.write("Global SHAP beeswarm")
        plt.figure(figsize=(10, 5))
        shap.plots.beeswarm(explanation, max_display=15, show=False)
        fig1 = plt.gcf()
        plot_matplotlib(fig1)

        st.write("Global SHAP bar plot")
        plt.figure(figsize=(10, 5))
        shap.plots.bar(explanation, max_display=15, show=False)
        fig2 = plt.gcf()
        plot_matplotlib(fig2)

        if task_type == "classification" and hasattr(explanation, "values"):
            values = np.asarray(explanation.values)

            if values.ndim == 2:
                st.write("Class specific SHAP view is already shown for the selected class.")

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")
# =========================================================
# MODELING UI
# =========================================================

def get_business_mode_models(task_type: str, is_large_data: bool):
    if task_type == "regression":
        if is_large_data:
            return ["Ridge", "Random Forest Regressor"]
        return ["Linear Regression", "Ridge", "Random Forest Regressor"]

    if is_large_data:
        return ["Logistic Regression", "Random Forest Classifier"]
    return ["Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier"]

def run_data_readiness_checks(df: pd.DataFrame, target: str):
    work = df.dropna(subset=[target]).copy()
    numeric_cols, categorical_cols, _ = split_columns(work)

    missing_summary = work.isna().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

    constant_cols = [c for c in work.columns if c != target and work[c].nunique(dropna=True) <= 1]

    high_cardinality = []
    for col in categorical_cols:
        if col == target:
            continue
        nunique = work[col].nunique(dropna=True)
        if nunique > max(20, int(0.2 * len(work))):
            high_cardinality.append(col)

    high_corr_pairs = []
    usable_numeric = [c for c in numeric_cols if c != target]
    if len(usable_numeric) >= 2:
        corr_df = work[usable_numeric].corr(numeric_only=True)
        for i in range(len(corr_df.columns)):
            for j in range(i + 1, len(corr_df.columns)):
                val = corr_df.iloc[i, j]
                if pd.notna(val) and abs(val) >= 0.85:
                    high_corr_pairs.append(
                        (corr_df.columns[i], corr_df.columns[j], round(float(val), 3))
                    )

    warnings_list = []
    if len(work) < 50:
        warnings_list.append("Small dataset. Generalization may be unreliable.")
    if work.isna().mean().mean() > 0.15:
        warnings_list.append("Missing values are fairly high. Modeling may be unstable.")
    if high_cardinality:
        warnings_list.append("Some categorical columns have very high cardinality.")
    if high_corr_pairs:
        warnings_list.append("Highly correlated numeric features detected. Multicollinearity may be present.")
    if constant_cols:
        warnings_list.append("Constant or near-constant columns found.")

    return {
        "rows": work.shape[0],
        "columns": work.shape[1],
        "numeric_count": len(usable_numeric),
        "categorical_count": len([c for c in categorical_cols if c != target]),
        "duplicate_rows": int(work.duplicated().sum()),
        "missing_summary": missing_summary,
        "constant_cols": constant_cols,
        "high_cardinality": high_cardinality,
        "high_corr_pairs": high_corr_pairs,
        "warnings": warnings_list,
    }


def get_advanced_models(task_type: str, random_state=42):
    if task_type == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=random_state),
            "Lasso": Lasso(random_state=random_state),
            "Random Forest Regressor": RandomForestRegressor(
                random_state=random_state,
                n_estimators=200,
                n_jobs=-1,
            ),
            "Gradient Boosting Regressor": GradientBoostingRegressor(
                random_state=random_state
            ),
        }

    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest Classifier": RandomForestClassifier(
            random_state=random_state,
            n_estimators=200,
            n_jobs=-1,
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            random_state=random_state
        ),
        "SVM": SVC(probability=True, random_state=random_state),
        "KNN Classifier": KNeighborsClassifier(),
    }


def get_advanced_param_grids(task_type: str):
    if task_type == "regression":
        return {
            "Ridge": {
                "model__alpha": [0.01, 0.1, 1.0, 10.0]
            },
            "Lasso": {
                "model__alpha": [0.001, 0.01, 0.1, 1.0]
            },
            "Random Forest Regressor": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5],
            },
            "Gradient Boosting Regressor": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
            },
        }

    return {
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 10.0]
        },
        "Random Forest Classifier": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
        },
        "Gradient Boosting Classifier": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        },
        "SVM": {
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["rbf", "linear"],
        },
        "KNN Classifier": {
            "model__n_neighbors": [3, 5, 7, 9]
        },
    }

def select_best_advanced_model(df: pd.DataFrame, task_type: str) -> str:
    work = df.copy()

    if task_type == "regression":
        # normalize metrics
        work["score"] = (
            0.4 * work["CV Mean"] +
            0.3 * work["Test R2"] -
            0.2 * work["RMSE"] -
            0.1 * work["Overfitting Gap"]
        )

    else:
        work["score"] = (
            0.4 * work["CV Mean"] +
            0.3 * work["F1"] +
            0.2 * work["Test Accuracy"] -
            0.3 * work["Overfitting Gap"]
        )

    work = work.sort_values("score", ascending=False)

    return work.iloc[0]["Model"]


def advanced_model_interpretation(best_row: pd.Series, results_df: pd.DataFrame, task_type: str) -> list[str]:
    notes = []

    if task_type == "regression":
        test_r2 = best_row.get("Test R2", np.nan)
        train_r2 = best_row.get("Train R2", np.nan)
        cv_mean = best_row.get("CV Mean", np.nan)

        if pd.notna(test_r2) and pd.notna(cv_mean):
            if test_r2 >= 0.7 and cv_mean >= 0.6:
                notes.append("The best model shows strong predictive performance and generalizes reasonably well.")
            elif test_r2 >= 0.4:
                notes.append("The model captures a meaningful signal, but performance is still moderate.")
            else:
                notes.append("The model performance is weak, which suggests limited predictive signal in the current features.")

        if pd.notna(train_r2) and pd.notna(test_r2):
            if (train_r2 - test_r2) > 0.15:
                notes.append("The gap between training and test performance suggests overfitting.")
            else:
                notes.append("The training and test scores are reasonably close, which supports more stable generalization.")

        linear_models = results_df[results_df["Model"].isin(["Linear Regression", "Ridge", "Lasso"])]
        tree_models = results_df[results_df["Model"].isin(["Random Forest Regressor", "Gradient Boosting Regressor"])]

        if not linear_models.empty and not tree_models.empty:
            if tree_models["CV Mean"].max() > linear_models["CV Mean"].max() + 0.05:
                notes.append("Tree based models outperform linear models, which suggests likely non linear relationships.")
            elif linear_models["CV Mean"].max() >= tree_models["CV Mean"].max():
                notes.append("Linear models are competitive here, so the relationships may be reasonably structured without heavy non linearity.")

        if "Ridge" in results_df["Model"].values and "Linear Regression" in results_df["Model"].values:
            ridge_cv = results_df.loc[results_df["Model"] == "Ridge", "CV Mean"].max()
            linear_cv = results_df.loc[results_df["Model"] == "Linear Regression", "CV Mean"].max()
            if pd.notna(ridge_cv) and pd.notna(linear_cv) and ridge_cv > linear_cv:
                notes.append("Ridge performed better than standard linear regression, which suggests regularization improved stability.")

        return notes[:4]

    test_f1 = best_row.get("F1", np.nan)
    train_acc = best_row.get("Train Accuracy", np.nan)
    test_acc = best_row.get("Test Accuracy", np.nan)
    cv_mean = best_row.get("CV Mean", np.nan)

    if pd.notna(test_f1) and pd.notna(cv_mean):
        if test_f1 >= 0.85 and cv_mean >= 0.8:
            notes.append("The best classifier shows strong predictive quality and stable validation performance.")
        elif test_f1 >= 0.7:
            notes.append("The classifier performs reasonably well, though there is still room to improve class separation.")
        else:
            notes.append("Classification performance is limited, which suggests weak signal or class overlap in the current features.")

    if pd.notna(train_acc) and pd.notna(test_acc):
        if (train_acc - test_acc) > 0.10:
            notes.append("The gap between training and test accuracy suggests overfitting.")
        else:
            notes.append("Training and test accuracy are fairly close, which supports more reliable generalization.")

    linear_models = results_df[results_df["Model"].isin(["Logistic Regression"])]
    tree_models = results_df[results_df["Model"].isin(["Random Forest Classifier", "Gradient Boosting Classifier"])]

    if not linear_models.empty and not tree_models.empty:
        if tree_models["CV Mean"].max() > linear_models["CV Mean"].max() + 0.05:
            notes.append("Tree based classifiers outperform logistic regression, which suggests more complex decision boundaries.")
        else:
            notes.append("Logistic regression remains competitive, so class separation may be reasonably structured.")

    return notes[:4]


def advanced_model_improving_tips(best_row: pd.Series, results_df: pd.DataFrame, readiness: dict, task_type: str) -> list[str]:
    tips = []

    if task_type == "regression":
        if results_df["Test R2"].max() < 0.3:
            tips.append("Collect more relevant features and improve feature engineering because all models are showing weak signal.")
            tips.append("Recheck target quality and remove weak, noisy, or identifier-like predictors.")

        if best_row.get("Overfitting Gap", 0) > 0.15:
            tips.append("Reduce model complexity or enable hyperparameter tuning to control overfitting.")
            tips.append("Use stronger regularization or simpler models if performance gap remains wide.")

        if readiness.get("high_corr_pairs"):
            tips.append("Review highly correlated numeric features because multicollinearity may be reducing model stability.")

        if readiness.get("high_cardinality"):
            tips.append("Reduce very high-cardinality categorical columns or group rare categories before modeling.")

        best_model = best_row.get("Model", "")
        if best_model in ["Random Forest Regressor", "Gradient Boosting Regressor"]:
            tips.append("Since tree based models are performing well, explore interaction features and non linear patterns further.")

        return list(dict.fromkeys(tips))[:5]

    if results_df["F1"].max() < 0.7:
        tips.append("Improve class-related features because all classifiers are showing limited separation power.")
        tips.append("Check whether the target classes are noisy, imbalanced, or too overlapping.")

    if best_row.get("Overfitting Gap", 0) > 0.10:
        tips.append("Reduce classifier complexity or enable hyperparameter tuning to control overfitting.")

    if readiness.get("high_cardinality"):
        tips.append("Reduce very high-cardinality categorical columns or group rare categories before classification.")

    if readiness.get("high_corr_pairs"):
        tips.append("Review highly correlated numeric predictors because they may reduce classifier stability.")

    best_model = best_row.get("Model", "")
    if best_model in ["Random Forest Classifier", "Gradient Boosting Classifier"]:
        tips.append("Since tree based classifiers are performing well, explore interaction effects and non linear decision boundaries further.")

    return list(dict.fromkeys(tips))[:5]

def show_modeling(df: pd.DataFrame, target: str, business_mode: bool):
    try:
        if target is None or target not in df.columns:
            st.info("Select a valid target to run modeling.")
            return None, None

        task_type = infer_task_type(df[target])
        is_large_data = len(df) > 50000

        all_models = list(model_bank(task_type, is_large_data=is_large_data).keys())

        st.subheader("Model comparison")

        try:
            _, preview_removed = prepare_modeling_dataset(df, target)
            removed_preview = {
                "identifier_like_cols": preview_removed.get("identifier_like_cols", []),
                "leakage_name_cols": preview_removed.get("leakage_name_cols", []),
                "near_target_cols": preview_removed.get("near_target_cols", []),
                "constant_cols": preview_removed.get("constant_cols", []),
            }

            removed_any = any(len(v) > 0 for v in removed_preview.values())
            if removed_any:
                with st.expander("Columns removed before modeling", expanded=False):
                    for k, v in removed_preview.items():
                        if v:
                            st.write({k: v[:20]})
        except Exception as e:
            st.warning(f"Modeling readiness warning: {e}")

        results_df = None
        best_name = None
        best_bundle = None

        current_schema_signature = (
            tuple(df.columns.tolist()),
            df.shape,
            target,
        )

        if business_mode:
            selected_models = [m for m in get_business_mode_models(task_type, is_large_data) if m in all_models]
            enable_fs = False

            current_config = {
                "target": target,
                "selected_models": tuple(selected_models),
                "enable_fs": enable_fs,
                "mode": "business",
                "schema_signature": current_schema_signature,
            }

            cached_config = st.session_state.get("business_model_config")
            cached_result = st.session_state.get("business_model_result")

            if cached_result is not None and cached_config == current_config:
                task_type, results_df, best_name, best_bundle = cached_result
                run_models = False
                st.success(f"Using saved model results: {best_name}")
            else:
                run_models = True

        else:
            st.info(
                "Technical Mode lets you choose models manually. "
                "Use the button below to run the selected models."
            )

            default_models = [m for m in get_business_mode_models(task_type, is_large_data) if m in all_models]
            if not default_models and all_models:
                default_models = all_models[: min(3, len(all_models))]

            selected_models = st.multiselect(
                "Select models",
                all_models,
                default=default_models,
                help="Choose one or more basic models to compare.",
                key="basic_model_select",
            )

            with st.expander("What each model is for", expanded=False):
                for m in all_models:
                    st.markdown(f"- **{m}**: {recommended_model_label(task_type, m)}")

            enable_fs = st.checkbox(
                "Enable feature selection",
                value=False,
                help="Usually keep this off for large or high-cardinality datasets.",
                key="basic_enable_fs",
            )

            if not selected_models:
                st.warning("Select at least one model to continue.")
                return None, None

            run_models = st.button("Run selected models", type="primary", key="run_basic_models")

            current_config = {
                "target": target,
                "selected_models": tuple(selected_models),
                "enable_fs": enable_fs,
                "mode": "technical",
                "schema_signature": current_schema_signature,
            }

            cached_config = st.session_state.get("technical_model_config")
            cached_result = st.session_state.get("technical_model_result")

            if run_models:
                pass
            elif cached_result is not None and cached_config == current_config:
                task_type, results_df, best_name, best_bundle = cached_result
                st.success(f"Using saved model results: {best_name}")
            else:
                st.caption("Choose the models you want, then click **Run selected models**.")

        if run_models:
            try:
                with st.spinner("Training basic models..."):
                    task_type, results_df, best_name, best_bundle = train_models(
                        df,
                        target,
                        selected_models=selected_models,
                        enable_feature_selection=enable_fs,
                    )

                if business_mode:
                    st.session_state["business_model_config"] = {
                        "target": target,
                        "selected_models": tuple(selected_models),
                        "enable_fs": enable_fs,
                        "mode": "business",
                        "schema_signature": current_schema_signature,
                    }
                    st.session_state["business_model_result"] = (
                        task_type,
                        results_df,
                        best_name,
                        best_bundle,
                    )
                else:
                    st.session_state["technical_model_config"] = {
                        "target": target,
                        "selected_models": tuple(selected_models),
                        "enable_fs": enable_fs,
                        "mode": "technical",
                        "schema_signature": current_schema_signature,
                    }
                    st.session_state["technical_model_result"] = (
                        task_type,
                        results_df,
                        best_name,
                        best_bundle,
                    )

            except Exception as e:
                st.error(f"Modeling failed: {e}")
                st.exception(e)
                st.info(
                    "Try using a cleaner target, or reduce high-cardinality columns like IDs, names, order numbers, and text-heavy fields."
                )
                results_df, best_name, best_bundle = None, None, None

        if results_df is not None and not results_df.empty:
            st.dataframe(results_df, width="stretch")

            if best_name is not None:
                st.success(f"Recommended model: {best_name}")
                st.caption(recommended_model_label(task_type, best_name))

            if best_bundle is not None and best_bundle.get("removed_info"):
                removed_info = best_bundle["removed_info"]
                removed_total = sum(len(v) for v in removed_info.values())
                if removed_total > 0:
                    st.caption(f"{removed_total} column(s) were excluded before modeling to reduce leakage and unstable results.")

            if task_type == "regression" and best_bundle is not None:
                compare_df = pd.DataFrame(
                    {
                        "Actual": pd.Series(best_bundle["y_test"]).reset_index(drop=True),
                        "Predicted": pd.Series(best_bundle["pred"]).reset_index(drop=True),
                    }
                )

                fig, ax = plt.subplots(figsize=(7.5, 5))
                sns.scatterplot(data=compare_df, x="Actual", y="Predicted", ax=ax)
                ax.set_title("Actual vs Predicted")
                plot_matplotlib(fig)

                residuals = compare_df["Actual"] - compare_df["Predicted"]
                resid_df = pd.DataFrame(
                    {
                        "Predicted": compare_df["Predicted"],
                        "Residual": residuals,
                    }
                )

                resid_fig = px.scatter(resid_df, x="Predicted", y="Residual", title="Residual Plot")
                resid_fig.add_hline(y=0, line_dash="dash")
                plot_plotly(resid_fig)

                st.info(residual_interpretation(best_bundle["y_test"], best_bundle["pred"]))

            elif task_type != "regression" and best_bundle is not None:
                y_true_raw = pd.Series(best_bundle["y_test"]).reset_index(drop=True)
                y_pred_raw = pd.Series(best_bundle["pred"]).reset_index(drop=True)

                display_labels = sorted(pd.Series(y_true_raw.astype(str)).unique().tolist())
                y_true = y_true_raw.astype(str)
                y_pred = y_pred_raw.astype(str)

                cm = metrics.confusion_matrix(y_true, y_pred, labels=display_labels)
                fig, ax = plt.subplots(figsize=(6.5, 5))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="viridis",
                    ax=ax,
                    xticklabels=display_labels,
                    yticklabels=display_labels,
                )
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                plot_matplotlib(fig)

        st.markdown("## Advanced Modeling")
        enable_advanced = st.checkbox("Enable advanced modeling", key="enable_advanced_modeling")

        if enable_advanced:
            st.info("Advanced Modeling compares stronger models, cross validation stability, diagnostics, and explainability.")
            st.caption("Keep advanced modeling off unless you need it. It is heavier than the basic model comparison.")

        if results_df is not None and best_name is not None:
            with st.expander("Statistical summary", expanded=False):
                run_stats_summary = st.button("Generate statistical summary", key="run_stats_summary_btn")
                if run_stats_summary:
                    sm_table, sm_error = statsmodels_summary(df, target, task_type)
                    if sm_table is not None:
                        st.dataframe(sm_table, width="stretch", height=320)
                    else:
                        st.info(sm_error)
                else:
                    st.caption("Click only when you need the statistical summary.")

            with st.expander("Custom test builder", expanded=False):
                enable_custom_tests = st.checkbox("Enable custom tests", key="enable_custom_tests_block")
                if enable_custom_tests:
                    run_custom_test_builder(df)
                else:
                    st.caption("Turn this on only when you want to run extra tests.")

            with st.expander("Advanced feature importance", expanded=False):
                run_expensive_importance = st.button(
                    "Generate feature importance and SHAP",
                    key="run_expensive_importance_btn",
                )
                if run_expensive_importance and best_bundle is not None:
                    show_permutation_importance(best_bundle)
                    show_shap(best_bundle, best_name)
                elif best_bundle is not None:
                    st.caption("Click the button only when you want to compute these expensive diagnostics.")

            st.markdown("### Improving tips")
            if best_bundle is not None:
                for tip in model_improving_tips(
                    task_type,
                    best_name,
                    results_df=results_df,
                    y_true=best_bundle["y_test"],
                    y_pred=best_bundle["pred"],
                ):
                    st.markdown(f"- {tip}")

            st.markdown("### Connected business insights")
            for item in build_key_insights(df, target, best_name):
                st.markdown(f"- {item}")

        return results_df, best_name

    except Exception as e:
        st.error(f"Technical Mode failed: {e}")
        st.exception(e)
        return None, None

# =========================================================
# FORECASTING
# =========================================================
def simple_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 6):
    if date_col not in df.columns or value_col not in df.columns:
        return None, None, "Selected forecast columns were not found."

    temp = df[[date_col, value_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[value_col] = safe_numeric(temp[value_col])
    temp = temp.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if temp.empty or temp[date_col].nunique() < 4:
        return None, None, "Not enough clean time points for forecasting."

    daily_span = (temp[date_col].max() - temp[date_col].min()).days

    if daily_span > 90:
        freq = "MS"
        temp["period"] = temp[date_col].dt.to_period("M").dt.to_timestamp()
    elif daily_span > 14:
        freq = "W-MON"
        temp["period"] = temp[date_col].dt.to_period("W-MON").dt.start_time
    else:
        freq = "D"
        temp["period"] = temp[date_col].dt.floor("D")

    series = (
        temp.groupby("period", as_index=False)[value_col]
        .sum()
        .sort_values("period")
        .rename(columns={"period": "ds", value_col: "y"})
    )

    if len(series) < 4:
        return None, None, "Not enough aggregated time points for forecasting."

    # Fill missing periods so the time trend is regular
    full_index = pd.date_range(series["ds"].min(), series["ds"].max(), freq=freq)
    series = (
        series.set_index("ds")
        .reindex(full_index, fill_value=0)
        .rename_axis("ds")
        .reset_index()
    )

    y = pd.to_numeric(series["y"], errors="coerce")
    if y.isna().all() or len(y) < 4:
        return None, None, "Forecasting failed because the value series is not usable."

    y = y.fillna(0).astype(float)
    x = np.arange(len(y), dtype=float)

    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return None, None, "Forecasting failed while fitting the trend."

    future_x = np.arange(len(y), len(y) + periods, dtype=float)
    future_dates = pd.date_range(series["ds"].iloc[-1] + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
    future_vals = intercept + slope * future_x

    # Clip only if all historical values are non-negative
    if (y >= 0).all():
        future_vals = np.maximum(future_vals, 0)

    history = series.rename(columns={"ds": date_col, "y": value_col})
    forecast = pd.DataFrame({date_col: future_dates, value_col: future_vals})

    return history, forecast, None


# =========================================================
# EXPORTS
# =========================================================
def _clean_pdf_text(text):
    if text is None:
        return ""
    text = str(text)
    replacements = {
        "•": "-",
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text).strip()
    return text.encode("latin-1", "replace").decode("latin-1")


def pdf_write_title(pdf, text):
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, _clean_pdf_text(text))
    pdf.ln(1)


def pdf_write_subtitle(pdf, text):
    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 8, _clean_pdf_text(text))
    pdf.ln(1)


def pdf_write_paragraph(pdf, text):
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, _clean_pdf_text(text))
    pdf.ln(1)


def pdf_write_bullets(pdf, items):
    pdf.set_font("Helvetica", "", 11)

    if not items:
        pdf.multi_cell(0, 7, "- No items available.")
        pdf.ln(1)
        return

    for item in items:
        pdf.multi_cell(0, 7, _clean_pdf_text(f"- {item}"))
    pdf.ln(1)


def create_manager_pdf_report(title, dataset_overview, chart_interpretations, model_summary, final_recommendations):
    if not FPDF_AVAILABLE or FPDF is None:
        raise ValueError("PDF export is unavailable because fpdf is not installed.")

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    pdf_write_title(pdf, title)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        7,
        _clean_pdf_text(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
        ln=1,
    )
    pdf.ln(2)

    sections = [
        ("1. Dataset Overview", dataset_overview),
        ("2. Visual Insights", chart_interpretations),
        ("3. Model Summary", model_summary),
        ("4. Business Interpretation", [
            "This report summarizes the main patterns in simple business language.",
            "It highlights what matters most and what action can be taken next.",
        ]),
        ("5. Recommendations", final_recommendations),
    ]

    for section_title, items in sections:
        pdf_write_subtitle(pdf, section_title)
        if isinstance(items, str):
            pdf_write_paragraph(pdf, items)
        else:
            pdf_write_bullets(pdf, items)

    output = pdf.output(dest="S")
    return output.encode("latin-1", errors="replace") if isinstance(output, str) else bytes(output)


def _safe_sheet_name(name: str, used_names: set[str]) -> str:
    name = re.sub(r"[:\\/*?\[\]]", "_", str(name)).strip()
    name = name[:31] if name else "Sheet"

    base = name
    counter = 1
    while name in used_names:
        suffix = f"_{counter}"
        name = f"{base[:31 - len(suffix)]}{suffix}"
        counter += 1

    used_names.add(name)
    return name


def create_excel_bytes(
    clean_df: pd.DataFrame,
    model_df: pd.DataFrame | None = None,
    pivots: dict[str, pd.DataFrame] | None = None,
):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        used_names = set()

        clean_sheet = _safe_sheet_name("cleaned_data", used_names)
        clean_df.to_excel(writer, sheet_name=clean_sheet, index=False)

        if model_df is not None and not model_df.empty:
            model_sheet = _safe_sheet_name("model_comparison", used_names)
            model_df.to_excel(writer, sheet_name=model_sheet, index=False)

        if pivots:
            for name, pivot_df in pivots.items():
                if pivot_df is not None and not pivot_df.empty:
                    sheet_name = _safe_sheet_name(name, used_names)
                    pivot_df.to_excel(writer, sheet_name=sheet_name, index=False)

    output.seek(0)
    return output.getvalue()


# =========================================================
# UI SECTIONS
# =========================================================
def show_sidebar(profile: DataProfile):
    st.sidebar.header("Mr Ready")
    st.sidebar.caption("Office style analytics with clear explanations.")
    st.sidebar.write(f"Rows: {profile.rows:,}")
    st.sidebar.write(f"Columns: {profile.cols}")
    st.sidebar.write(f"Duplicates: {profile.duplicates}")
    st.sidebar.write(f"Memory: {profile.memory_mb:.2f} MB")
    st.sidebar.write(f"Numeric columns: {len(profile.numeric_cols)}")
    st.sidebar.write(f"Categorical columns: {len(profile.categorical_cols)}")
    st.sidebar.write(f"Datetime columns: {len(profile.datetime_cols)}")

    with st.sidebar.expander("Missing values"):
        st.dataframe(profile.missing_summary, use_container_width=True, height=260)

    with st.sidebar.expander("Column names"):
        st.write(list(profile.missing_summary["column"].values))


# =========================================================
# SMART TARGET SELECTION
# =========================================================
def suggest_target_column(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    keywords = ["target", "label", "outcome", "sales", "revenue", "price", "churn", "profit"]

    for col in all_cols:
        if any(k in str(col).lower() for k in keywords):
            return col

    for col in numeric_cols:
        if df[col].nunique(dropna=True) > 10:
            return col

    return all_cols[-1] if all_cols else ""


def target_warning(df: pd.DataFrame, target: str) -> list[str]:
    warnings_list = []

    if target not in df.columns:
        return ["Selected target was not found in the dataset."]

    target_non_null = df[target].dropna()

    if not target_non_null.empty and target_non_null.nunique() == len(target_non_null):
        warnings_list.append("⚠️ This column looks like a unique ID. Not suitable as target.")

    if target_non_null.nunique() == 1:
        warnings_list.append("⚠️ This column has only one value. Cannot be used for analysis.")

    missing_pct = float(df[target].isna().mean() * 100)
    if missing_pct > 40:
        warnings_list.append(f"⚠️ High missing values ({missing_pct:.1f}%). This may reduce model quality.")

    if (pd.api.types.is_object_dtype(df[target]) or pd.api.types.is_string_dtype(df[target])) and target_non_null.nunique() > 50:
        warnings_list.append("⚠️ Too many categories. This may not be a good target.")

    return warnings_list


def show_target_selector(df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("Upload a dataset to begin.")
        st.stop()

    st.markdown("### Select Target Column")
    st.caption("Target is only needed for prediction tasks in Technical Mode.")

    all_cols = df.columns.tolist()

    target = st.selectbox(
        "Select target column",
        ["-- Select target --"] + all_cols,
        index=0,
        key="target_selector_main",
    )

    if target == "-- Select target --":
        st.info("No target selected yet.")
        return None

    task_type = infer_task_type(df[target])
    st.info(f"Selected target '{target}' will be treated as {task_type.upper()}.")

    for w in target_warning(df, target):
        st.warning(w)

    return target

# =========================================================
# OVERVIEW + INSIGHTS
# =========================================================
def show_overview(df: pd.DataFrame, clean_df: pd.DataFrame, plan_df: pd.DataFrame, target: str):
    profile = profile_data(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{profile.rows:,}")
    c2.metric("Columns", profile.cols)
    c3.metric("Duplicates", profile.duplicates)
    c4.metric("Missing Cells", int(df.isna().sum().sum()))

    with st.container(border=True):
        st.subheader("Data preview")
        st.dataframe(df.head(15), use_container_width=True)

    with st.container(border=True):
        st.subheader("Missing value treatment")
        if plan_df is None or plan_df.empty:
            st.success("No missing values needed treatment.")
        else:
            st.dataframe(plan_df, use_container_width=True)

    with st.container(border=True):
        st.subheader("Plain language summary")
        for item in business_summary(clean_df, target):
            st.markdown(f"- {item}")

        with st.expander("Statistical summary", expanded=False):
            sm_table, sm_error = statsmodels_summary(clean_df, target, infer_task_type(clean_df[target]))
            if sm_table is not None:
                st.dataframe(sm_table, use_container_width=True, height=320)
            else:
                st.info(sm_error)

def show_business_overview(df: pd.DataFrame, clean_df: pd.DataFrame, plan_df: pd.DataFrame):
    profile = profile_data(clean_df)
    numeric_cols, categorical_cols, datetime_cols = split_columns(clean_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{profile.rows:,}")
    c2.metric("Columns", profile.cols)
    c3.metric("Duplicates", profile.duplicates)
    c4.metric("Missing Cells", int(df.isna().sum().sum()))

    with st.container(border=True):
        st.subheader("Data preview")
        st.dataframe(clean_df.head(20), use_container_width=True)

    with st.container(border=True):
        st.subheader("Cleaning summary")
        if plan_df is None or plan_df.empty:
            st.success("No missing values needed treatment.")
        else:
            st.dataframe(plan_df, use_container_width=True)

    with st.container(border=True):
        st.subheader("Business summary")
        st.markdown(f"- Rows: {clean_df.shape[0]:,}")
        st.markdown(f"- Columns: {clean_df.shape[1]}")
        st.markdown(f"- Numeric columns: {len(numeric_cols)}")
        st.markdown(f"- Categorical columns: {len(categorical_cols)}")
        st.markdown(f"- Date columns: {len(datetime_cols)}")
        st.markdown(f"- Duplicate rows: {int(clean_df.duplicated().sum())}")

        missing_pct = float(clean_df.isna().mean().mean() * 100)
        st.markdown(f"- Average missingness: {missing_pct:.1f}%")

    with st.container(border=True):
        st.subheader("What you can do next")
        st.markdown("- Clean the data")
        st.markdown("- Join another file")
        st.markdown("- Forecast if a date column exists")
        st.markdown("- Export the prepared dataset")

def detect_outliers_iqr(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    outlier_summary = {}

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = outliers.shape[0]

        if outlier_count > 0:
            outlier_summary[col] = outlier_count

    return dict(sorted(outlier_summary.items(), key=lambda x: x[1], reverse=True))

def get_outlier_summary_iqr(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    numeric_cols, _, _ = split_columns(df)

    ignore_keywords = [
        "id", "transaction", "invoice", "order", "customer", "row", "index", "code"
    ]

    for col in numeric_cols:
        col_lower = str(col).lower().strip()

        if any(k in col_lower for k in ignore_keywords):
            continue

        s = safe_numeric(df[col])

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (s < lower) | (s > upper)
        outlier_count = int(mask.sum())

        if outlier_count == 0:
            continue

        outlier_pct = round((outlier_count / max(len(df), 1)) * 100, 2)
        row_numbers = (np.where(mask)[0] + 2).tolist()

        summary_rows.append({
            "column": col,
            "row_numbers": ", ".join(map(str, row_numbers[:50])) + (" ..." if len(row_numbers) > 50 else ""),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
            "reason": f"Values below {lower:.4f} or above {upper:.4f} fall outside the IQR 1.5 rule."
        })

    if not summary_rows:
        return pd.DataFrame(columns=[
            "column", "row_numbers", "lower_bound", "upper_bound",
            "outlier_count", "outlier_pct", "reason"
        ])

    outlier_df = pd.DataFrame(summary_rows)
    outlier_df = outlier_df.sort_values("outlier_count", ascending=False).reset_index(drop=True)
    return outlier_df
        
def show_data_cleaning_studio(df: pd.DataFrame, key_prefix: str = "default") -> pd.DataFrame:
    st.subheader("Data Cleaning Studio")

    work = df.copy()
    before_rows, before_cols = work.shape

    with st.form(key=f"{key_prefix}_cleaning_form"):
        st.markdown("### Basic cleaning")
        drop_dupes = st.checkbox("Drop duplicate rows", key=f"{key_prefix}_clean_drop_dupes")
        drop_empty_rows = st.checkbox("Drop fully empty rows", key=f"{key_prefix}_clean_drop_empty_rows")
        drop_empty_cols = st.checkbox("Drop fully empty columns", key=f"{key_prefix}_clean_drop_empty_cols")
        drop_high_missing_cols = st.checkbox(
            "Drop columns with more than 75% missing values",
            key=f"{key_prefix}_clean_drop_high_missing_cols"
        )
        standardize_names = st.checkbox("Standardize column names", key=f"{key_prefix}_clean_standardize_names")
        trim_text = st.checkbox("Trim whitespace in text columns", key=f"{key_prefix}_clean_trim_text")
        fix_case = st.checkbox("Convert text columns to title case", key=f"{key_prefix}_clean_fix_case")

        st.markdown("### Missing values")
        missing_action = st.selectbox(
            "Missing value handling",
            [
                "Leave as is",
                "Drop rows with missing values",
                "Fill numeric with median",
                "Fill numeric with mean",
                "Fill categorical with mode",
            ],
            key=f"{key_prefix}_clean_missing_action",
        )

        st.markdown("### Outlier handling")
        outlier_action = st.selectbox(
            "Outlier handling",
            [
                "Detect only",
                "Cap numeric outliers using IQR",
                "Remove rows outside IQR",
            ],
            key=f"{key_prefix}_clean_outlier_action",
        )

        apply_cleaning = st.form_submit_button("Apply cleaning", type="primary")

    if apply_cleaning:
        if drop_dupes:
            work = work.drop_duplicates()

        if drop_empty_rows:
            work = work.dropna(how="all")

        if drop_empty_cols:
            work = work.dropna(axis=1, how="all")

        if drop_high_missing_cols:
            high_missing_cols = [
                col for col in work.columns
                if work[col].isna().mean() >= 0.75
            ]
            if high_missing_cols:
                work = work.drop(columns=high_missing_cols)

        if standardize_names:
            work.columns = (
                pd.Index(work.columns)
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(" ", "_", regex=False)
            )

        text_cols = work.select_dtypes(include=["object", "string"]).columns.tolist()

        if trim_text:
            for col in text_cols:
                work[col] = work[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

        if fix_case:
            for col in text_cols:
                work[col] = work[col].apply(lambda x: x.title() if isinstance(x, str) else x)

        numeric_cols, categorical_cols, _ = split_columns(work)

        if missing_action == "Drop rows with missing values":
            work = work.dropna()

        elif missing_action == "Fill numeric with median":
            for col in numeric_cols:
                s = safe_numeric(work[col])
                work[col] = s.fillna(s.median())

        elif missing_action == "Fill numeric with mean":
            for col in numeric_cols:
                s = safe_numeric(work[col])
                work[col] = s.fillna(s.mean())

        elif missing_action == "Fill categorical with mode":
            for col in categorical_cols:
                mode = work[col].mode(dropna=True)
                if not mode.empty:
                    work[col] = work[col].fillna(mode.iloc[0])

        outlier_df = get_outlier_summary_iqr(work)

        if outlier_action in ["Cap numeric outliers using IQR", "Remove rows outside IQR"]:
            numeric_cols, _, _ = split_columns(work)

            for col in numeric_cols:
                col_lower = str(col).lower().strip()

                if any(k in col_lower for k in ["id", "transaction", "invoice", "order", "customer", "row", "index", "code"]):
                    continue

                s = safe_numeric(work[col])
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1

                if pd.isna(iqr) or iqr == 0:
                    continue

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                if outlier_action == "Cap numeric outliers using IQR":
                    work[col] = s.clip(lower, upper)
                else:
                    keep_mask = ((s >= lower) & (s <= upper)) | s.isna()
                    work = work.loc[keep_mask].copy()

        after_rows, after_cols = work.shape

        st.success("Cleaning applied.")
        st.write({
            "rows_before": before_rows,
            "rows_after": after_rows,
            "cols_before": before_cols,
            "cols_after": after_cols,
            "affected_rows": before_rows - after_rows,
            "affected_cols": before_cols - after_cols,
        })

        if outlier_action == "Detect only":
            if not outlier_df.empty:
                st.markdown("### Outliers detected by column using IQR")
                st.dataframe(outlier_df, use_container_width=True, hide_index=True)
            else:
                st.success("No outliers detected using IQR with 1.5 range.")

        st.dataframe(work.head(20), use_container_width=True)
        return work

    st.dataframe(work.head(20), use_container_width=True)
    return None


def show_join_tables(base_df: pd.DataFrame, key_prefix: str = "default") -> pd.DataFrame:
    st.subheader("Join Tables")

    uploaded_join_file = st.file_uploader(
        "Upload second file to join",
        type=["csv", "xls", "xlsx", "xlsm", "xlsb"],
        key=f"{key_prefix}_join_file_uploader",
    )

    if uploaded_join_file is None:
        st.info("Upload a second file if you want to merge tables.")
        return base_df

    join_file_bytes = uploaded_join_file.getvalue()
    join_file_hash = hashlib.md5(join_file_bytes).hexdigest()
    join_cache_key = f"{key_prefix}_join_cache"

    cached_join = st.session_state.get(join_cache_key)

    if cached_join and cached_join.get("hash") == join_file_hash:
        right_df = cached_join["df"]
    else:
        try:
            right_df = load_uploaded_file_from_bytes(
                join_file_bytes,
                uploaded_join_file.name,
            )
            st.session_state[join_cache_key] = {
                "hash": join_file_hash,
                "df": right_df,
            }
        except Exception as e:
            st.error(f"Second file could not be loaded: {e}")
            return base_df

    if right_df is None or right_df.empty:
        st.warning("Second file could not be loaded.")
        return base_df

    st.markdown("### Join setup")

    left_key = st.selectbox(
        "Left join key",
        base_df.columns.tolist(),
        key=f"{key_prefix}_left_join_key",
    )

    right_key = st.selectbox(
        "Right join key",
        right_df.columns.tolist(),
        key=f"{key_prefix}_right_join_key",
    )

    join_type = st.selectbox(
        "Join type",
        ["inner", "left", "right", "outer"],
        key=f"{key_prefix}_join_type",
    )

    if st.button("Create merged dataset", type="primary", key=f"{key_prefix}_create_joined_dataset"):
        try:
            left_series = base_df[left_key]
            right_series = right_df[right_key]

            left_dup = int(left_series.duplicated().sum())
            right_dup = int(right_series.duplicated().sum())

            left_non_null = left_series.dropna()
            right_non_null = right_series.dropna()

            left_matches = int(left_non_null.isin(right_non_null).sum())
            right_matches = int(right_non_null.isin(left_non_null).sum())

            merged = base_df.merge(
                right_df,
                how=join_type,
                left_on=left_key,
                right_on=right_key,
                indicator=True,
                suffixes=("_left", "_right"),
            )

            st.markdown("### Join quality summary")
            st.write({
                "left_rows": len(base_df),
                "right_rows": len(right_df),
                "merged_rows": len(merged),
                "matched_left_keys": left_matches,
                "matched_right_keys": right_matches,
                "left_duplicate_keys": left_dup,
                "right_duplicate_keys": right_dup,
            })

            if left_dup > 0 or right_dup > 0:
                st.warning("Duplicate join keys found. This may create one-to-many or many-to-many joins.")

            st.markdown("### Merged preview")
            st.dataframe(merged.head(20), use_container_width=True)

            st.success("Merged dataset created.")
            return merged.drop(columns=["_merge"], errors="ignore")

        except Exception as e:
            st.error(f"Join failed: {e}")
            return base_df

    return base_df

def show_insights(clean_df: pd.DataFrame, target: str | None = None):
    with st.container(border=True):
        st.subheader("🔍 Key Insights")
        for ins in generate_key_insights(clean_df, target):
            st.markdown(f"- {ins}")

    with st.container(border=True):
        st.subheader("📖 Executive Summary")
        st.write(executive_story(clean_df, target))

    with st.container(border=True):
        st.subheader("💡 Business Recommendations")
        for rec in generate_business_recommendations(clean_df, target):
            st.markdown(f"- {rec}")


# =========================================================
# BUSINESS VISUALS
# =========================================================

def show_custom_chart_builder(df: pd.DataFrame, target: str | None = None):
    st.markdown("### Custom chart builder")

    chart_type = st.selectbox(
        "Chart type",
        [
            "Column Chart",
            "Bar Chart",
            "Line Chart",
            "Area Chart",
            "Pie Chart",
            "Donut Chart",
            "Stacked Bar Chart",
            "Combo Chart",
        ],
        key="custom_business_chart_type",
    )

    all_cols = df.columns.tolist()
    numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not all_cols:
        st.info("No columns available for chart building.")
        return

    x_col = st.selectbox("X axis or category column", all_cols, key="custom_x")
    y_choices = numeric_cols if numeric_cols else all_cols
    y_col = st.selectbox("Y axis or value column", y_choices, key="custom_y")
    color_col = st.selectbox("Second series for color or line", ["None"] + numeric_cols, key="custom_color")

    if chart_type in ["Column Chart", "Bar Chart", "Line Chart", "Area Chart", "Stacked Bar Chart", "Combo Chart"]:
        work = df.copy()

        if y_col in work.columns and not pd.api.types.is_numeric_dtype(work[y_col]):
            work[y_col] = safe_numeric(work[y_col])

        if y_col not in work.columns or work[y_col].notna().sum() == 0:
            st.info("The selected Y column is not numeric enough for this chart.")
            return

        grouped = work.groupby(x_col, dropna=False)[y_col].mean().reset_index()
        line_col = None

        if color_col != "None" and color_col in work.columns and color_col not in [x_col, y_col]:
            second = work.groupby(x_col, dropna=False)[color_col].mean().reset_index()
            grouped = grouped.merge(second, on=x_col, how="left")
            line_col = color_col

        fig = make_business_chart(
            grouped,
            chart_type,
            x_col=x_col,
            y_col=y_col,
            color_col=line_col,
            title=f"{chart_type}: {y_col} by {x_col}",
        )
        plot_plotly(fig)

        st.markdown("**Interpretation**")
        st.info(business_chart_story(work, chart_type, x_col, y_col, target))

    elif chart_type in ["Pie Chart", "Donut Chart"]:
        grouped = df[x_col].dropna().value_counts().reset_index()
        grouped.columns = [x_col, "count"]

        if grouped.empty:
            st.info("Not enough valid data for this chart.")
            return

        fig = make_business_chart(
            grouped,
            chart_type,
            x_col=x_col,
            y_col="count",
            title=f"{chart_type}: {x_col}",
        )
        plot_plotly(fig)

        st.markdown("**Interpretation**")
        st.info(business_chart_story(grouped, chart_type, x_col, "count", target))

def recommended_chart_interpretation(grouped: pd.DataFrame, x_col: str, y_col: str, chart_type: str) -> str:
    if grouped is None or grouped.empty or x_col not in grouped.columns or y_col not in grouped.columns:
        return "Not enough data to interpret this chart."

    work = grouped[[x_col, y_col]].copy()
    work[y_col] = safe_numeric(work[y_col])
    work = work.dropna(subset=[x_col, y_col])

    if work.empty:
        return "Not enough data to interpret this chart."

    try:
        top_row = work.loc[work[y_col].idxmax()]
        low_row = work.loc[work[y_col].idxmin()]
    except Exception:
        return "Not enough data to interpret this chart."

    top_x = top_row[x_col]
    top_y = float(top_row[y_col])
    low_x = low_row[x_col]
    low_y = float(low_row[y_col])

    if chart_type in ["line", "trend"]:
        first_row = work.iloc[0]
        last_row = work.iloc[-1]
        first_y = float(first_row[y_col])
        last_y = float(last_row[y_col])

        if last_y > first_y:
            trend_text = "Overall, the pattern ends higher than it starts."
        elif last_y < first_y:
            trend_text = "Overall, the pattern ends lower than it starts."
        else:
            trend_text = "Overall, the pattern stays fairly level from start to end."

        return (
            f"The peak appears at '{top_x}' with {y_col} = {top_y:.2f}, "
            f"while the lowest point appears at '{low_x}' with {y_col} = {low_y:.2f}. "
            f"{trend_text}"
        )

    return (
        f"The highest value appears at '{top_x}' with {y_col} = {top_y:.2f}, "
        f"while the lowest value appears at '{low_x}' with {y_col} = {low_y:.2f}."
    )


def render_recommended_target_charts(df: pd.DataFrame, target: str):
    st.markdown("### Recommended target charts")

    if target not in df.columns:
        st.info("Focus column not found.")
        return

    target_task = infer_task_type(df[target])
    _, _, datetime_cols = split_columns(df)

    line_keywords = ["date", "time", "year", "month", "day", "price", "sales", "revenue", "amount", "cost", "profit"]

    for col in df.columns:
        if col == target:
            continue

        with st.container(border=True):
            st.markdown(f"**{target} vs {col}**")

            col_lower = str(col).lower().strip()

            # Datetime column
            if col in datetime_cols:
                work = df[[col, target]].copy()
                work[col] = pd.to_datetime(work[col], errors="coerce")

                if target_task == "regression":
                    work[target] = safe_numeric(work[target])

                work = work.dropna(subset=[col, target])

                if work.empty:
                    st.info("Not enough valid data for this chart.")
                    continue

                temp = work.copy()
                temp["period"] = temp[col].dt.to_period("M").astype(str)

                if target_task == "classification":
                    grouped = temp.groupby(["period", target]).size().reset_index(name="count")

                    fig = px.line(
                        grouped,
                        x="period",
                        y="count",
                        color=target,
                        markers=True,
                        title=f"{target} over time by {col}",
                    )
                    plot_plotly(fig)

                    totals = grouped.groupby("period", as_index=False)["count"].sum()
                    st.info(recommended_chart_interpretation(totals, "period", "count", "line"))

                else:
                    grouped = temp.groupby("period", as_index=False)[target].mean()

                    fig = px.line(
                        grouped,
                        x="period",
                        y=target,
                        markers=True,
                        title=f"{target} over time by {col}",
                    )
                    plot_plotly(fig)

                    st.info(recommended_chart_interpretation(grouped, "period", target, "line"))

            # Numeric predictor
            elif pd.api.types.is_numeric_dtype(df[col]):
                work = df[[col, target]].copy()
                work[col] = safe_numeric(work[col])

                if target_task == "regression":
                    work[target] = safe_numeric(work[target])

                work = work.dropna(subset=[col, target])

                if work.empty:
                    st.info("Not enough valid data for this chart.")
                    continue

                try:
                    unique_count = work[col].nunique(dropna=True)
                    binned_col = f"{col}_bin"

                    if unique_count > 12:
                        work[binned_col] = pd.qcut(
                            work[col],
                            q=min(8, unique_count),
                            duplicates="drop",
                        ).astype(str)
                    else:
                        work[binned_col] = work[col].round(2).astype(str)

                    work[binned_col] = work[binned_col].replace("nan", np.nan)
                    work = work.dropna(subset=[binned_col])

                    if work.empty:
                        st.info("Not enough valid grouped data.")
                        continue

                    bin_count = work[binned_col].nunique(dropna=True)

                    if target_task == "classification":
                        grouped = (
                            work.groupby([binned_col, target], dropna=False)
                            .size()
                            .reset_index(name="count")
                        )

                        if grouped.empty:
                            st.info("Not enough valid grouped data.")
                            continue

                        totals = grouped.groupby(binned_col, as_index=False)["count"].sum()

                        if bin_count <= 4:
                            fig = px.pie(
                                totals,
                                names=binned_col,
                                values="count",
                                title=f"{target} vs {col}",
                            )
                            plot_plotly(fig)
                            st.info(recommended_chart_interpretation(totals, binned_col, "count", "pie"))
                        elif bin_count <= 8:
                            fig = px.bar(
                                grouped,
                                x=binned_col,
                                y="count",
                                color=target,
                                barmode="group",
                                text="count",
                                title=f"{target} vs {col}",
                            )
                            plot_plotly(fig)
                            st.info(recommended_chart_interpretation(totals, binned_col, "count", "column"))
                        else:
                            fig = px.bar(
                                grouped,
                                x="count",
                                y=binned_col,
                                color=target,
                                barmode="group",
                                orientation="h",
                                text="count",
                                title=f"{target} vs {col}",
                            )
                            plot_plotly(fig)
                            st.info(recommended_chart_interpretation(totals, binned_col, "count", "bar"))

                    else:
                        grouped = (
                            work.groupby(binned_col, dropna=False)[target]
                            .mean()
                            .reset_index()
                        )

                        if grouped.empty:
                            st.info("Not enough valid grouped data.")
                            continue

                        if any(k in col_lower for k in line_keywords):
                            fig = px.line(
                                grouped,
                                x=binned_col,
                                y=target,
                                markers=True,
                                title=f"{target} vs {col}",
                            )
                            plot_plotly(fig)
                            st.info(recommended_chart_interpretation(grouped, binned_col, target, "line"))

                        elif bin_count <= 4:
                            fig = px.pie(
                                grouped,
                                names=binned_col,
                                values=target,
                                title=f"{target} vs {col}",
                            )
                            plot_plotly(fig)
                            st.info(recommended_chart_interpretation(grouped, binned_col, target, "pie"))

                        elif bin_count <= 8:
                            fig = px.bar(
                                grouped,
                                x=binned_col,
                                y=target,
                                text=target,
                                title=f"{target} vs {col}",
                            )
                            plot_plotly(fig)
                            st.info(recommended_chart_interpretation(grouped, binned_col, target, "bar"))

                        else:
                            grouped_sorted = grouped.sort_values(target, ascending=False)

                            fig = px.bar(
                                grouped_sorted,
                                x=target,
                                y=binned_col,
                                orientation="h",
                                text=target,
                                title=f"{target} vs {col}",
                            )
                            plot_plotly(fig)
                            st.info(recommended_chart_interpretation(grouped_sorted, binned_col, target, "bar"))

                except Exception as e:
                    st.info(f"Could not build a recommended chart for numeric column '{col}': {e}")

            # Categorical predictor
            else:
                work = df[[col, target]].copy()

                if target_task == "classification":
                    work = work.dropna(subset=[col, target])

                    if work.empty:
                        st.info("Not enough valid data for this chart.")
                        continue

                    cat_count = work[col].nunique(dropna=True)

                    grouped = work.groupby([col, target]).size().reset_index(name="count")
                    top_cats = work[col].value_counts().head(15).index
                    grouped = grouped[grouped[col].isin(top_cats)]

                    if grouped.empty:
                        st.info("Not enough valid grouped data.")
                        continue

                    totals = grouped.groupby(col, as_index=False)["count"].sum()

                    if cat_count <= 5:
                        fig = px.pie(
                            totals,
                            names=col,
                            values="count",
                            title=f"{target} vs {col}",
                        )
                        plot_plotly(fig)
                        st.info(recommended_chart_interpretation(totals, col, "count", "pie"))

                    elif cat_count <= 10:
                        fig = px.bar(
                            grouped,
                            x=col,
                            y="count",
                            color=target,
                            barmode="group",
                            text="count",
                            title=f"{target} vs {col}",
                        )
                        plot_plotly(fig)
                        st.info(recommended_chart_interpretation(totals, col, "count", "column"))

                    else:
                        fig = px.bar(
                            grouped,
                            x="count",
                            y=col,
                            color=target,
                            barmode="group",
                            orientation="h",
                            text="count",
                            title=f"{target} vs {col}",
                        )
                        plot_plotly(fig)
                        st.info(recommended_chart_interpretation(totals, col, "count", "bar"))

                else:
                    work[target] = safe_numeric(work[target])
                    work = work.dropna(subset=[col, target])

                    if work.empty:
                        st.info("Not enough valid data for this chart.")
                        continue

                    cat_count = work[col].nunique(dropna=True)

                    grouped = (
                        work.groupby(col, dropna=False)[target]
                        .mean()
                        .reset_index()
                        .sort_values(target, ascending=False)
                    )

                    top_cats = work[col].value_counts().head(15).index
                    grouped = grouped[grouped[col].isin(top_cats)]

                    if grouped.empty:
                        st.info("Not enough valid grouped data.")
                        continue

                    if cat_count <= 4:
                        fig = px.pie(
                            grouped,
                            names=col,
                            values=target,
                            title=f"{target} vs {col}",
                        )
                        plot_plotly(fig)
                        st.info(recommended_chart_interpretation(grouped, col, target, "pie"))

                    elif cat_count <= 8:
                        fig = px.bar(
                            grouped,
                            x=col,
                            y=target,
                            text=target,
                            title=f"{target} vs {col}",
                        )
                        plot_plotly(fig)
                        st.info(recommended_chart_interpretation(grouped, col, target, "bar"))

                    else:
                        fig = px.bar(
                            grouped,
                            x=target,
                            y=col,
                            orientation="h",
                            text=target,
                            title=f"{target} vs {col}",
                        )
                        plot_plotly(fig)
                        st.info(recommended_chart_interpretation(grouped, col, target, "bar"))

def show_business_visuals(df: pd.DataFrame, target: str | None = None):
    st.subheader("Office style visual summary")

    if df is None or df.empty:
        st.info("No data available for business visuals.")
        return

    numeric_cols, categorical_cols, datetime_cols = split_columns(df)

    # keep old target-based behavior when target is already available
    if target is not None and target in df.columns:
        st.caption(
            "One recommended chart is shown for each column against the selected target. "
            "Time fields use line charts. Category fields use pie, bar, or column charts depending on category count."
        )

        render_recommended_target_charts(df, target)

        st.markdown("---")
        show_custom_chart_builder(df, target)

        st.markdown("### Pivot style tables")
        pivots = build_pivot_tables(df, target)

        if not pivots:
            st.info("No pivot style tables were created for this dataset.")
        else:
            for name, pivot_df in pivots.items():
                with st.expander(f"Pivot style table: {name}", expanded=False):
                    st.dataframe(pivot_df, use_container_width=True)
        return

    # business mode without forced target
    st.caption(
        "Business Mode does not require a target. "
        "A focus column is selected automatically so the recommended target charts can still be shown."
    )

    all_cols = df.columns.tolist()

    # choose a good default focus column
    suggested_focus = None

    usable_cats = [
        c for c in categorical_cols
        if not is_identifier_like_column(df, c) and 2 <= df[c].nunique(dropna=True) <= max(20, int(0.15 * len(df)))
    ]
    usable_nums = [
        c for c in numeric_cols
        if not is_identifier_like_column(df, c)
    ]

    if usable_cats:
        suggested_focus = usable_cats[0]
    elif usable_nums:
        suggested_focus = usable_nums[0]
    elif all_cols:
        suggested_focus = all_cols[0]

    focus_options = all_cols
    default_index = focus_options.index(suggested_focus) if suggested_focus in focus_options else 0

    focus_col = st.selectbox(
        "Business focus column",
        focus_options,
        index=default_index,
        key="business_focus_column",
    )

    
    render_recommended_target_charts(df, focus_col)

    st.markdown("---")
    show_custom_chart_builder(df, focus_col)

    st.markdown("### Pivot style tables")
    pivots = build_pivot_tables(df, focus_col)

    if not pivots:
        st.info("No pivot style tables were created for this dataset.")
    else:
        for name, pivot_df in pivots.items():
            with st.expander(f"Pivot style table: {name}", expanded=False):
                st.dataframe(pivot_df, use_container_width=True)

# =========================================================
# TECHNICAL VISUAL INTERPRETATION HELPERS
# =========================================================
def boxplot_interpretation(df: pd.DataFrame, col: str) -> str:
    s = safe_numeric(df[col]).dropna()

    if s.empty:
        return "There is not enough valid data to interpret this box plot."

    mean_val = float(s.mean())
    median_val = float(s.median())
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((s < lower_bound) | (s > upper_bound)).sum())

    if abs(mean_val - median_val) < max(0.1 * abs(mean_val), 0.15):
        center_text = "The mean and median are close, so the center looks fairly balanced."
    elif mean_val > median_val:
        center_text = "The mean is above the median, which suggests some pull from higher values."
    else:
        center_text = "The mean is below the median, which suggests some pull from lower values."

    if iqr < max(abs(median_val) * 0.15, 0.25):
        spread_text = "The middle half of the data is packed fairly tightly."
    elif iqr < max(abs(median_val) * 0.35, 0.75):
        spread_text = "The middle half of the data has a moderate spread."
    else:
        spread_text = "The middle half of the data is widely spread."

    if outlier_count == 0:
        outlier_text = "No strong outlier signal is visible."
    elif outlier_count <= max(3, int(0.02 * len(s))):
        outlier_text = "A few possible outliers are present."
    else:
        outlier_text = "Several possible outliers are present."

    return (
        f"For '{col}', the median is {median_val:.2f} and the mean is {mean_val:.2f}. "
        f"{center_text} {spread_text} {outlier_text}"
    )


def violinplot_interpretation(df: pd.DataFrame, col: str) -> str:
    s = safe_numeric(df[col]).dropna()

    if s.empty:
        return "There is not enough valid data to interpret this violin plot."

    mean_val = float(s.mean())
    median_val = float(s.median())
    std_val = float(s.std()) if len(s) > 1 else 0.0
    skew_val = float(stats.skew(s, bias=False)) if len(s) > 2 else 0.0

    if abs(skew_val) < 0.5:
        shape_text = "The shape looks fairly balanced."
    elif skew_val > 0.5:
        shape_text = "The shape is pulled more toward higher values."
    else:
        shape_text = "The shape is pulled more toward lower values."

    if std_val < max(abs(mean_val) * 0.15, 0.25):
        spread_text = "The spread is fairly tight."
    elif std_val < max(abs(mean_val) * 0.35, 0.75):
        spread_text = "The spread is moderate."
    else:
        spread_text = "The spread is wide."

    if abs(mean_val - median_val) < max(0.1 * abs(mean_val), 0.15):
        center_text = "The mean and median are close."
    elif mean_val > median_val:
        center_text = "The mean is above the median."
    else:
        center_text = "The mean is below the median."

    return (
        f"For '{col}', the violin plot shows how values are concentrated across the range. "
        f"{shape_text} {spread_text} {center_text}"
    )


def scatter_interpretation(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    sub = df[[x_col, y_col]].copy()
    sub[x_col] = safe_numeric(sub[x_col])
    sub[y_col] = safe_numeric(sub[y_col])
    sub = sub.dropna()

    if len(sub) < 3:
        return "There is not enough valid data to interpret this scatter plot."

    corr = sub[x_col].corr(sub[y_col])

    if pd.isna(corr):
        return "A clear relationship could not be measured from this scatter plot."

    abs_corr = abs(corr)

    if abs_corr >= 0.7:
        strength = "strong"
    elif abs_corr >= 0.4:
        strength = "moderate"
    else:
        strength = "weak"

    if corr > 0:
        direction = "positive"
    elif corr < 0:
        direction = "negative"
    else:
        direction = "no clear"

    zx = np.abs(stats.zscore(sub[x_col], nan_policy="omit"))
    zy = np.abs(stats.zscore(sub[y_col], nan_policy="omit"))
    outlier_count = int(((zx > 3) | (zy > 3)).sum())

    if outlier_count == 0:
        outlier_text = "No strong outlier signal is visible."
    elif outlier_count <= max(3, int(0.02 * len(sub))):
        outlier_text = "A few possible outliers are present."
    else:
        outlier_text = "Several possible outliers are present."

    return (
        f"The relationship between '{x_col}' and '{y_col}' looks {direction} with {strength} strength "
        f"(correlation ≈ {corr:.2f}). {outlier_text}"
    )


def heatmap_interpretation(df: pd.DataFrame) -> str:
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return "At least two numeric variables are needed to interpret correlation."

    corr_df = numeric_df.corr(numeric_only=True)

    pairs = []
    cols = corr_df.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_df.iloc[i, j]
            if pd.notna(val):
                pairs.append((cols[i], cols[j], float(val), abs(float(val))))

    if not pairs:
        return "No usable correlation pairs were found."

    pairs = sorted(pairs, key=lambda x: x[3], reverse=True)

    top_texts = []
    for a, b, val, abs_val in pairs[:3]:
        if abs_val >= 0.7:
            strength = "strong"
        elif abs_val >= 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        direction = "positive" if val > 0 else "negative"
        top_texts.append(f"{a} and {b}: {strength} {direction} correlation ({val:.2f})")

    return "Top relationships from the heatmap: " + "; ".join(top_texts) + "."


def pairplot_interpretation(df: pd.DataFrame, cols: list[str]) -> str:
    if not cols or len(cols) < 2:
        return "At least two numeric variables are needed for pair plot interpretation."

    numeric_df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()

    if numeric_df.shape[0] < 3:
        return "There is not enough valid data to interpret this pair plot."

    corr_df = numeric_df.corr()
    pairs = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_df.loc[cols[i], cols[j]]
            if pd.notna(val):
                pairs.append((cols[i], cols[j], float(val), abs(float(val))))

    if not pairs:
        return "No clear pairwise relationships were found."

    pairs = sorted(pairs, key=lambda x: x[3], reverse=True)

    top_texts = []
    for a, b, val, abs_val in pairs[:3]:
        if abs_val >= 0.7:
            strength = "strong"
        elif abs_val >= 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        direction = "positive" if val > 0 else "negative"
        top_texts.append(f"{a} and {b} show a {strength} {direction} relationship ({val:.2f})")

    return "The pair plot suggests: " + "; ".join(top_texts) + "."


def show_technical_visuals(df: pd.DataFrame, target: str):
    st.subheader("Technical visualization")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    categorical_options = categorical_cols + ([target] if target not in categorical_cols else [])

    options = []
    if numeric_cols:
        options.extend(["Histogram + KDE", "Box Plot", "Violin Plot", "QQ Plot"])
        if len(numeric_cols) >= 2:
            options.extend(["Scatter Plot", "Correlation Heatmap", "Pair Plot"])
    if categorical_options:
        options.extend(["Count Plot", "Pie Chart"])
    if categorical_options and target in df.columns:
        options.append("Category vs Target Plot")

    if not options:
        st.info("No suitable visualizations are available for this dataset.")
        return

    viz_type = st.selectbox("Choose visualization", options, key="tech_viz_type")

    if viz_type == "Histogram + KDE":
        col = st.selectbox("Numeric column", numeric_cols, key="hist_col")
        fig, ax = plt.subplots(figsize=(9, 4.8))
        sns.histplot(df[col].dropna(), kde=True, bins=20, ax=ax)
        ax.set_title(f"Distribution of {col}")
        plot_matplotlib(fig)
        st.info(numeric_distribution_interpretation(df, col))

    elif viz_type == "Box Plot":
        col = st.selectbox("Numeric column", numeric_cols, key="box_col")
        fig, ax = plt.subplots(figsize=(9, 3.2))
        sns.boxplot(x=df[col].dropna(), ax=ax)
        ax.set_title(f"Box Plot: {col}")
        plot_matplotlib(fig)
        st.info(boxplot_interpretation(df, col))

    elif viz_type == "Violin Plot":
        col = st.selectbox("Numeric column", numeric_cols, key="violin_col")
        fig, ax = plt.subplots(figsize=(9, 3.2))
        sns.violinplot(x=df[col].dropna(), ax=ax)
        ax.set_title(f"Violin Plot: {col}")
        plot_matplotlib(fig)
        st.info(violinplot_interpretation(df, col))

    elif viz_type == "QQ Plot":
        col = st.selectbox("Numeric column", numeric_cols, key="qq_col")
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        stats.probplot(df[col].dropna(), dist="norm", plot=ax)
        ax.set_title(f"QQ Plot: {col}")
        plot_matplotlib(fig)
        st.info("If the points stay close to the line, the column behaves more like a normal distribution.")

    elif viz_type == "Scatter Plot":
        x_col = st.selectbox("X axis", numeric_cols, key="scatter_x")
        y_options = [c for c in numeric_cols if c != x_col]

        if not y_options:
            st.info("Need at least two numeric columns.")
            return

        y_col = st.selectbox("Y axis", y_options, key="scatter_y")

        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        hue_col = target if classify_target_kind(df[target]) != "numerical" else None

        if hue_col and hue_col in df.columns:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax, s=65)
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, s=65)

        ax.set_title(f"{x_col} vs {y_col}")
        plot_matplotlib(fig)
        st.info(scatter_interpretation(df, x_col, y_col))

    elif viz_type == "Correlation Heatmap":
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            st.info("Need at least two numeric columns for a correlation heatmap.")
            return

        fig, ax = plt.subplots(figsize=(10, 6.5))
        sns.heatmap(
            numeric_df.corr(numeric_only=True),
            annot=True,
            fmt=".2f",
            cmap="viridis",
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Correlation Heatmap")
        plot_matplotlib(fig)
        st.info(heatmap_interpretation(df))

    elif viz_type == "Pair Plot":
        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if numeric_df.shape[1] < 2:
            st.info("Need at least two numeric columns for a pair plot.")
            return

        cols = list(numeric_df.columns[:5])
        grid = sns.pairplot(numeric_df[cols], diag_kind="kde")
        st.pyplot(grid.figure)
        plt.close(grid.figure)
        st.info(pairplot_interpretation(df, cols))

    elif viz_type == "Count Plot":
        col = st.selectbox("Categorical column", categorical_options, key="count_col")
        counts = df[col].dropna().value_counts().reset_index()
        counts.columns = [col, "count"]

        if counts.empty:
            st.info("Not enough valid data for this plot.")
            return

        plot_plotly(px.bar(counts, x=col, y="count", color=col, title=f"Count Plot: {col}"))
        st.info(categorical_chart_insight(df, col))

    elif viz_type == "Pie Chart":
        col = st.selectbox("Categorical column", categorical_options, key="pie_col")
        plot_plotly(px.pie(df.dropna(subset=[col]), names=col, title=f"Pie Chart: {col}"))
        st.info(categorical_chart_insight(df, col))

    elif viz_type == "Category vs Target Plot":
        cat_col = st.selectbox("Category column", categorical_options, key="cat_target_col")

        if pd.api.types.is_numeric_dtype(df[target]):
            work = df[[cat_col, target]].copy()
            work[target] = safe_numeric(work[target])
            work = work.dropna(subset=[cat_col, target])

            if work.empty:
                st.info("Not enough valid data for this view.")
                return

            grouped = (
                work.groupby(cat_col, dropna=False)[target]
                .mean()
                .reset_index()
                .sort_values(target, ascending=False)
            )
            plot_plotly(px.bar(grouped, x=cat_col, y=target, color=cat_col, title=f"{cat_col} vs {target}"))
            st.info(f"This chart shows the average target value across categories in {cat_col}.")

        else:
            work = df[[cat_col, target]].dropna()

            if work.empty:
                st.info("Not enough valid data for this view.")
                return

            grouped = work.groupby([cat_col, target]).size().reset_index(name="count")
            plot_plotly(
                px.bar(
                    grouped,
                    x=cat_col,
                    y="count",
                    color=target,
                    barmode="group",
                    title=f"{cat_col} vs {target}",
                )
            )
            st.info(f"This chart shows how target groups are distributed across categories in '{cat_col}'.")
            

# =========================================================
# MODELING
# =========================================================
def build_preprocessor(X: pd.DataFrame):
    X = X.copy()

    datetime_features = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]

    for col in datetime_features:
        X[f"{col}_year"] = X[col].dt.year
        X[f"{col}_month"] = X[col].dt.month
        X[f"{col}_day"] = X[col].dt.day
        X[f"{col}_dayofweek"] = X[col].dt.dayofweek

    X = X.drop(columns=datetime_features, errors="ignore")

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )

    return preprocessor, numeric_features, categorical_features, X
    
def get_feature_names(preprocessor, numeric_features, categorical_features):
    names = list(numeric_features)

    if categorical_features:
        try:
            cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            names.extend(cat_encoder.get_feature_names_out(categorical_features).tolist())
        except Exception:
            names.extend(categorical_features)

    return names


def model_bank(task_type: str, is_large_data: bool = False):
    if task_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "Decision Tree Regressor": DecisionTreeRegressor(
                random_state=42,
                max_depth=10,
                min_samples_leaf=2,
            ),
            "Random Forest Regressor": RandomForestRegressor(
                random_state=42,
                n_estimators=80 if not is_large_data else 60,
                max_depth=12 if is_large_data else 14,
                min_samples_leaf=2,
                n_jobs=-1,
            ),
        }

        if not is_large_data:
            models["Lasso"] = Lasso(random_state=42)
            models["ElasticNet"] = ElasticNet(random_state=42)
            models["KNN Regressor"] = KNeighborsRegressor(n_neighbors=5)

            if XGBOOST_AVAILABLE:
                models["XGBoost Regressor"] = XGBRegressor(
                    random_state=42,
                    n_estimators=80,
                    max_depth=4,
                    learning_rate=0.08,
                    objective="reg:squarederror",
                    n_jobs=-1,
                    verbosity=0,
                )

        return models

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            min_samples_leaf=2,
        ),
        "Random Forest Classifier": RandomForestClassifier(
            random_state=42,
            n_estimators=80 if not is_large_data else 60,
            max_depth=12 if is_large_data else 14,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
    }

    if not is_large_data:
        models["KNN Classifier"] = KNeighborsClassifier(n_neighbors=5)
        models["SVM"] = SVC(probability=False)

        if XGBOOST_AVAILABLE:
            models["XGBoost Classifier"] = XGBClassifier(
                random_state=42,
                n_estimators=80,
                max_depth=4,
                learning_rate=0.08,
                eval_metric="mlogloss",
                n_jobs=-1,
                verbosity=0,
            )

    return models

def reduce_training_size(X: pd.DataFrame, y: pd.Series, task_type: str, max_rows: int = 120000):
    if len(X) <= max_rows:
        return X, y

    if task_type == "classification":
        sample_df = X.copy()
        sample_df["__target__"] = y.values

        sampled = (
            sample_df.groupby("__target__", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max(1, int(max_rows * len(g) / len(sample_df)))), random_state=42))
        )

        y_new = sampled["__target__"].copy()
        X_new = sampled.drop(columns="__target__")
        return X_new, y_new

    sampled_idx = X.sample(n=max_rows, random_state=42).index
    return X.loc[sampled_idx], y.loc[sampled_idx]

def train_models(df: pd.DataFrame, target: str, selected_models=None, enable_feature_selection=True):
    cleaned_df, removed_info = prepare_modeling_dataset(df, target)
    data = cleaned_df.dropna(subset=[target]).copy()

    if data.empty:
        raise ValueError("No usable rows after removing missing target.")

    if len(data) < 10:
        raise ValueError("Dataset too small for modeling.")

    X_raw = data.drop(columns=[target])
    y = data[target]

    task_type = infer_task_type(y)

    preprocessor, numeric_features, categorical_features, X = build_preprocessor(X_raw)

    if X.shape[1] == 0:
        raise ValueError("No usable predictors.")

    models = model_bank(task_type)

    if selected_models:
        models = {k: v for k, v in models.items() if k in selected_models}

    if not models:
        raise ValueError("No models selected.")

    y_clean = y.dropna()

    stratify = None
    if task_type == "classification":
        vc = y_clean.value_counts()
        if vc.min() >= 2 and len(vc) > 1:
            stratify = y_clean

    X = X.loc[y_clean.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_clean,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    if X_train.empty or X_test.empty:
        raise ValueError("Split failed.")

    preprocessor.fit(X_train)

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    if hasattr(X_train_t, "toarray") and X_train_t.shape[1] < 5000:
        X_train_t = X_train_t.toarray()
        X_test_t = X_test_t.toarray()

    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)

    rows = []
    fitted = {}

    for name, model in models.items():
        try:
            model.fit(X_train_t, y_train)
            pred = model.predict(X_test_t)

            metric_row = evaluate_model(task_type, y_test, pred)

            rows.append({
                "Model": name,
                "Recommended use": recommended_model_label(task_type, name),
                **metric_row
            })

            fitted[name] = {
                "model": model,
                "pred": pred,
                "y_test": y_test,
                "X_train": X_train_t,
                "X_test": X_test_t,
                "feature_names": feature_names,
                "task_type": task_type,
                "removed_info": removed_info,
            }

        except Exception as e:
            rows.append({
                "Model": name,
                "Recommended use": f"Failed: {str(e)}"
            })

    results_df = pd.DataFrame(rows)

    metric_cols = ["R2", "MAE", "RMSE"] if task_type == "regression" else ["Accuracy", "Precision", "Recall", "F1"]
    success_df = results_df.dropna(subset=[c for c in metric_cols if c in results_df.columns], how="all").copy()

    if success_df.empty:
        raise ValueError("All models failed.")

    if task_type == "regression":
        success_df["rank_score"] = (
            success_df["R2"].rank(ascending=False)
            + success_df["MAE"].rank(ascending=True)
            + success_df["RMSE"].rank(ascending=True)
        )
    else:
        success_df["rank_score"] = (
            success_df["F1"].rank(ascending=False)
            + success_df["Accuracy"].rank(ascending=False)
        )

    success_df = success_df.sort_values("rank_score")
    best_name = success_df.iloc[0]["Model"]

    return task_type, results_df, best_name, fitted[best_name]

# =========================================================
# FORECASTING
# =========================================================
def show_forecasting(df: pd.DataFrame):
    st.subheader("Forecasting")

    work = normalize_uploaded_data(df)
    recs = recommend_forecast_columns(work)

    if not recs["date_cols"]:
        st.info("No usable date column found for forecasting.")
        return

    if not recs["value_cols"]:
        st.info("No usable numeric value column found for forecasting.")
        return

    date_col = st.selectbox("Date column", recs["date_cols"])
    value_options = [c for c in recs["value_cols"] if c != date_col]

    if not value_options:
        st.info("No usable numeric value column found for forecasting.")
        return

    value_col = st.selectbox("Value column for forecasting", value_options)
    periods = st.slider("Forecast periods", 3, 24, 6)

    history, forecast, error = simple_forecast(work, date_col, value_col, periods)

    if error:
        st.info(error)
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history[date_col], y=history[value_col], mode="lines+markers", name="History"))
    fig.add_trace(go.Scatter(x=forecast[date_col], y=forecast[value_col], mode="lines+markers", name="Forecast"))
    fig.update_layout(title=f"Forecast for {value_col}", xaxis_title=date_col, yaxis_title=value_col)
    plot_plotly(fig)

    st.dataframe(forecast, use_container_width=True)
    st.info(forecast_interpretation(history, forecast, value_col))


# =========================================================
# EXPORTS
# =========================================================
def show_exports(df: pd.DataFrame, target: str, results_df: pd.DataFrame | None, best_model_name: str | None):
    st.subheader("Downloads and pivot builder")

    st.markdown("### Custom pivot builder")
    pivot_numeric_cols, pivot_categorical_cols, _ = detect_pivot_column_types(df)

    pivot_dimension_options = df.columns.tolist()

    pivot_index = st.multiselect(
        "Rows",
        pivot_dimension_options,
        key="pivot_rows",
    )

    pivot_columns = st.multiselect(
        "Columns",
        [c for c in pivot_dimension_options if c not in pivot_index],
        key="pivot_cols",
    )

    calc_fields = build_calculated_fields(df)
    value_options = ["__row_count__"] + list(df.columns) + list(calc_fields.keys())
    value_col = st.selectbox("Calculated field", value_options)

    if value_col == "__row_count__":
        agg_options = ["count"]
    elif value_col in calc_fields or value_col in pivot_numeric_cols:
        agg_options = ["sum", "mean", "count", "median", "min", "max", "std", "nunique"]
    elif value_col in pivot_categorical_cols:
        agg_options = ["count", "nunique"]
    else:
        agg_options = ["count"]

    aggfunc = st.selectbox("Aggregation", agg_options, key="pivot_agg")

    export_pivots = build_pivot_tables(df, target)
    custom_pivot = create_custom_pivot(df, pivot_index, pivot_columns, value_col, aggfunc)

    if custom_pivot is not None:
        st.dataframe(custom_pivot, use_container_width=True)
        export_pivots["custom_pivot"] = custom_pivot
    else:
        st.info("Choose at least one row or one column field to build a pivot table.")

    st.markdown("### Recommended pivots")
    for name, pivot_df in build_pivot_tables(df, target).items():
        with st.expander(name, expanded=False):
            st.dataframe(pivot_df, use_container_width=True)

    excel_bytes = create_excel_bytes(df, results_df, export_pivots)
    st.download_button(
        "Download Excel workbook",
        data=excel_bytes,
        file_name="mr_ready_outputs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    dataset_overview = business_summary(df, target)
    chart_interpretations = (
        ["The visuals focused on how the numeric target changes across categories and over time where possible."]
        if classify_target_kind(df[target]) == "numerical"
        else ["The visuals focused on target distribution and on which columns best separate the target groups."]
    )

    model_summary = []
    if results_df is not None and not results_df.empty:
        valid_metric_rows = results_df.dropna(how="all")
        if not valid_metric_rows.empty:
            top_row = valid_metric_rows.iloc[0]
            if "Model" in top_row:
                model_summary.append(f"Recommended model: {top_row['Model']}.")

            metrics_only = []
            for col in results_df.columns:
                if col in ["Model", "rank_score", "Recommended use"]:
                    continue
                if col in top_row.index and pd.notna(top_row[col]) and isinstance(top_row[col], (int, float, np.integer, np.floating)):
                    metrics_only.append(f"{col} = {top_row[col]:.4f}")

            if metrics_only:
                model_summary.append("Top model metrics: " + ", ".join(metrics_only))

    elif best_model_name:
        model_summary.append(f"Recommended model: {best_model_name}.")

    pdf_bytes = create_manager_pdf_report(
        title="Mr Ready Executive Report",
        dataset_overview=dataset_overview,
        chart_interpretations=chart_interpretations,
        model_summary=model_summary,
        final_recommendations=[
            "Use the business charts to explain the main pattern clearly.",
            "Use the Excel workbook for detailed reporting and pivot review.",
            "Use the PDF report for a simple manager level summary.",
        ],
    )

    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name="mr_ready_report.pdf",
        mime="application/pdf",
    )
    st.download_button(
        "Download cleaned CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv",
    )

    # =========================================================
# MAIN
# =========================================================
def main():
    st.markdown(
        """
        <div class="mr-top">
            <h2 style="margin:0;">Mr Ready</h2>
            <div style="margin-top:.35rem;">Upload a file, explore it, clean it, join it, model it if needed, forecast it, and export manager ready outputs.</div>
            <div style="margin-top:.5rem;">
                <span class="mr-pill">Business Mode</span>
                <span class="mr-pill">Technical Mode</span>
                <span class="mr-pill">Data Cleaning</span>
                <span class="mr-pill">Join Tables</span>
                <span class="mr-pill">Forecasting</span>
                <span class="mr-pill">Custom Tests</span>
                <span class="mr-pill">Pivot Builder</span>
                <span class="mr-pill">PDF Export</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.2, 1])
    with c1:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xls", "xlsx", "xlsm", "xlsb"],
        )
    with c2:
        dataset_name = st.selectbox(
            "Or choose preloaded data",
            ["None", "Iris", "Wine", "Breast Cancer", "Diabetes"],
        )

    if uploaded_file is not None:
        df = load_uploaded_file(uploaded_file)
        dataset_key = f"upload::{st.session_state.get('uploaded_file_hash', uploaded_file.name)}"
    elif dataset_name != "None":
        df = get_preloaded_dataset(dataset_name)
        df = normalize_uploaded_data(df)
        dataset_key = f"preloaded::{dataset_name}"
    else:
        df = None
        dataset_key = None

    if df is None or df.empty:
        st.info("Upload a CSV or Excel file, or choose a preloaded dataset to start.")
        st.stop()

    base_df, clean_df, plan_df = prepare_base_dataset(df)

    if clean_df is None or clean_df.empty:
        st.error("The dataset could not be prepared for analysis.")
        st.stop()

    if st.session_state.get("active_dataset_key") != dataset_key:
        st.session_state["active_dataset_key"] = dataset_key
        st.session_state["base_df"] = base_df
        st.session_state["base_clean_df"] = clean_df
        st.session_state["base_plan_df"] = plan_df

        # IMPORTANT:
        # manual cleaning must start from base_df, not clean_df,
        # otherwise high-missing columns were already imputed and will never be dropped
        st.session_state["business_clean_df"] = base_df.copy()
        st.session_state["technical_clean_df"] = base_df.copy()
        st.session_state["business_joined_df"] = base_df.copy()
        st.session_state["technical_joined_df"] = base_df.copy()

        st.session_state.pop("technical_model_config", None)
        st.session_state.pop("technical_model_result", None)
        st.session_state.pop("business_model_config", None)
        st.session_state.pop("business_model_result", None)
        st.session_state.pop("advanced_model_config", None)
        st.session_state.pop("advanced_model_result", None)
        st.session_state.pop("business_join_cache", None)
        st.session_state.pop("technical_join_cache", None)
        st.session_state.pop("feature_names", None)

    df = st.session_state["base_df"]
    clean_df = st.session_state["base_clean_df"]
    plan_df = st.session_state["base_plan_df"]

    profile = profile_data(clean_df)
    show_sidebar(profile)

    top1, top2 = st.columns([1, 2])
    with top1:
        mode = st.radio("Mode", ["Business Mode", "Technical Mode"], horizontal=True)

    with top2:
        if mode == "Technical Mode":
            target = show_target_selector(st.session_state["technical_joined_df"])
        else:
            target = None
            st.info("Business Mode does not require a target.")

    active_business_df = st.session_state.get("business_joined_df", st.session_state["business_clean_df"])
    active_technical_df = st.session_state.get("technical_joined_df", st.session_state["technical_clean_df"])

    leakage_items = leakage_check(active_technical_df, target) if target else []

    if target:
        if leakage_items:
            st.warning(f"{len(leakage_items)} possible leakage or ID warning(s) found.")
        else:
            st.success("No strong leakage warnings detected.")

        st.caption(f"Detected target type: {classify_target_kind(active_technical_df[target])}")

    results_df = None
    best_model_name = None

    # =====================================================
    # BUSINESS MODE
    # =====================================================
    if mode == "Business Mode":
        business_section = st.radio(
            "Business section",
            ["Overview", "Clean Data", "Join Tables", "Visualizations", "Forecasting", "Exports"],
            horizontal=True,
            key="business_section",
        )

        if business_section == "Overview":
            try:
                show_business_overview(df, active_business_df, plan_df)
            except Exception as e:
                st.error(f"Overview error: {e}")

        elif business_section == "Clean Data":
            try:
                cleaned_business_df = show_data_cleaning_studio(
                    st.session_state["business_clean_df"],
                    key_prefix="business",
                )
                if cleaned_business_df is not None and not cleaned_business_df.empty:
                    st.session_state["business_clean_df"] = cleaned_business_df.copy()
                    st.session_state["business_joined_df"] = cleaned_business_df.copy()
                    active_business_df = st.session_state["business_joined_df"]
            except Exception as e:
                st.error(f"Data cleaning error: {e}")

        elif business_section == "Join Tables":
            try:
                joined_business_df = show_join_tables(
                    st.session_state["business_clean_df"],
                    key_prefix="business",
                )
                if joined_business_df is not None and not joined_business_df.empty:
                    st.session_state["business_joined_df"] = joined_business_df.copy()
                    active_business_df = st.session_state["business_joined_df"]
            except Exception as e:
                st.error(f"Join error: {e}")

        elif business_section == "Visualizations":
            try:
                show_business_visuals(active_business_df, target=None)
            except Exception as e:
                st.error(f"Visualization error: {e}")

        elif business_section == "Forecasting":
            try:
                show_forecasting(st.session_state.get("business_joined_df", st.session_state["business_clean_df"]))
            except Exception as e:
                st.error(f"Forecasting error: {e}")

        elif business_section == "Exports":
            try:
                export_df = st.session_state.get("business_joined_df", st.session_state["business_clean_df"])
                fallback_target = export_df.columns[0] if len(export_df.columns) > 0 else None
                if fallback_target is None:
                    st.error("No columns available for export.")
                else:
                    show_exports(export_df, fallback_target, results_df, best_model_name)
            except Exception as e:
                st.error(f"Export error: {e}")

    # =====================================================
    # TECHNICAL MODE
    # =====================================================
    else:
        technical_section = st.radio(
            "Technical section",
            ["Overview", "Clean Data", "Join Tables", "Visualizations", "Models", "Forecasting", "Exports"],
            horizontal=True,
            key="technical_section",
        )

        if technical_section == "Overview":
            try:
                if target is not None:
                    show_overview(df, active_technical_df, plan_df, target)
                    show_insights(active_technical_df, target)
                else:
                    st.info("Select a target to generate target-based insights.")
            except Exception as e:
                st.error(f"Overview error: {e}")

        elif technical_section == "Clean Data":
            try:
                cleaned_technical_df = show_data_cleaning_studio(
                    st.session_state["technical_clean_df"],
                    key_prefix="technical",
                )
                if cleaned_technical_df is not None and not cleaned_technical_df.empty:
                    st.session_state["technical_clean_df"] = cleaned_technical_df.copy()
                    st.session_state["technical_joined_df"] = cleaned_technical_df.copy()
                    active_technical_df = st.session_state["technical_joined_df"]
            except Exception as e:
                st.error(f"Data cleaning error: {e}")

        elif technical_section == "Join Tables":
            try:
                joined_technical_df = show_join_tables(
                    st.session_state["technical_clean_df"],
                    key_prefix="technical",
                )
                if joined_technical_df is not None and not joined_technical_df.empty:
                    st.session_state["technical_joined_df"] = joined_technical_df.copy()
                    active_technical_df = st.session_state["technical_joined_df"]
            except Exception as e:
                st.error(f"Join error: {e}")

        elif technical_section == "Visualizations":
            try:
                if target is None:
                    st.info("Select a target first to use technical visualizations.")
                else:
                    show_technical_visuals(st.session_state["technical_joined_df"], target)
            except Exception as e:
                st.error(f"Visualization error: {e}")

        elif technical_section == "Models":
            try:
                if target is None:
                    st.info("Select a target first to use modeling.")
                    results_df, best_model_name = None, None
                else:
                    results_df, best_model_name = show_modeling(
                        st.session_state["technical_joined_df"],
                        target,
                        business_mode=False,
                    )
            except Exception as e:
                st.error(f"Modeling error: {e}")
                results_df, best_model_name = None, None

        elif technical_section == "Forecasting":
            try:
                show_forecasting(st.session_state["technical_joined_df"])
            except Exception as e:
                st.error(f"Forecasting error: {e}")

        elif technical_section == "Exports":
            try:
                export_df = st.session_state["technical_joined_df"]

                if target is None:
                    fallback_target = export_df.columns[0] if len(export_df.columns) > 0 else None
                    if fallback_target is None:
                        st.error("No columns available for export.")
                    else:
                        show_exports(export_df, fallback_target, results_df, best_model_name)
                else:
                    show_exports(export_df, target, results_df, best_model_name)
            except Exception as e:
                st.error(f"Export error: {e}")


if __name__ == "__main__":
    main()