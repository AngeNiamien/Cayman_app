from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st
import streamlit.components.v1 as components
import statsmodels.api as sm
from scipy.stats import beta, binom
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import jenkspy
except Exception:  # pragma: no cover - deployment fallback
    jenkspy = None

try:
    from weasyprint import HTML
except Exception:  # pragma: no cover - deployment fallback
    HTML = None


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Cayman Scoring PD",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


WORKFLOW = [
    "Accueil",
    "Manquants",
    "Graphiques",
    "Discriminant",
    "Modèles",
    "Performance",
    "Segmentation",
    "Rapport",
]


@dataclass
class ModelArtifacts:
    pipeline: Pipeline
    model_type: str
    selected_features: list[str]
    encoded_feature_names: list[str]
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_train_proba: np.ndarray
    y_test_proba: np.ndarray
    metrics: dict[str, float]
    coefficients: pd.DataFrame


def init_state() -> None:
    defaults: dict[str, Any] = {
        "df": None,
        "clean_df": None,
        "target": None,
        "id_col": None,
        "date_col": None,
        "excluded_cols": [],
        "missing_log": [],
        "discriminant": None,
        "model": None,
        "segmentation": None,
        "report_html": None,
        "report_pdf": None,
        "benchmark_results": None,
        "cv_results": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
        [data-testid="stSidebar"] img {margin-bottom: 0.75rem;}
        .section-title {
            margin: 0.75rem 0 1rem 0;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid #e6eef5;
            color: #1f2937;
        }
        .status-ok {color: #15803d; font-weight: 700;}
        .status-warn {color: #b45309; font-weight: 700;}
        .small-note {color: #64748b; font-size: .9rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def brand_bar() -> None:
    logo = APP_DIR / "logo1.png"
    logo_html = ""
    if logo.exists():
        import base64

        encoded = base64.b64encode(logo.read_bytes()).decode("ascii")
        logo_html = f'<img src="data:image/png;base64,{encoded}" alt="Cayman Consulting">'
    st.markdown(
        f"""
        <div class="brand-bar">
            <div class="brand-left">
                {logo_html}
                <div>
                    <div class="brand-title">Cayman Scoring PD</div>
                    <div class="brand-subtitle">Plateforme de développement, validation et reporting de modèles de crédit</div>
                </div>
            </div>
            <div class="brand-pill">Credit Risk Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def section(title: str) -> None:
    st.markdown(f"<h3 class='section-title'>{title}</h3>", unsafe_allow_html=True)


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude=np.number).columns.tolist()


def usable_features(df: pd.DataFrame, target: str | None) -> list[str]:
    if df is None or not target:
        return []
    excluded = set(st.session_state.excluded_cols or [])
    excluded.update({target})
    if st.session_state.id_col:
        excluded.add(st.session_state.id_col)
    if st.session_state.date_col:
        excluded.add(st.session_state.date_col)
    return [c for c in df.columns if c not in excluded]


def target_series(df: pd.DataFrame, target: str) -> pd.Series:
    y = df[target].copy()
    if y.dtype.name == "category" or y.dtype == object:
        values = list(pd.Series(y.dropna().unique()).sort_values())
        mapping = {values[0]: 0, values[-1]: 1} if len(values) == 2 else {}
        return y.map(mapping).astype("Int64")
    return y.astype("Int64")


def validate_target(df: pd.DataFrame, target: str | None) -> tuple[bool, str]:
    if df is None or not target:
        return False, "Sélectionnez une variable cible."
    values = df[target].dropna().unique()
    if len(values) != 2:
        return False, "La variable cible doit être binaire pour un modèle de PD."
    y = target_series(df, target)
    if y.isna().any():
        return False, "La cible contient des valeurs non convertibles en 0/1."
    if y.nunique() != 2:
        return False, "La cible doit contenir deux classes après conversion."
    return True, "Cible valide."


def read_uploaded_file(file, file_type: str, delimiter: str | None) -> pd.DataFrame:
    if file_type in {"CSV", "TXT"}:
        return pd.read_csv(file, sep=delimiter or None, engine="python")
    return pd.read_excel(file, engine="openpyxl")


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    out = pd.DataFrame(
        {
            "Variable": miss.index,
            "Manquants": miss.values,
            "Taux": miss.values / max(len(df), 1),
            "Type": [str(df[c].dtype) for c in miss.index],
            "Modalités": [df[c].nunique(dropna=True) for c in miss.index],
        }
    )
    return out.sort_values(["Manquants", "Variable"], ascending=[False, True])


def quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    nunique = df.nunique(dropna=False)
    return {
        "duplicates": int(df.duplicated().sum()),
        "empty_cols": nunique[nunique <= 1].index.tolist(),
        "high_missing": missing_summary(df).query("Taux >= 0.5")["Variable"].tolist(),
    }


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x.fillna("__MISSING__"), y.fillna("__MISSING__"))
    if table.empty or min(table.shape) <= 1:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min((kcorr - 1), (rcorr - 1))
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def bin_series(s: pd.Series, max_bins: int = 5) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > max_bins:
        try:
            return pd.qcut(s, q=max_bins, duplicates="drop").astype(str)
        except ValueError:
            return pd.cut(s, bins=max_bins, duplicates="drop").astype(str)
    return s.astype(str)


def information_value(feature: pd.Series, y: pd.Series, bins: int = 5) -> tuple[float, pd.DataFrame]:
    binned = bin_series(feature, bins)
    tmp = pd.DataFrame({"bin": binned.fillna("__MISSING__"), "target": y})
    grouped = tmp.groupby("bin", dropna=False)["target"].agg(["count", "sum"]).rename(columns={"sum": "bad"})
    grouped["good"] = grouped["count"] - grouped["bad"]
    total_good = max(grouped["good"].sum(), 1)
    total_bad = max(grouped["bad"].sum(), 1)
    grouped["dist_good"] = grouped["good"] / total_good
    grouped["dist_bad"] = grouped["bad"] / total_bad
    grouped["woe"] = np.log((grouped["dist_good"] + 1e-6) / (grouped["dist_bad"] + 1e-6))
    grouped["iv"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]
    grouped["bad_rate"] = grouped["bad"] / grouped["count"].replace(0, np.nan)
    return float(grouped["iv"].sum()), grouped.reset_index()


def univariate_auc(feature: pd.Series, y: pd.Series) -> float:
    try:
        if not pd.api.types.is_numeric_dtype(feature):
            codes = pd.factorize(feature.fillna("__MISSING__"))[0]
            values = pd.Series(codes, index=feature.index)
        else:
            values = feature.fillna(feature.median())
        score = roc_auc_score(y, values)
        return float(max(score, 1 - score))
    except Exception:
        return np.nan


def prepare_model(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    selected_raw: list[str],
    test_size: float,
    model_type: str = "Logistique",
    date_col: str | None = None,
    oot_cutoff=None,
) -> ModelArtifacts:
    y = target_series(df, target).astype(int)
    X = df[features].copy()
    X_selected = X[selected_raw].copy()

    num = X_selected.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X_selected.columns if c not in num]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if date_col and oot_cutoff is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        cutoff = pd.Timestamp(oot_cutoff)
        train_mask = dates < cutoff
        X_train = X_selected[train_mask]
        X_test = X_selected[~train_mask]
        y_train = y[train_mask]
        y_test = y[~train_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=42, stratify=y
        )

    if model_type == "Random Forest":
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    elif model_type == "Gradient Boosting":
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    else:
        clf = LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced")

    pipeline = Pipeline([("preprocess", preprocessor), ("model", clf)])
    pipeline.fit(X_train, y_train)

    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    metrics = {
        "auc_train": float(train_auc),
        "auc_test": float(test_auc),
        "gini_train": float(2 * train_auc - 1),
        "gini_test": float(2 * test_auc - 1),
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
    }

    encoded_names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    if model_type == "Logistique":
        raw_vals = pipeline.named_steps["model"].coef_[0]
        importances = np.abs(raw_vals)
    else:
        raw_vals = pipeline.named_steps["model"].feature_importances_
        importances = raw_vals
    total = importances.sum()
    coefficients = pd.DataFrame({
        "Variable encodée": encoded_names,
        "Coefficient": raw_vals,
        "Poids absolu": importances,
        "Poids (%)": np.where(total > 0, importances / total * 100, 0),
    }).sort_values("Poids (%)", ascending=False)

    return ModelArtifacts(
        pipeline=pipeline,
        model_type=model_type,
        selected_features=selected_raw,
        encoded_feature_names=encoded_names,
        X_train_raw=X_train,
        X_test_raw=X_test,
        y_train=y_train,
        y_test=y_test,
        y_train_proba=y_train_proba,
        y_test_proba=y_test_proba,
        metrics=metrics,
        coefficients=coefficients,
    )


def benchmark_models(df: pd.DataFrame, target: str, features: list[str], selected_raw: list[str], test_size: float) -> pd.DataFrame:
    rows = []
    for mtype in ["Logistique", "Random Forest", "Gradient Boosting"]:
        try:
            m = prepare_model(df, target, features, selected_raw, test_size, model_type=mtype)
            rows.append({
                "Modèle": mtype,
                "AUC train": round(m.metrics["auc_train"], 4),
                "AUC test": round(m.metrics["auc_test"], 4),
                "Gini test": round(m.metrics["gini_test"], 4),
                "Recall": round(m.metrics["recall"], 4),
                "F1": round(m.metrics["f1"], 4),
            })
        except Exception as exc:
            rows.append({"Modèle": mtype, "AUC train": None, "AUC test": None, "Gini test": None, "Recall": None, "F1": None, "Erreur": str(exc)})
    return pd.DataFrame(rows)


def l1_candidate_features(df: pd.DataFrame, target: str, features: list[str]) -> list[str]:
    y = target_series(df, target).astype(int)
    X = df[features].copy()
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ],
        verbose_feature_names_out=False,
    )
    selector = Pipeline(
        [
            ("preprocess", preprocessor),
            ("select", SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=2000))),
        ]
    )
    selector.fit(X, y)
    encoded = selector.named_steps["preprocess"].get_feature_names_out()
    selected_encoded = encoded[selector.named_steps["select"].get_support()].tolist()
    raw = []
    for feat in features:
        if feat in selected_encoded or any(name.startswith(f"{feat}_") for name in selected_encoded):
            raw.append(feat)
    return raw or features[: min(len(features), 10)]


def _stepwise_design_matrix(df: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    X = df[selected].copy()
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].astype("object").fillna("Missing")
    X = pd.get_dummies(X, columns=[c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])], drop_first=True, dtype=float)
    X = X.loc[:, X.nunique(dropna=False) > 1]
    return sm.add_constant(X, has_constant="add")


def stepwise_logistic_features(
    df: pd.DataFrame,
    target: str,
    candidates: list[str],
    direction: str = "both",
    max_features: int = 12,
) -> tuple[list[str], pd.DataFrame]:
    """Select raw variables using logistic-regression AIC.

    Categorical variables are evaluated as grouped dummy variables, so the
    returned list remains readable for the final model and report.
    """
    y = target_series(df, target).astype(int)
    candidates = [c for c in candidates if c != target and df[c].nunique(dropna=True) > 1]
    selected: list[str] = []
    current_aic = np.inf
    history = []

    def fit_aic(cols: list[str]) -> float:
        if not cols:
            X0 = pd.DataFrame({"const": np.ones(len(df))}, index=df.index)
        else:
            X0 = _stepwise_design_matrix(df, cols)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = sm.Logit(y, X0).fit(disp=False, maxiter=200)
            return float(result.aic)
        except Exception:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = sm.GLM(y, X0, family=sm.families.Binomial()).fit()
                return float(result.aic)
            except Exception:
                return np.inf

    improved = True
    while improved and len(selected) < max_features:
        improved = False
        remaining = [c for c in candidates if c not in selected]
        add_scores = []
        for col in remaining:
            aic_value = fit_aic(selected + [col])
            add_scores.append((aic_value, "add", col))
        if add_scores:
            best_aic, action, best_col = min(add_scores, key=lambda x: x[0])
            if best_aic + 1e-6 < current_aic:
                selected.append(best_col)
                current_aic = best_aic
                improved = True
                history.append({"Étape": len(history) + 1, "Action": action, "Variable": best_col, "AIC": current_aic})

        if direction == "both" and len(selected) > 1:
            remove_scores = []
            for col in selected:
                trial = [c for c in selected if c != col]
                aic_value = fit_aic(trial)
                remove_scores.append((aic_value, "remove", col, trial))
            best_aic, action, best_col, trial = min(remove_scores, key=lambda x: x[0])
            if best_aic + 1e-6 < current_aic:
                selected = trial
                current_aic = best_aic
                improved = True
                history.append({"Étape": len(history) + 1, "Action": action, "Variable": best_col, "AIC": current_aic})

    return selected, pd.DataFrame(history)


def vif_table(model: ModelArtifacts) -> pd.DataFrame:
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        X_enc = model.pipeline.named_steps["preprocess"].transform(model.X_train_raw)
        names = model.encoded_feature_names
        if X_enc.shape[1] > 80:
            return pd.DataFrame({"Information": ["VIF ignoré: trop de variables encodées (>80)."]})
        data = sm.add_constant(pd.DataFrame(X_enc, columns=names))
        rows = []
        for i, col in enumerate(data.columns):
            if col == "const":
                continue
            rows.append({"Variable": col, "VIF": variance_inflation_factor(data.values, i)})
        return pd.DataFrame(rows).sort_values("VIF", ascending=False)
    except Exception as exc:
        return pd.DataFrame({"Information": [f"VIF indisponible: {exc}"]})


def psi_score(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    ref_pct = np.histogram(reference, bins=breakpoints)[0] / max(len(reference), 1)
    cur_pct = np.histogram(current, bins=breakpoints)[0] / max(len(current), 1)
    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-4, cur_pct)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def csi_table(model: ModelArtifacts, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for col in model.selected_features:
        if col not in model.X_train_raw.columns or col not in model.X_test_raw.columns:
            continue
        if not pd.api.types.is_numeric_dtype(model.X_train_raw[col]):
            continue
        ref = model.X_train_raw[col].dropna().values
        cur = model.X_test_raw[col].dropna().values
        if len(ref) < n_bins or len(cur) < n_bins:
            continue
        psi = psi_score(ref, cur, n_bins)
        statut = "Stable" if psi < 0.10 else ("Attention" if psi < 0.25 else "Instable")
        rows.append({"Variable": col, "PSI (CSI)": round(psi, 4), "Statut": statut})
    return pd.DataFrame(rows).sort_values("PSI (CSI)", ascending=False) if rows else pd.DataFrame(columns=["Variable", "PSI (CSI)", "Statut"])


def cap_outliers(df: pd.DataFrame, cols: list[str], method: str, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        if method == "IQR (1.5x)":
            q1, q3 = out[col].quantile(0.25), out[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:
            lo = out[col].quantile(lower_pct)
            hi = out[col].quantile(upper_pct)
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def score_dataset(df: pd.DataFrame, model: ModelArtifacts, n_classes: int, method: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    data = df.copy()
    proba = model.pipeline.predict_proba(data[model.selected_features])[:, 1]
    data["PD"] = proba
    score = -np.log(np.clip(proba, 1e-8, 1 - 1e-8))
    data["Score"] = 1000 * (score - score.min()) / max(score.max() - score.min(), 1e-8)

    if method == "Jenks" and jenkspy is not None and data["Score"].nunique() >= n_classes:
        breaks = jenkspy.jenks_breaks(data["Score"].values, n_classes)
        labels = np.digitize(data["Score"], breaks[1:-1], right=True) + 1
    else:
        labels = pd.qcut(data["Score"], q=n_classes, labels=False, duplicates="drop").astype(int) + 1

    max_label = int(np.max(labels))
    data["Classe de risque"] = [f"CR {max_label - int(x) + 1}" for x in labels]
    y = target_series(data, target).astype(int)

    stats_df = data.groupby("Classe de risque", sort=True).agg(
        Nombre=("PD", "size"),
        Pourcentage=("PD", lambda s: len(s) / len(data)),
        Score_min=("Score", "min"),
        Score_max=("Score", "max"),
        PD_moyenne=("PD", "mean"),
    )
    stats_df["Défauts"] = data.assign(_y=y).groupby("Classe de risque")["_y"].sum()
    stats_df["Taux_défaut"] = stats_df["Défauts"] / stats_df["Nombre"].replace(0, np.nan)
    hhi = float(((stats_df["Nombre"] / len(data)) ** 2).sum() * 100)
    auc_value = float(roc_auc_score(y, proba))
    metrics = {"auc": auc_value, "gini": 2 * auc_value - 1, "hhi": hhi}
    return data, stats_df.reset_index(), metrics


def calibrate_stats(stats_df: pd.DataFrame, method: str, confidence: float) -> pd.DataFrame:
    out = stats_df.copy()
    if method == "Beta":
        out["PD_calibrée"] = out.apply(
            lambda r: beta.ppf(confidence, r["Défauts"] + 0.5, r["Nombre"] - r["Défauts"] + 0.5), axis=1
        )
    else:
        out["PD_calibrée"] = out.apply(
            lambda r: binom.ppf(confidence, max(int(r["Nombre"]), 1), r["Taux_défaut"]) / max(int(r["Nombre"]), 1),
            axis=1,
        )
    return out


def homogeneity_tests(scored: pd.DataFrame, split_var: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cls, group in scored.groupby("Classe de risque"):
        if split_var not in group or not pd.api.types.is_numeric_dtype(group[split_var]) or group[split_var].nunique() < 2:
            continue
        med = group[split_var].median()
        a = group.loc[group[split_var] < med, "Score"]
        b = group.loc[group[split_var] >= med, "Score"]
        if len(a) > 1 and len(b) > 1:
            t_stat, p_t = stats.ttest_ind(a, b, equal_var=False)
            w_stat, p_w = stats.ranksums(a, b)
            rows.append({"Classe": cls, "t_stat": t_stat, "p_value_t": p_t, "w_stat": w_stat, "p_value_w": p_w})
    classes = dict(tuple(scored.groupby("Classe de risque")))
    inter = []
    keys = sorted(classes)
    for i, c1 in enumerate(keys):
        for c2 in keys[i + 1 :]:
            a, b = classes[c1]["Score"], classes[c2]["Score"]
            if len(a) > 1 and len(b) > 1:
                t_stat, p_t = stats.ttest_ind(a, b, equal_var=False)
                inter.append({"Classe 1": c1, "Classe 2": c2, "t_stat": t_stat, "p_value": p_t})
    return pd.DataFrame(rows), pd.DataFrame(inter)


def page_sidebar() -> str:
    logo = APP_DIR.parent / "logo1.png"
    if not logo.exists():
        logo = APP_DIR.parent / "logo.png"
    if logo.exists():
        st.sidebar.image(str(logo), width=160)
    st.sidebar.header("Cayman-App `version 1`")
    st.sidebar.header("Pages")
    page = st.sidebar.radio("Workflow", WORKFLOW, label_visibility="collapsed")
    st.sidebar.divider()
    df_ok = st.session_state.df is not None
    target_ok, _ = validate_target(st.session_state.clean_df if st.session_state.clean_df is not None else st.session_state.df, st.session_state.target)
    df_status = '<span class="status-ok">OK</span>' if df_ok else '<span class="status-warn">à charger</span>'
    target_status = '<span class="status-ok">OK</span>' if target_ok else '<span class="status-warn">à valider</span>'
    model_status = '<span class="status-ok">OK</span>' if st.session_state.model else '<span class="status-warn">à construire</span>'
    seg_status = '<span class="status-ok">OK</span>' if st.session_state.segmentation else '<span class="status-warn">à produire</span>'
    st.sidebar.markdown(f"Données: {df_status}", unsafe_allow_html=True)
    st.sidebar.markdown(f"Cible: {target_status}", unsafe_allow_html=True)
    st.sidebar.markdown(f"Modèle: {model_status}", unsafe_allow_html=True)
    st.sidebar.markdown(f"Segmentation: {seg_status}", unsafe_allow_html=True)
    return page


def require_data() -> pd.DataFrame | None:
    df = st.session_state.clean_df if st.session_state.clean_df is not None else st.session_state.df
    if df is None:
        st.warning("Chargez d'abord une base dans la page Accueil.")
        return None
    return df


def page_accueil() -> None:
    st.title("APPLICATION DE DESCRIPTION DE DONNEES - CAYMAN CONSULTING")
    header("🏚️ Accueil", "Afficher un échantillon du jeu de données et cadrer le périmètre de modélisation.")
    left, right = st.columns([0.62, 0.38], gap="large")
    with left:
        section("Importer les données")
        file_type = st.selectbox("Type de fichier", ["CSV", "TXT", "XLSX"])
        delimiter = None
        if file_type in {"CSV", "TXT"}:
            delimiter = st.selectbox("Délimiteur", [",", ";", "|", "\t"], index=1)
        file = st.file_uploader("Fichier de modélisation", type=[file_type.lower()])
        if file is not None:
            try:
                df = read_uploaded_file(file, file_type, delimiter)
                st.session_state.df = df
                st.session_state.clean_df = df.copy()
                st.session_state.model = None
                st.session_state.segmentation = None
                st.success("Base chargée avec succès.")
            except Exception as exc:
                st.error(f"Import impossible: {exc}")

    df = st.session_state.df
    with right:
        section("Paramètres métier")
        if df is not None:
            cols = df.columns.tolist()
            st.session_state.target = st.selectbox("Variable cible défaut / non défaut", cols, index=cols.index(st.session_state.target) if st.session_state.target in cols else 0)
            optional = ["Aucune"] + cols
            id_value = st.selectbox("Identifiant client", optional, index=optional.index(st.session_state.id_col) if st.session_state.id_col in optional else 0)
            date_value = st.selectbox("Date d'observation", optional, index=optional.index(st.session_state.date_col) if st.session_state.date_col in optional else 0)
            st.session_state.id_col = None if id_value == "Aucune" else id_value
            st.session_state.date_col = None if date_value == "Aucune" else date_value
            st.session_state.excluded_cols = st.multiselect("Variables à exclure du modèle", cols, default=st.session_state.excluded_cols)
        else:
            st.info("Les paramètres apparaîtront après import.")

    if df is not None:
        ok, msg = validate_target(df, st.session_state.target)
        if ok:
            y = target_series(df, st.session_state.target).astype(int)
            default_rate = y.mean()
            st.success(msg)
        else:
            default_rate = np.nan
            st.warning(msg)

        section("Vue d'ensemble")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Observations", f"{len(df):,}".replace(",", " "))
        m2.metric("Variables", df.shape[1])
        m3.metric("Numériques", len(numeric_cols(df)))
        m4.metric("Catégorielles", len(categorical_cols(df)))
        m5.metric("Taux de défaut", "" if np.isnan(default_rate) else f"{default_rate:.2%}")
        st.dataframe(df.head(50), use_container_width=True)

        q = quality_summary(df)
        c1, c2, c3 = st.columns(3)
        c1.info(f"Doublons détectés: {q['duplicates']}")
        c2.info(f"Colonnes constantes: {len(q['empty_cols'])}")
        c3.info(f"Colonnes avec >= 50% manquants: {len(q['high_missing'])}")


def page_manquants() -> None:
    header("🧾 Manquants", "Valeurs manquantes des variables du jeu de données, en nombre et pourcentage.")
    df = require_data()
    if df is None:
        return

    summary = missing_summary(df)
    section("Diagnostic des valeurs manquantes")
    c1, c2 = st.columns([0.55, 0.45], gap="large")
    with c1:
        st.dataframe(summary, use_container_width=True)
    with c2:
        fig = px.bar(summary.head(30), x="Variable", y="Taux", color="Taux", title="Taux de manquants par variable")
        fig.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-45, height=430)
        st.plotly_chart(fig, use_container_width=True)

    section("Traitements")
    target = st.session_state.target
    candidates = [c for c in df.columns if c != target and df[c].isna().any()]
    selected = st.multiselect("Variables à traiter", candidates, default=candidates[: min(8, len(candidates))])
    method = st.selectbox(
        "Méthode d'imputation",
        ["Médiane numériques + mode catégorielles", "Moyenne numériques + mode catégorielles", "Catégorie Missing pour qualitatives", "Supprimer lignes avec manquants sélectionnés"],
    )
    if st.button("Appliquer le traitement", type="primary", disabled=not selected):
        cleaned = df.copy()
        if method == "Supprimer lignes avec manquants sélectionnés":
            before = len(cleaned)
            cleaned = cleaned.dropna(subset=selected)
            log = f"Suppression de {before - len(cleaned)} lignes sur {selected}."
        else:
            for col in selected:
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    value = cleaned[col].mean() if method.startswith("Moyenne") else cleaned[col].median()
                    cleaned[col] = cleaned[col].fillna(value)
                else:
                    value = "Missing" if method.startswith("Catégorie") else cleaned[col].mode(dropna=True)
                    cleaned[col] = cleaned[col].fillna(value.iloc[0] if hasattr(value, "iloc") and not value.empty else "Missing")
            log = f"Imputation '{method}' appliquée sur {len(selected)} variable(s)."
        st.session_state.clean_df = cleaned
        st.session_state.missing_log.append(log)
        st.session_state.model = None
        st.session_state.segmentation = None
        st.success(log)

    section("Traitement des valeurs aberrantes (winsorisation)")
    num_cols_df = st.session_state.clean_df if st.session_state.clean_df is not None else df
    num_cols_list = [c for c in numeric_cols(num_cols_df) if c != st.session_state.target]
    selected_cap = st.multiselect("Variables numériques à écrêter", num_cols_list, default=[])
    if selected_cap:
        cap_method = st.selectbox("Méthode d'écrêtement", ["Percentiles", "IQR (1.5x)"])
        lower_pct, upper_pct = 0.01, 0.99
        if cap_method == "Percentiles":
            col_lo, col_hi = st.columns(2)
            lower_pct = col_lo.slider("Percentile bas (%)", 0.5, 5.0, 1.0, 0.5) / 100
            upper_pct = col_hi.slider("Percentile haut (%)", 95.0, 99.5, 99.0, 0.5) / 100
        if st.button("Appliquer l'écrêtement", type="secondary"):
            src = st.session_state.clean_df if st.session_state.clean_df is not None else df
            capped = cap_outliers(src, selected_cap, cap_method, lower_pct, upper_pct)
            st.session_state.clean_df = capped
            st.session_state.model = None
            st.session_state.segmentation = None
            log_msg = f"Écrêtement '{cap_method}' appliqué sur {len(selected_cap)} variable(s): {', '.join(selected_cap)}."
            st.session_state.missing_log.append(log_msg)
            st.success(log_msg)

    if st.session_state.missing_log:
        section("Journal des traitements")
        for item in st.session_state.missing_log:
            st.write(f"- {item}")


def page_graphiques() -> None:
    header("Analyse univariée et bivariée", "Visualisation avancée des variables, choix des graphiques et comparaison avec la cible.")
    df = require_data()
    if df is None:
        return
    ok, msg = validate_target(df, st.session_state.target)
    if not ok:
        st.warning(msg)
        return

    target = st.session_state.target
    y = target_series(df, target).astype(int)
    features = usable_features(df, target)
    if not features:
        st.warning("Aucune variable explicative disponible.")
        return

    num_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]

    with st.expander("Options d'analyse", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        var = c1.selectbox("Variable principale", features)
        bins = c2.slider("Classes / bins", 3, 30, 10)
        top_n = c3.slider("Top modalités", 5, 40, 20)
        sample_size = c4.number_input(
            "Échantillon max",
            min_value=500,
            max_value=100000,
            value=min(len(df), 20000),
            step=500,
        )

    plot_df = df.sample(int(sample_size), random_state=42) if len(df) > sample_size else df.copy()
    plot_y = target_series(plot_df, target).astype(int)
    target_label = y.map({0: "Non défaut", 1: "Défaut"})
    plot_target_label = plot_y.map({0: "Non défaut", 1: "Défaut"})

    tab_uni, tab_target, tab_cross, tab_woe = st.tabs([
        "Analyse univariée",
        "Bivarié avec la cible",
        "Croisement variables",
        "WOE / IV",
    ])

    with tab_uni:
        section("Analyse univariée")
        is_num = pd.api.types.is_numeric_dtype(df[var])
        if is_num:
            graph_type = st.selectbox(
                "Type de graphique",
                ["Histogramme", "Histogramme + boxplot", "Boxplot", "Violin plot", "Densité", "ECDF", "Résumé statistique"],
            )
            if graph_type == "Histogramme":
                fig = px.histogram(plot_df, x=var, nbins=bins, title=f"Histogramme - {var}")
                st.plotly_chart(fig, use_container_width=True)
            elif graph_type == "Histogramme + boxplot":
                fig = px.histogram(plot_df, x=var, nbins=bins, marginal="box", title=f"Distribution - {var}")
                st.plotly_chart(fig, use_container_width=True)
            elif graph_type == "Boxplot":
                fig = px.box(plot_df, y=var, points="outliers", title=f"Boxplot - {var}")
                st.plotly_chart(fig, use_container_width=True)
            elif graph_type == "Violin plot":
                fig = px.violin(plot_df, y=var, box=True, points=False, title=f"Violin plot - {var}")
                st.plotly_chart(fig, use_container_width=True)
            elif graph_type == "Densité":
                fig = px.histogram(plot_df, x=var, nbins=bins, histnorm="probability density", title=f"Densité - {var}")
                st.plotly_chart(fig, use_container_width=True)
            elif graph_type == "ECDF":
                fig = px.ecdf(plot_df, x=var, title=f"Distribution cumulée - {var}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                desc = df[var].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_frame("Valeur")
                desc.loc["manquants"] = df[var].isna().sum()
                desc.loc["taux_manquants"] = df[var].isna().mean()
                st.dataframe(desc, use_container_width=True)
        else:
            graph_type = st.selectbox(
                "Type de graphique",
                ["Barres - effectifs", "Barres - pourcentages", "Treemap", "Donut", "Table des fréquences"],
            )
            freq = df[var].fillna("Missing").value_counts(dropna=False).head(top_n).reset_index()
            freq.columns = [var, "Effectif"]
            freq["Pourcentage"] = freq["Effectif"] / len(df)
            if graph_type == "Barres - effectifs":
                fig = px.bar(freq, x=var, y="Effectif", title=f"Effectifs - {var}")
                fig.update_layout(xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)
            elif graph_type == "Barres - pourcentages":
                fig = px.bar(freq, x=var, y="Pourcentage", title=f"Pourcentages - {var}")
                fig.update_layout(yaxis_tickformat=".1%", xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)
            elif graph_type == "Treemap":
                fig = px.treemap(freq, path=[var], values="Effectif", title=f"Treemap - {var}")
                st.plotly_chart(fig, use_container_width=True)
            elif graph_type == "Donut":
                fig = px.pie(freq, names=var, values="Effectif", hole=0.45, title=f"Répartition - {var}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                display = freq.copy()
                display["Pourcentage"] = display["Pourcentage"].map(lambda x: f"{x:.2%}")
                st.dataframe(display, use_container_width=True)

    with tab_target:
        section("Analyse bivariée avec la cible")
        is_num = pd.api.types.is_numeric_dtype(df[var])
        if is_num:
            graph_type = st.selectbox(
                "Type de graphique cible",
                ["Histogramme superposé", "Boxplot par cible", "Violin par cible", "Taux de défaut par bins", "Densité par cible"],
            )
            tmp = plot_df[[var]].copy()
            tmp["Cible"] = plot_target_label
            if graph_type == "Histogramme superposé":
                fig = px.histogram(tmp, x=var, color="Cible", nbins=bins, barmode="overlay", opacity=0.65, title=f"{var} par cible")
                st.plotly_chart(fig, use_container_width=True)
            elif graph_type == "Boxplot par cible":
                fig = px.box(tmp, x="Cible", y=var, color="Cible", points="outliers", title=f"{var} par cible")
                st.plotly_chart(fig, use_container_width=True)
            elif graph_type == "Violin par cible":
                fig = px.violin(tmp, x="Cible", y=var, color="Cible", box=True, title=f"{var} par cible")
                st.plotly_chart(fig, use_container_width=True)
            elif graph_type == "Densité par cible":
                fig = px.histogram(tmp, x=var, color="Cible", nbins=bins, histnorm="probability density", barmode="overlay", opacity=0.55, title=f"Densité de {var} par cible")
                st.plotly_chart(fig, use_container_width=True)
            else:
                binned = bin_series(df[var], bins)
                rate = pd.DataFrame({"Classe": binned, "target": y}).groupby("Classe")["target"].agg(["count", "mean"]).reset_index()
                fig = px.bar(rate, x="Classe", y="mean", color="count", title=f"Taux de défaut par classes - {var}")
                fig.update_layout(yaxis_tickformat=".1%", xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(rate.rename(columns={"count": "Effectif", "mean": "Taux défaut"}), use_container_width=True)
        else:
            graph_type = st.selectbox(
                "Type de graphique cible",
                ["Taux de défaut par modalité", "Barres empilées 100%", "Effectifs par cible", "Heatmap cible x modalité"],
            )
            tmp = pd.DataFrame({var: df[var].fillna("Missing"), "target": y, "Cible": target_label})
            top_values = tmp[var].value_counts().head(top_n).index
            tmp = tmp[tmp[var].isin(top_values)]
            if graph_type == "Taux de défaut par modalité":
                rate = tmp.groupby(var)["target"].agg(["count", "mean"]).reset_index().sort_values("mean", ascending=False)
                fig = px.bar(rate, x=var, y="mean", color="count", title=f"Taux de défaut par modalité - {var}")
                fig.update_layout(yaxis_tickformat=".1%", xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(rate.rename(columns={"count": "Effectif", "mean": "Taux défaut"}), use_container_width=True)
            elif graph_type == "Barres empilées 100%":
                ct = pd.crosstab(tmp[var], tmp["Cible"], normalize="index").reset_index()
                fig = px.bar(ct, x=var, y=[c for c in ct.columns if c != var], barmode="stack", title=f"Structure cible par modalité - {var}")
                fig.update_layout(yaxis_tickformat=".1%", xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)
            elif graph_type == "Effectifs par cible":
                fig = px.histogram(tmp, x=var, color="Cible", barmode="group", title=f"Effectifs par cible - {var}")
                fig.update_layout(xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)
            else:
                ct = pd.crosstab(tmp[var], tmp["Cible"])
                fig = px.imshow(ct, text_auto=True, color_continuous_scale="Blues", title=f"Heatmap {var} x cible")
                st.plotly_chart(fig, use_container_width=True)

    with tab_cross:
        section("Croisement entre deux variables explicatives")
        var2_candidates = [c for c in features if c != var]
        if not var2_candidates:
            st.info("Ajoutez au moins deux variables explicatives pour utiliser ce module.")
        else:
            var2 = st.selectbox("Deuxième variable", var2_candidates)
            kind = st.selectbox("Type de croisement", ["Nuage de points", "Boxplot groupé", "Heatmap de fréquence", "Taux de défaut croisé"])
            a_num = pd.api.types.is_numeric_dtype(df[var])
            b_num = pd.api.types.is_numeric_dtype(df[var2])
            tmp = plot_df[[var, var2]].copy()
            tmp["Cible"] = plot_target_label
            tmp["target"] = plot_y
            if kind == "Nuage de points" and a_num and b_num:
                fig = px.scatter(tmp, x=var, y=var2, color="Cible", opacity=0.55, title=f"{var} vs {var2}")
                st.plotly_chart(fig, use_container_width=True)
            elif kind == "Boxplot groupé":
                group_var = var2 if not b_num else bin_series(plot_df[var2], min(bins, 10))
                fig_df = pd.DataFrame({var: plot_df[var], var2: group_var, "Cible": plot_target_label})
                if a_num:
                    fig = px.box(fig_df, x=var2, y=var, color="Cible", title=f"{var} par {var2}")
                    fig.update_layout(xaxis_tickangle=-35)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Le boxplot groupé nécessite une variable principale numérique.")
            elif kind == "Taux de défaut croisé":
                a = bin_series(df[var], min(bins, 10))
                b = bin_series(df[var2], min(bins, 10))
                cross = pd.DataFrame({"A": a, "B": b, "target": y}).groupby(["A", "B"])["target"].mean().reset_index()
                fig = px.density_heatmap(cross, x="A", y="B", z="target", histfunc="avg", color_continuous_scale="RdYlGn_r", title=f"Taux de défaut croisé: {var} x {var2}")
                fig.update_layout(xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)
            else:
                a = bin_series(df[var], min(bins, 10))
                b = bin_series(df[var2], min(bins, 10))
                ct = pd.crosstab(a, b)
                fig = px.imshow(ct, text_auto=False, color_continuous_scale="Blues", title=f"Fréquence croisée: {var} x {var2}")
                st.plotly_chart(fig, use_container_width=True)

    with tab_woe:
        section("WOE / Information Value")
        iv, detail = information_value(df[var], y, min(bins, 10))
        st.metric("Information Value", f"{iv:.3f}")
        detail_display = detail.copy()
        for col in ["dist_good", "dist_bad", "bad_rate"]:
            detail_display[col] = detail_display[col].map(lambda x: f"{x:.2%}")
        st.dataframe(detail_display, use_container_width=True)
        fig = px.bar(detail, x="bin", y="woe", title=f"Weight of Evidence - {var}")
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

def page_discriminant() -> None:
    header("🎚️ Discriminant", "Discrétisation, pouvoir discriminant, V de Cramer et redondance entre variables.")
    df = require_data()
    if df is None:
        return
    ok, msg = validate_target(df, st.session_state.target)
    if not ok:
        st.warning(msg)
        return
    target = st.session_state.target
    y = target_series(df, target).astype(int)
    features = usable_features(df, target)
    bins = st.slider("Nombre de classes pour les variables numériques", 3, 10, 5)

    if st.button("Calculer le pouvoir discriminant", type="primary") or st.session_state.discriminant is None:
        rows = []
        for col in features:
            iv, _ = information_value(df[col], y, bins)
            auc_uni = univariate_auc(df[col], y)
            v = cramers_v(bin_series(df[col], bins), y)
            rows.append({"Variable": col, "IV": iv, "AUC univarié": auc_uni, "V de Cramer": v, "Type": str(df[col].dtype), "Manquants": df[col].isna().mean()})
        disc = pd.DataFrame(rows).sort_values(["IV", "AUC univarié"], ascending=False)
        st.session_state.discriminant = disc

    disc = st.session_state.discriminant
    st.dataframe(disc, use_container_width=True)

    section("Matrice de dépendance entre variables candidates")
    top_vars = disc.head(15)["Variable"].tolist()
    chosen = st.multiselect("Variables dans la matrice", features, default=top_vars)
    if len(chosen) >= 2:
        matrix = pd.DataFrame(index=chosen, columns=chosen, dtype=float)
        for a in chosen:
            for b in chosen:
                if a == b:
                    matrix.loc[a, b] = 1
                elif pd.api.types.is_numeric_dtype(df[a]) and pd.api.types.is_numeric_dtype(df[b]):
                    matrix.loc[a, b] = abs(df[[a, b]].corr(method="spearman").iloc[0, 1])
                else:
                    matrix.loc[a, b] = cramers_v(bin_series(df[a], bins), bin_series(df[b], bins))
        fig = px.imshow(matrix.astype(float), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=0, zmax=1)
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)


def page_modeles() -> None:
    header("🤖 Modèles", "Modélisation, sélection de variables et régression logistique.")
    df = require_data()
    if df is None:
        return
    ok, msg = validate_target(df, st.session_state.target)
    if not ok:
        st.warning(msg)
        return
    target = st.session_state.target
    features = usable_features(df, target)
    if not features:
        st.warning("Aucune variable explicative disponible.")
        return

    section("Choix du modèle")
    model_type = st.selectbox("Type de modèle", ["Logistique", "Random Forest", "Gradient Boosting"])

    section("Sélection des variables")
    disc = st.session_state.discriminant
    default_features = disc.head(12)["Variable"].tolist() if isinstance(disc, pd.DataFrame) else features[: min(12, len(features))]
    mode = st.radio("Méthode de présélection", ["Stepwise", "Manuelle", "Top IV", "L1 régularisée"], horizontal=True)
    if mode == "Top IV" and isinstance(disc, pd.DataFrame):
        k = st.slider("Nombre de variables à retenir", 2, min(30, len(features)), min(10, len(features)))
        selected = disc.head(k)["Variable"].tolist()
        st.info(", ".join(selected))
    elif mode == "Stepwise":
        base_candidates = default_features if isinstance(disc, pd.DataFrame) else features
        max_candidates = st.slider("Nombre max de candidates testées", 4, min(25, len(features)), min(15, len(base_candidates)))
        max_selected = st.slider("Nombre max de variables retenues", 2, min(15, len(features)), min(8, len(features)))
        direction = st.selectbox("Type de stepwise", ["both", "forward"], format_func=lambda x: "Forward + Backward" if x == "both" else "Forward")
        candidates = st.multiselect("Variables candidates au stepwise", features, default=base_candidates[:max_candidates])
        if st.button("Lancer le stepwise", type="secondary", disabled=len(candidates) < 1):
            with st.spinner("Stepwise logistique en cours..."):
                selected_stepwise, history = stepwise_logistic_features(
                    df,
                    target,
                    candidates[:max_candidates],
                    direction=direction,
                    max_features=max_selected,
                )
            st.session_state.stepwise_selected = selected_stepwise
            st.session_state.stepwise_history = history
        selected = st.session_state.get("stepwise_selected", candidates[: min(max_selected, len(candidates))])
        if selected:
            st.success("Variables retenues par stepwise: " + ", ".join(selected))
        history = st.session_state.get("stepwise_history")
        if isinstance(history, pd.DataFrame) and not history.empty:
            st.dataframe(history, use_container_width=True)
    elif mode == "L1 régularisée":
        with st.spinner("Sélection L1 en cours..."):
            selected = l1_candidate_features(df, target, features)
        st.info(", ".join(selected))
    else:
        selected = st.multiselect("Variables explicatives", features, default=default_features)

    test_size = st.slider("Part test", 0.15, 0.40, 0.20, 0.01)

    date_col = st.session_state.date_col
    oot_cutoff = None
    if date_col:
        use_oot = st.checkbox("Utiliser un découpage temporel Out-of-Time (OOT)")
        if use_oot:
            st.info(f"Colonne de date détectée : **{date_col}**. Les observations avant la date de coupure servent au train, celles après au test OOT.")
            oot_cutoff = st.date_input("Date de coupure OOT")

    col_train, col_bench = st.columns(2)
    with col_train:
        if col_train.button("Entraîner le modèle", type="primary", disabled=len(selected) < 1):
            try:
                with st.spinner("Estimation du modèle..."):
                    model = prepare_model(df, target, features, selected, test_size, model_type=model_type, date_col=date_col if oot_cutoff else None, oot_cutoff=oot_cutoff)
                st.session_state.model = model
                st.session_state.segmentation = None
                st.session_state.benchmark_results = None
                st.session_state.cv_results = None
                st.success(f"Modèle {model_type} entraîné.")
            except Exception as exc:
                st.error(f"Le modèle n'a pas pu être entraîné: {exc}")
    with col_bench:
        if col_bench.button("Benchmark (3 modèles)", type="secondary", disabled=len(selected) < 1):
            with st.spinner("Benchmark en cours (Logistique + RF + GBM)..."):
                try:
                    bench = benchmark_models(df, target, features, selected, test_size)
                    st.session_state.benchmark_results = bench
                except Exception as exc:
                    st.error(f"Benchmark impossible: {exc}")

    bench = st.session_state.benchmark_results
    if isinstance(bench, pd.DataFrame):
        section("Benchmark — comparaison des 3 modèles")
        st.dataframe(bench, use_container_width=True)
        best_row = bench.loc[bench["AUC test"].idxmax()]
        st.success(f"Meilleur modèle selon l'AUC test : **{best_row['Modèle']}** (AUC = {best_row['AUC test']:.4f})")

    model = st.session_state.model
    if model:
        section(f"Résultats du modèle ({model.model_type})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC test", f"{model.metrics['auc_test']:.3f}")
        c2.metric("Gini test", f"{model.metrics['gini_test']:.3f}")
        c3.metric("Recall", f"{model.metrics['recall']:.3f}")
        c4.metric("F1", f"{model.metrics['f1']:.3f}")
        coef_label = "Coefficients / Importances"
        section(coef_label)
        st.dataframe(model.coefficients, use_container_width=True)
        if model.model_type == "Logistique":
            section("Multicolinéarité (VIF)")
            st.dataframe(vif_table(model), use_container_width=True)


def page_performance() -> None:
    header("🏋🏾 Performance", "Statistiques et performance du modèle.")
    model: ModelArtifacts = st.session_state.model
    if not model:
        st.warning("Entraînez d'abord un modèle dans la page Modèles.")
        return

    section("Indicateurs clés")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUC train", f"{model.metrics['auc_train']:.3f}")
    c2.metric("AUC test", f"{model.metrics['auc_test']:.3f}")
    c3.metric("Gini test", f"{model.metrics['gini_test']:.3f}")
    c4.metric("Precision", f"{model.metrics['precision']:.3f}")
    c5.metric("Recall", f"{model.metrics['recall']:.3f}")

    tab1, tab2, tab3, tab4, tab_cv, tab_psi = st.tabs(["ROC", "Matrice de confusion", "Scores", "Poids", "Validation croisée", "Stabilité PSI"])
    with tab1:
        fpr, tpr, _ = roc_curve(model.y_test, model.y_test_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={auc(fpr, tpr):.3f}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Aléatoire"))
        fig.update_layout(xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs", height=520)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        y_pred = (model.y_test_proba >= 0.5).astype(int)
        cm = confusion_matrix(model.y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, labels=dict(x="Prédit", y="Réel"), color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        tmp = pd.DataFrame({"PD": model.y_test_proba, "Statut": model.y_test.map({0: "Non défaut", 1: "Défaut"})})
        fig = px.histogram(tmp, x="PD", color="Statut", nbins=35, barmode="overlay", opacity=0.65, title="Distribution des PD sur l'échantillon test")
        st.plotly_chart(fig, use_container_width=True)
    with tab4:
        st.dataframe(model.coefficients, use_container_width=True)
        fig = px.bar(model.coefficients.head(25), x="Poids (%)", y="Variable encodée", orientation="h", title="Variables les plus contributives")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=650)
        st.plotly_chart(fig, use_container_width=True)
    with tab_cv:
        section("Validation croisée k-fold stratifiée")
        n_folds = st.slider("Nombre de folds", 3, 10, 5, key="cv_folds")
        if st.button("Lancer la validation croisée", key="btn_cv"):
            X_full = pd.concat([model.X_train_raw, model.X_test_raw])
            y_full = pd.concat([model.y_train, model.y_test])
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            with st.spinner(f"Validation croisée {n_folds}-fold en cours..."):
                cv_scores = cross_val_score(model.pipeline, X_full, y_full, cv=kf, scoring="roc_auc", n_jobs=-1)
            st.session_state.cv_results = cv_scores
        cv_res = st.session_state.cv_results
        if cv_res is not None:
            mean_cv, std_cv = float(cv_res.mean()), float(cv_res.std())
            c1, c2, c3 = st.columns(3)
            c1.metric("AUC CV moyen", f"{mean_cv:.4f}")
            c2.metric("Écart-type", f"{std_cv:.4f}")
            c3.metric("Intervalle 95%", f"[{mean_cv - 2*std_cv:.4f} ; {mean_cv + 2*std_cv:.4f}]")
            fold_df = pd.DataFrame({"Fold": range(1, len(cv_res) + 1), "AUC": cv_res})
            fig_cv = px.bar(fold_df, x="Fold", y="AUC", title="AUC par fold", color="AUC", color_continuous_scale="teal")
            fig_cv.add_hline(y=mean_cv, line_dash="dash", annotation_text=f"Moyenne: {mean_cv:.4f}")
            fig_cv.update_layout(yaxis_range=[max(0, mean_cv - 0.15), min(1, mean_cv + 0.15)])
            st.plotly_chart(fig_cv, use_container_width=True)
            if std_cv > 0.05:
                st.warning(f"Écart-type élevé ({std_cv:.4f}) : les métriques sont instables. Vérifiez la taille des données ou augmentez le nombre de folds.")
    with tab_psi:
        section("Stabilité des variables — PSI / CSI (train vs test)")
        st.caption("PSI < 0.10 : stable | 0.10–0.25 : attention | > 0.25 : instable")
        score_psi_train = psi_score(model.y_train_proba, model.y_test_proba)
        st.metric("PSI du score (train vs test)", f"{score_psi_train:.4f}", delta=None)
        if score_psi_train >= 0.25:
            st.error("PSI du score > 0.25 : distribution très différente entre train et test.")
        elif score_psi_train >= 0.10:
            st.warning("PSI du score entre 0.10 et 0.25 : surveiller la stabilité.")
        else:
            st.success("PSI du score < 0.10 : distribution stable.")
        csi_df = csi_table(model)
        if not csi_df.empty:
            fig_csi = px.bar(csi_df, x="Variable", y="PSI (CSI)", color="Statut",
                             color_discrete_map={"Stable": "#15803d", "Attention": "#b45309", "Instable": "#dc2626"},
                             title="CSI par variable (train vs test)")
            fig_csi.add_hline(y=0.10, line_dash="dot", line_color="#b45309", annotation_text="Seuil attention (0.10)")
            fig_csi.add_hline(y=0.25, line_dash="dash", line_color="#dc2626", annotation_text="Seuil instabilité (0.25)")
            fig_csi.update_layout(xaxis_tickangle=-35, height=480)
            st.plotly_chart(fig_csi, use_container_width=True)
            st.dataframe(csi_df, use_container_width=True)
        else:
            st.info("CSI disponible uniquement pour les variables numériques sélectionnées.")


def page_segmentation() -> None:
    header("🧮 Segmentation", "Création des classes de risque, calibration PD et tests statistiques.")
    df = require_data()
    model: ModelArtifacts = st.session_state.model
    if df is None:
        return
    if not model:
        st.warning("Entraînez d'abord un modèle.")
        return

    c1, c2, c3 = st.columns(3)
    n_classes = c1.slider("Nombre de classes de risque", 3, 10, 7)
    method = c2.selectbox("Méthode de segmentation", ["Jenks", "Quantiles"])
    calibration = c3.selectbox("Calibration", ["Binomiale", "Beta"])
    confidence = st.slider("Niveau de confiance", 0.90, 0.99, 0.95, 0.01)

    if st.button("Produire la segmentation", type="primary"):
        try:
            scored, stats_df, metrics = score_dataset(df, model, n_classes, method, st.session_state.target)
            stats_df = calibrate_stats(stats_df, calibration, confidence)
            st.session_state.segmentation = {
                "scored": scored,
                "stats": stats_df,
                "metrics": metrics,
                "method": method,
                "calibration": calibration,
                "confidence": confidence,
            }
            st.success("Segmentation produite.")
        except Exception as exc:
            st.error(f"Segmentation impossible: {exc}")

    seg = st.session_state.segmentation
    if not seg:
        return

    stats_df = seg["stats"]
    metrics = seg["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("AUC portefeuille", f"{metrics['auc']:.3f}")
    c2.metric("Gini portefeuille", f"{metrics['gini']:.3f}")
    c3.metric("HHI", f"{metrics['hhi']:.2f}")
    st.dataframe(stats_df, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stats_df["Classe de risque"], y=stats_df["Taux_défaut"], mode="lines+markers", name="Taux défaut"))
    fig.add_trace(go.Scatter(x=stats_df["Classe de risque"], y=stats_df["PD_calibrée"], mode="lines+markers", name="PD calibrée"))
    fig.update_layout(yaxis_tickformat=".1%", height=460)
    st.plotly_chart(fig, use_container_width=True)

    section("Tests d'homogénéité et d'hétérogénéité")
    num = [c for c in numeric_cols(seg["scored"]) if c not in ["PD", "Score"]]
    split_var = st.selectbox("Variable de découpage intra-classe", num) if num else None
    if split_var:
        intra, inter = homogeneity_tests(seg["scored"], split_var)
        t1, t2 = st.tabs(["Homogénéité intra-classe", "Hétérogénéité inter-classes"])
        with t1:
            st.dataframe(intra, use_container_width=True)
        with t2:
            st.dataframe(inter, use_container_width=True)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        seg["scored"].to_excel(writer, index=False, sheet_name="Scores")
        stats_df.to_excel(writer, index=False, sheet_name="Classes")
    st.download_button("Télécharger les scores Excel", buffer.getvalue(), "cayman_scores_classes.xlsx")


def html_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    out = df.head(max_rows).copy()
    for col in out.select_dtypes(include=[float]).columns:
        out[col] = out[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return out.to_html(index=False, classes="report-table", border=0)


def build_report_html() -> str:
    df = st.session_state.clean_df if st.session_state.clean_df is not None else st.session_state.df
    model: ModelArtifacts = st.session_state.model
    seg = st.session_state.segmentation
    target = st.session_state.target
    y = target_series(df, target).astype(int)
    missing = missing_summary(df)
    disc = st.session_state.discriminant if isinstance(st.session_state.discriminant, pd.DataFrame) else pd.DataFrame()
    today = pd.Timestamp.today().strftime("%d/%m/%Y")
    seg_stats = seg["stats"] if seg else pd.DataFrame()
    seg_metrics = seg["metrics"] if seg else {}
    model_metrics = model.metrics if model else {}
    coef = model.coefficients if model else pd.DataFrame()

    return f"""
    <!doctype html>
    <html lang="fr">
    <head>
      <meta charset="utf-8">
      <title>Rapport de développement de modèle PD</title>
      <style>
        body {{font-family: Arial, sans-serif; color:#111827; margin: 28px; line-height:1.45;}}
        h1, h2, h3 {{color:#0f172a;}}
        h1 {{text-align:center; border-bottom:3px solid #0f766e; padding-bottom:14px;}}
        h2 {{background:#ecfeff; padding:8px 10px; border-left:4px solid #0f766e; margin-top:28px;}}
        .kpis {{display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin:16px 0;}}
        .kpi {{border:1px solid #cbd5e1; border-radius:6px; padding:10px;}}
        .kpi b {{display:block; font-size:12px; color:#64748b;}}
        .kpi span {{font-size:20px; font-weight:700;}}
        .report-table {{border-collapse:collapse; width:100%; font-size:12px; margin:12px 0;}}
        .report-table th {{background:#f1f5f9; border:1px solid #cbd5e1; padding:6px;}}
        .report-table td {{border:1px solid #cbd5e1; padding:6px;}}
        .page-break {{page-break-before: always;}}
      </style>
    </head>
    <body>
      <h1>Développement de modèle de Probabilité de Défaut</h1>
      <p><b>Version:</b> {today}<br><b>Variable cible:</b> {target}<br><b>Outil:</b> Cayman Scoring PD</p>
      <h2>Abstract</h2>
      <p>Ce rapport décrit le développement d'un modèle de probabilité de défaut construit à partir de la base importée dans l'application. Le processus couvre la qualité des données, l'analyse discriminante, la sélection des variables, l'estimation d'un modèle {model.model_type if model else "non spécifié"}, la mesure de performance, la segmentation en classes de risque et la calibration des probabilités de défaut.</p>
      <div class="kpis">
        <div class="kpi"><b>Observations</b><span>{len(df):,}</span></div>
        <div class="kpi"><b>Variables</b><span>{df.shape[1]}</span></div>
        <div class="kpi"><b>Taux de défaut</b><span>{y.mean():.2%}</span></div>
        <div class="kpi"><b>Variables modèle</b><span>{len(model.selected_features) if model else 0}</span></div>
      </div>
      <h2>1. Processus de collecte et périmètre</h2>
      <p>La base de développement contient {len(df):,} observations et {df.shape[1]} variables. Les variables exclues du modèle sont: {", ".join(st.session_state.excluded_cols) or "aucune"}.</p>
      <h2>2. Qualité et pré-traitements</h2>
      <p>Le tableau suivant synthétise les valeurs manquantes les plus significatives. Les traitements appliqués sont: {"; ".join(st.session_state.missing_log) or "aucun traitement manuel enregistré"}.</p>
      {html_table(missing.head(20))}
      <h2>3. Analyse univariée et pouvoir discriminant</h2>
      <p>Les variables sont classées selon l'Information Value, l'AUC univarié et le V de Cramer après discrétisation lorsque nécessaire.</p>
      {html_table(disc.head(25)) if not disc.empty else "<p>Analyse discriminante non calculée.</p>"}
      <h2>4. Fonction de score</h2>
      <p>Le modèle retenu est une régression logistique avec imputation, standardisation des variables numériques et encodage one-hot des variables qualitatives.</p>
      <div class="kpis">
        <div class="kpi"><b>AUC test</b><span>{model_metrics.get("auc_test", 0):.3f}</span></div>
        <div class="kpi"><b>Gini test</b><span>{model_metrics.get("gini_test", 0):.3f}</span></div>
        <div class="kpi"><b>Precision</b><span>{model_metrics.get("precision", 0):.3f}</span></div>
        <div class="kpi"><b>Recall</b><span>{model_metrics.get("recall", 0):.3f}</span></div>
      </div>
      <h3>Variables et importance</h3>
      {html_table(coef.head(30)) if not coef.empty else "<p>Modèle non entraîné.</p>"}
      <h2>5. Segmentation et calibration</h2>
      <p>La segmentation repose sur la méthode {seg.get("method") if seg else "non produite"} et la calibration {seg.get("calibration") if seg else "non produite"}.</p>
      <div class="kpis">
        <div class="kpi"><b>AUC portefeuille</b><span>{seg_metrics.get("auc", 0):.3f}</span></div>
        <div class="kpi"><b>Gini portefeuille</b><span>{seg_metrics.get("gini", 0):.3f}</span></div>
        <div class="kpi"><b>HHI</b><span>{seg_metrics.get("hhi", 0):.2f}</span></div>
        <div class="kpi"><b>Classes</b><span>{len(seg_stats) if not seg_stats.empty else 0}</span></div>
      </div>
      {html_table(seg_stats) if not seg_stats.empty else "<p>Segmentation non produite.</p>"}
      <h2>6. Synthèse du modèle final</h2>
      <p>Le modèle final retient les variables suivantes: {", ".join(model.selected_features) if model else "aucune"}. Les résultats doivent être complétés par la revue experte du sens économique des variables, la stabilité temporelle et les validations out-of-time lorsque les données sont disponibles.</p>
    </body>
    </html>
    """


def page_rapport() -> None:
    header("Rapport", "Génération automatique du rapport de développement PD avec les résultats réels.")
    df = require_data()
    if df is None:
        return
    if not st.session_state.model:
        st.warning("Le rapport final nécessite un modèle entraîné.")
        return
    if not st.session_state.segmentation:
        st.warning("Le rapport final nécessite une segmentation produite.")
        return

    if st.button("Générer le rapport", type="primary"):
        html = build_report_html()
        st.session_state.report_html = html
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f"rapport_cayman_pd_{ts}.html"
        html_path = REPORT_DIR / html_filename
        html_path.write_text(html, encoding="utf-8")
        if HTML is not None:
            try:
                pdf_bytes = HTML(string=html, base_url=str(APP_DIR)).write_pdf()
                st.session_state.report_pdf = pdf_bytes
                (REPORT_DIR / f"rapport_cayman_pd_{ts}.pdf").write_bytes(pdf_bytes)
                st.success(f"Rapport HTML et PDF générés : {html_filename}")
            except Exception as exc:
                st.session_state.report_pdf = None
                st.warning(f"HTML généré ({html_filename}). PDF indisponible dans cet environnement: {exc}")
        else:
            st.session_state.report_pdf = None
            st.warning(f"HTML généré ({html_filename}). WeasyPrint n'est pas disponible pour produire le PDF.")

    if st.session_state.report_html:
        st.download_button("Télécharger le rapport HTML", st.session_state.report_html, "rapport_cayman_pd.html", mime="text/html")
        if st.session_state.report_pdf:
            st.download_button("Télécharger le rapport PDF", st.session_state.report_pdf, "rapport_cayman_pd.pdf", mime="application/pdf")
        with st.expander("Prévisualiser le rapport HTML"):
            components.html(st.session_state.report_html, height=700, scrolling=True)


def main() -> None:
    init_state()
    css()
    page = page_sidebar()
    pages = {
        "Accueil": page_accueil,
        "Manquants": page_manquants,
        "Graphiques": page_graphiques,
        "Discriminant": page_discriminant,
        "Modèles": page_modeles,
        "Performance": page_performance,
        "Segmentation": page_segmentation,
        "Rapport": page_rapport,
    }
    pages[page]()


if __name__ == "__main__":
    main()
