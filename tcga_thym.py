import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# Global settings
# =============================================================================

GDC_API = "https://api.gdc.cancer.gov"
PROJECT_ID = "TCGA-THYM"

OUT_DIR = Path("results_thym")
RANDOM_STATE = None

LOW_RISK_LABEL = "low_risk_A_AB_B1"
HIGH_RISK_LABEL = "high_risk_B2_B3"

# In this script, y = 1 means high-risk B2/B3.
TARGET_LABEL_NAMES = {
    0: LOW_RISK_LABEL,
    1: HIGH_RISK_LABEL,
}

IMMUNE_GENES = sorted(set([
    # T cell markers / cytotoxicity
    "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "TRAC",
    "GZMA", "GZMB", "GZMH", "GZMK", "PRF1", "NKG7", "GNLY",
    "IFNG", "TBX21", "EOMES",

    # Immune checkpoints / costimulation
    "PDCD1", "CD274", "PDCD1LG2", "CTLA4", "LAG3", "TIGIT",
    "HAVCR2", "VSIR", "BTLA", "ICOS", "CD28", "CD40", "CD40LG",
    "TNFRSF4", "TNFRSF9", "TNFRSF18", "CD27", "CD70",

    # Antigen presentation / HLA
    "HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1", "HLA-DPA1",
    "HLA-DPB1", "HLA-DQA1", "HLA-DQB1", "B2M", "TAP1", "TAP2",
    "PSMB8", "PSMB9", "CIITA",

    # Interferon response
    "STAT1", "STAT2", "IRF1", "IRF3", "IRF7", "ISG15", "MX1",
    "OAS1", "OAS2", "OAS3", "IFI6", "IFI27", "IFI44", "IFI44L",
    "CXCL9", "CXCL10", "CXCL11",

    # Cytokines and receptors
    "IL2", "IL2RA", "IL6", "IL6R", "IL7R", "IL10", "IL10RA",
    "IL12A", "IL12B", "IL15", "IL15RA", "IL18", "TGFB1",
    "TNF", "CSF1", "CSF1R",

    # Chemokines and receptors
    "CCL2", "CCL3", "CCL4", "CCL5", "CCL17", "CCL18", "CCL19",
    "CCL20", "CCL21", "CCL22", "CCR2", "CCR5", "CCR7", "CXCR3",
    "CXCR4", "CXCR5",

    # B cells / plasma cells
    "MS4A1", "CD19", "CD79A", "CD79B", "BANK1", "BLK", "MZB1",
    "SDC1", "JCHAIN", "IGHG1",

    # NK cells
    "NCAM1", "KLRD1", "KLRK1", "KLRB1", "FCGR3A", "NCR1", "NCR3",

    # Myeloid / macrophage / dendritic
    "CD14", "LYZ", "ITGAM", "ITGAX", "FCGR1A", "FCGR2A", "FCGR3B",
    "MSR1", "MRC1", "CD68", "CD163", "MARCO", "LST1",
    "CLEC9A", "BATF3", "IRF8", "CCR7",

    # Treg / immune suppression
    "FOXP3", "IKZF2", "IL2RA", "ENTPD1", "NT5E", "IDO1", "ARG1",

    # Inflammation / innate immune sensing
    "TLR2", "TLR3", "TLR4", "TLR7", "TLR8", "TLR9", "MYD88",
    "NFKB1", "NFKBIA", "RELA", "CASP1", "NLRP3",
]))

IMMUNE_PATHWAYS = {
    "t_cell_score": [
        "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "TRAC",
        "TBX21", "EOMES",
    ],
    "cytotoxicity_score": [
        "GZMA", "GZMB", "GZMH", "GZMK", "PRF1", "NKG7", "GNLY", "IFNG",
    ],
    "checkpoint_score": [
        "PDCD1", "CD274", "PDCD1LG2", "CTLA4", "LAG3", "TIGIT",
        "HAVCR2", "VSIR", "BTLA",
    ],
    "antigen_presentation_score": [
        "HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1", "HLA-DPA1",
        "HLA-DPB1", "HLA-DQA1", "HLA-DQB1", "B2M", "TAP1", "TAP2",
        "PSMB8", "PSMB9", "CIITA",
    ],
    "interferon_score": [
        "STAT1", "STAT2", "IRF1", "IRF3", "IRF7", "ISG15", "MX1",
        "OAS1", "OAS2", "OAS3", "IFI6", "IFI27", "IFI44", "IFI44L",
        "CXCL9", "CXCL10", "CXCL11",
    ],
    "chemokine_score": [
        "CCL2", "CCL3", "CCL4", "CCL5", "CCL17", "CCL18", "CCL19",
        "CCL20", "CCL21", "CCL22", "CCR2", "CCR5", "CCR7", "CXCR3",
        "CXCR4", "CXCR5",
    ],
    "b_cell_score": [
        "MS4A1", "CD19", "CD79A", "CD79B", "BANK1", "BLK", "MZB1",
        "SDC1", "JCHAIN", "IGHG1",
    ],
    "nk_cell_score": [
        "NCAM1", "KLRD1", "KLRK1", "KLRB1", "FCGR3A", "NCR1", "NCR3",
    ],
    "myeloid_macrophage_score": [
        "CD14", "LYZ", "ITGAM", "ITGAX", "FCGR1A", "FCGR2A", "FCGR3B",
        "MSR1", "MRC1", "CD68", "CD163", "MARCO", "LST1", "CSF1R",
    ],
    "treg_suppression_score": [
        "FOXP3", "IKZF2", "IL2RA", "ENTPD1", "NT5E", "IDO1", "ARG1",
        "TGFB1", "IL10",
    ],
    "innate_inflammation_score": [
        "TLR2", "TLR3", "TLR4", "TLR7", "TLR8", "TLR9", "MYD88",
        "NFKB1", "NFKBIA", "RELA", "CASP1", "NLRP3",
    ],
}


# =============================================================================
# General utilities
# =============================================================================

def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_k_values(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return sorted(set(values))


def make_onehot_encoder():
    """
    Handles scikit-learn version differences.
    New versions use sparse_output=False.
    Old versions use sparse=False.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def post_gdc(endpoint: str, payload: dict, timeout: int = 120) -> dict:
    url = f"{GDC_API}/{endpoint}"
    response = requests.post(url, json=payload, timeout=timeout)

    if not response.ok:
        print(f"\nGDC request failed: {url}", file=sys.stderr)
        print(response.text[:2000], file=sys.stderr)

    response.raise_for_status()
    return response.json()


def first_or_none(value):
    if isinstance(value, list) and len(value) > 0:
        return value[0]
    return None


def get_nested(obj, path: List[str], default=None):
    current = obj
    for key in path:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current


# =============================================================================
# GDC metadata and expression download
# =============================================================================

def flatten_file_hit(hit: dict) -> dict:
    case = first_or_none(hit.get("cases", [])) or {}
    sample = first_or_none(case.get("samples", [])) or {}
    diagnosis = first_or_none(case.get("diagnoses", [])) or {}
    demographic = case.get("demographic", {}) or {}

    return {
        "file_id": hit.get("file_id"),
        "file_name": hit.get("file_name"),
        "data_type": hit.get("data_type"),
        "workflow_type": get_nested(hit, ["analysis", "workflow_type"]),

        "case_id": case.get("case_id"),
        "case_submitter_id": case.get("submitter_id"),

        "sample_id": sample.get("sample_id"),
        "sample_submitter_id": sample.get("submitter_id"),
        "sample_type": sample.get("sample_type"),

        "primary_diagnosis": diagnosis.get("primary_diagnosis"),
        "morphology": diagnosis.get("morphology"),
        "tissue_or_organ_of_origin": diagnosis.get("tissue_or_organ_of_origin"),
        "ajcc_pathologic_stage": diagnosis.get("ajcc_pathologic_stage"),
        "tumor_stage": diagnosis.get("tumor_stage"),
        "age_at_diagnosis": diagnosis.get("age_at_diagnosis"),

        "vital_status": demographic.get("vital_status"),
        "gender": demographic.get("gender"),
        "race": demographic.get("race"),
        "ethnicity": demographic.get("ethnicity"),
    }


def query_tcga_thym_star_count_files() -> pd.DataFrame:
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [PROJECT_ID],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_category",
                    "value": ["Transcriptome Profiling"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_type",
                    "value": ["Gene Expression Quantification"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "analysis.workflow_type",
                    "value": ["STAR - Counts"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "cases.samples.sample_type",
                    "value": ["Primary Tumor"],
                },
            },
        ],
    }

    fields = [
        "file_id",
        "file_name",
        "data_type",
        "analysis.workflow_type",
        "cases.case_id",
        "cases.submitter_id",
        "cases.samples.sample_id",
        "cases.samples.submitter_id",
        "cases.samples.sample_type",
        "cases.diagnoses.primary_diagnosis",
        "cases.diagnoses.morphology",
        "cases.diagnoses.tissue_or_organ_of_origin",
        "cases.diagnoses.ajcc_pathologic_stage",
        "cases.diagnoses.tumor_stage",
        "cases.diagnoses.age_at_diagnosis",
        "cases.demographic.vital_status",
        "cases.demographic.gender",
        "cases.demographic.race",
        "cases.demographic.ethnicity",
    ]

    payload = {
        "filters": filters,
        "fields": ",".join(fields),
        "format": "JSON",
        "size": 2000,
    }

    print("Querying GDC for TCGA-THYM STAR-counts files...")
    data = post_gdc("files", payload)
    hits = data.get("data", {}).get("hits", [])

    if not hits:
        raise RuntimeError("No TCGA-THYM STAR-counts files were found.")

    rows = [flatten_file_hit(hit) for hit in hits]
    meta = pd.DataFrame(rows)

    meta = (
        meta.sort_values(["case_submitter_id", "file_name"])
        .drop_duplicates(subset=["case_submitter_id"], keep="first")
        .reset_index(drop=True)
    )

    print(f"Found {len(meta)} TCGA-THYM primary-tumor cases with STAR-counts.")
    return meta


def download_gdc_file(file_id: str, cache_dir: Path, file_name: Optional[str] = None) -> Path:
    safe_mkdir(cache_dir)

    if file_name is None:
        file_name = f"{file_id}.tsv"

    out_path = cache_dir / file_name

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    url = f"{GDC_API}/data/{file_id}"
    print(f"Downloading {file_id}...")
    response = requests.get(url, timeout=300)

    if not response.ok:
        print(response.text[:2000], file=sys.stderr)

    response.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(response.content)

    time.sleep(0.05)
    return out_path


def read_star_counts_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        df = pd.read_csv(path, sep="\t", comment="#", compression="gzip")
    else:
        df = pd.read_csv(path, sep="\t", comment="#")

    df.columns = [str(c).strip() for c in df.columns]

    if "gene_name" not in df.columns:
        raise ValueError(f"{path} does not contain a gene_name column.")

    value_col = None
    for candidate in [
        "tpm_unstranded",
        "fpkm_uq_unstranded",
        "fpkm_unstranded",
        "unstranded",
    ]:
        if candidate in df.columns:
            value_col = candidate
            break

    if value_col is None:
        raise ValueError(
            f"No usable expression column found in {path}. "
            f"Columns: {list(df.columns)}"
        )

    out = df[["gene_name", value_col]].copy()
    out = out.rename(columns={value_col: "expression"})
    out["gene_name"] = out["gene_name"].astype(str)

    out = out[~out["gene_name"].str.startswith("__", na=False)]
    out["expression"] = pd.to_numeric(out["expression"], errors="coerce").fillna(0.0)

    # If duplicate gene symbols exist, keep the row with highest expression.
    out = (
        out.sort_values("expression", ascending=False)
        .drop_duplicates(subset=["gene_name"], keep="first")
    )

    return out[["gene_name", "expression"]]


def build_expression_matrix(meta: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    sample_series = {}

    for _, row in meta.iterrows():
        file_id = row["file_id"]
        file_name = row["file_name"]
        case_submitter_id = row["case_submitter_id"]

        path = download_gdc_file(file_id, cache_dir, file_name=file_name)

        try:
            expr = read_star_counts_file(path)
        except Exception as e:
            print(f"Could not parse {path}: {e}", file=sys.stderr)
            continue

        sample_series[case_submitter_id] = expr.set_index("gene_name")["expression"]

    if not sample_series:
        raise RuntimeError("No expression files were parsed successfully.")

    expression = pd.DataFrame(sample_series).T
    expression.index.name = "case_submitter_id"
    expression = expression.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    print(f"Built expression matrix: {expression.shape[0]} samples x {expression.shape[1]} genes.")
    return expression


# =============================================================================
# Label creation
# =============================================================================

def parse_thymoma_who_subtype(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None

    t = text.upper()

    # Check AB before A.
    patterns = [
        ("AB", r"\bTYPE\s+AB\b|\bAB\b"),
        ("B1", r"\bTYPE\s+B1\b|\bB1\b"),
        ("B2", r"\bTYPE\s+B2\b|\bB2\b"),
        ("B3", r"\bTYPE\s+B3\b|\bB3\b"),
        ("A", r"\bTYPE\s+A\b|\bTYPE-A\b"),
    ]

    for subtype, pattern in patterns:
        if re.search(pattern, t):
            return subtype

    return None


def create_target_labels(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()

    subtypes = []
    for _, row in meta.iterrows():
        combined_text = " | ".join([
            str(row.get("primary_diagnosis", "")),
            str(row.get("morphology", "")),
            str(row.get("tissue_or_organ_of_origin", "")),
            str(row.get("tumor_stage", "")),
        ])
        subtypes.append(parse_thymoma_who_subtype(combined_text))

    meta["who_subtype"] = subtypes

    risk_mapping = {
        "A": LOW_RISK_LABEL,
        "AB": LOW_RISK_LABEL,
        "B1": LOW_RISK_LABEL,
        "B2": HIGH_RISK_LABEL,
        "B3": HIGH_RISK_LABEL,
    }

    meta["target_label"] = meta["who_subtype"].map(risk_mapping)
    meta = meta.dropna(subset=["target_label"]).copy()

    meta["target_binary"] = (meta["target_label"] == HIGH_RISK_LABEL).astype(int)

    print("\nWHO subtype counts:")
    print(meta["who_subtype"].value_counts(dropna=False))

    print("\nTarget counts:")
    print(meta["target_label"].value_counts(dropna=False))

    if meta["target_binary"].nunique() < 2:
        raise RuntimeError("Only one target class was found.")

    min_class_count = meta["target_binary"].value_counts().min()
    if min_class_count < 5:
        raise RuntimeError(
            f"Smallest target class has only {min_class_count} samples. "
            "This is too small for stable cross-validation."
        )

    return meta.reset_index(drop=True)


# =============================================================================
# Feature preparation
# =============================================================================

def simplify_stage(value) -> str:
    if pd.isna(value):
        return "unknown"

    text = str(value).strip().upper()

    if text in ["", "NAN", "NONE", "NOT REPORTED", "UNKNOWN"]:
        return "unknown"

    # Check higher stages first so IV is not accidentally parsed as I.
    if "STAGE IV" in text or text == "IV":
        return "stage_iv"
    if "STAGE III" in text or text == "III":
        return "stage_iii"
    if "STAGE II" in text or text == "II":
        return "stage_ii"
    if "STAGE I" in text or text == "I":
        return "stage_i"

    return "other_or_unknown"


def prepare_expression_features(
    expression: pd.DataFrame,
    meta_labeled: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    common_cases = sorted(
        set(expression.index).intersection(set(meta_labeled["case_submitter_id"]))
    )

    if not common_cases:
        raise RuntimeError("No overlap between expression and metadata cases.")

    meta_indexed = meta_labeled.set_index("case_submitter_id").loc[common_cases].copy()
    y = meta_indexed["target_binary"].astype(int)

    present_genes = [g for g in IMMUNE_GENES if g in expression.columns]
    missing_genes = [g for g in IMMUNE_GENES if g not in expression.columns]

    print(f"\nImmune genes requested: {len(IMMUNE_GENES)}")
    print(f"Immune genes found in expression matrix: {len(present_genes)}")
    print(f"Immune genes missing: {len(missing_genes)}")

    if len(present_genes) < 20:
        raise RuntimeError("Too few immune genes were found in the expression matrix.")

    X_genes = expression.loc[common_cases, present_genes].copy()
    X_genes = np.log2(X_genes + 1.0)

    X_genes.index.name = "case_submitter_id"

    print(f"Using all {X_genes.shape[1]} immune genes before feature selection.")
    print(f"Final gene-expression matrix: {X_genes.shape[0]} samples x {X_genes.shape[1]} genes.")

    return X_genes, y, meta_indexed


def make_pathway_scores(X_genes: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame(index=X_genes.index)

    for pathway_name, genes in IMMUNE_PATHWAYS.items():
        present = [g for g in genes if g in X_genes.columns]

        if len(present) == 0:
            scores[pathway_name] = np.nan
        else:
            scores[pathway_name] = X_genes[present].mean(axis=1)

    scores = scores.dropna(axis=1, how="all")
    scores = scores.fillna(scores.median(numeric_only=True))

    return scores


def make_clinical_features(meta_indexed: pd.DataFrame) -> pd.DataFrame:
    clinical = pd.DataFrame(index=meta_indexed.index)

    age_days = pd.to_numeric(meta_indexed.get("age_at_diagnosis"), errors="coerce")
    clinical["age_at_diagnosis_years"] = age_days / 365.25

    for col in ["gender", "race", "ethnicity"]:
        if col in meta_indexed.columns:
            clinical[col] = (
                meta_indexed[col]
                .astype(str)
                .replace({"nan": "unknown", "None": "unknown", "not reported": "unknown"})
                .fillna("unknown")
            )
        else:
            clinical[col] = "unknown"

    stage_source = None
    if "ajcc_pathologic_stage" in meta_indexed.columns:
        stage_source = meta_indexed["ajcc_pathologic_stage"]
    elif "tumor_stage" in meta_indexed.columns:
        stage_source = meta_indexed["tumor_stage"]

    if stage_source is not None:
        clinical["stage_simplified"] = stage_source.apply(simplify_stage)
    else:
        clinical["stage_simplified"] = "unknown"

    return clinical


# =============================================================================
# Model construction
# =============================================================================

def make_rf(
    n_estimators: int = 700,
    max_depth=None,
    min_samples_leaf: int = 3,
    max_features="sqrt",
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_logistic() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def make_gene_rf_all() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", make_rf()),
    ])


def make_gene_rf_selectk(k: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=f_classif, k=k)),
        ("clf", make_rf()),
    ])


def make_gene_logistic_selectk(k: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=f_classif, k=k)),
        ("scaler", StandardScaler()),
        ("clf", make_logistic()),
    ])


def make_gene_rf_selectfrommodel() -> Pipeline:
    selector_rf = make_rf(
        n_estimators=700,
        min_samples_leaf=3,
        max_features="sqrt",
    )

    final_rf = make_rf(
        n_estimators=700,
        min_samples_leaf=3,
        max_features="sqrt",
    )

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectFromModel(selector_rf, threshold="median")),
        ("clf", final_rf),
    ])


def make_numeric_rf_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", make_rf()),
    ])


def make_numeric_logistic_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", make_logistic()),
    ])


def make_clinical_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder()),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])


def make_clinical_rf_model(X_clinical: pd.DataFrame) -> Pipeline:
    preprocessor = make_clinical_preprocessor(X_clinical)

    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", make_rf()),
    ])


def make_clinical_logistic_model(X_clinical: pd.DataFrame) -> Pipeline:
    preprocessor = make_clinical_preprocessor(X_clinical)

    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", make_logistic()),
    ])


def make_combined_preprocessor(
    gene_cols: List[str],
    clinical_df: pd.DataFrame,
    k: Optional[int] = None,
    scale_genes: bool = False,
) -> ColumnTransformer:
    clinical_numeric_cols = list(clinical_df.select_dtypes(include=[np.number]).columns)
    clinical_categorical_cols = [c for c in clinical_df.columns if c not in clinical_numeric_cols]

    gene_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if k is not None:
        gene_steps.append(("select", SelectKBest(score_func=f_classif, k=k)))

    if scale_genes:
        gene_steps.append(("scaler", StandardScaler()))

    gene_pipe = Pipeline(gene_steps)

    clinical_numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    clinical_categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder()),
    ])

    transformers = [
        ("genes", gene_pipe, gene_cols),
    ]

    if clinical_numeric_cols:
        transformers.append(("clinical_numeric", clinical_numeric_pipe, clinical_numeric_cols))

    if clinical_categorical_cols:
        transformers.append(("clinical_categorical", clinical_categorical_pipe, clinical_categorical_cols))

    return ColumnTransformer(transformers)


def make_gene_clinical_rf_model(
    gene_cols: List[str],
    clinical_df: pd.DataFrame,
    k: int,
) -> Pipeline:
    preprocessor = make_combined_preprocessor(
        gene_cols=gene_cols,
        clinical_df=clinical_df,
        k=k,
        scale_genes=False,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", make_rf()),
    ])


def make_gene_clinical_logistic_model(
    gene_cols: List[str],
    clinical_df: pd.DataFrame,
    k: int,
) -> Pipeline:
    preprocessor = make_combined_preprocessor(
        gene_cols=gene_cols,
        clinical_df=clinical_df,
        k=k,
        scale_genes=True,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", make_logistic()),
    ])


# =============================================================================
# Evaluation
# =============================================================================

def get_positive_class_probability(fitted_model, X_test: pd.DataFrame) -> np.ndarray:
    prob = fitted_model.predict_proba(X_test)

    if hasattr(fitted_model, "classes_"):
        classes = list(fitted_model.classes_)
    elif hasattr(fitted_model, "named_steps"):
        classes = list(fitted_model.named_steps["clf"].classes_)
    else:
        raise RuntimeError("Could not determine estimator classes.")

    if 1 not in classes:
        raise RuntimeError(f"Positive class 1 not found in classes: {classes}")

    positive_index = classes.index(1)
    return prob[:, positive_index]


def calculate_metrics(y_true: np.ndarray, prob_high: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob_high >= threshold).astype(int)

    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
        "precision_high_risk": precision_score(y_true, pred, pos_label=1, zero_division=0),
        "recall_high_risk": recall_score(y_true, pred, pos_label=1, zero_division=0),
        "f1_high_risk": f1_score(y_true, pred, pos_label=1, zero_division=0),
    }

    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, prob_high)
    else:
        out["roc_auc"] = np.nan

    return out


def evaluate_repeated_cv(
    model_name: str,
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    y_array = np.asarray(y).astype(int)
    n = len(y_array)

    prob_sum = np.zeros(n, dtype=float)
    prob_count = np.zeros(n, dtype=float)

    fold_metric_rows = []
    prediction_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y_array), start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y_array[train_idx]
        y_test = y_array[test_idx]

        model = clone(estimator)
        model.fit(X_train, y_train)

        prob_high = get_positive_class_probability(model, X_test)
        pred = (prob_high >= 0.5).astype(int)

        prob_sum[test_idx] += prob_high
        prob_count[test_idx] += 1

        fold_metrics = calculate_metrics(y_test, prob_high)
        fold_metrics["model"] = model_name
        fold_metrics["fold_id"] = fold_id
        fold_metrics["n_test"] = len(test_idx)
        fold_metric_rows.append(fold_metrics)

        for local_i, global_i in enumerate(test_idx):
            prediction_rows.append({
                "model": model_name,
                "fold_id": fold_id,
                "case_submitter_id": X.index[global_i],
                "true_binary": int(y_array[global_i]),
                "true_label": TARGET_LABEL_NAMES[int(y_array[global_i])],
                "prob_high_risk": float(prob_high[local_i]),
                "pred_binary": int(pred[local_i]),
                "pred_label": TARGET_LABEL_NAMES[int(pred[local_i])],
            })

    if np.any(prob_count == 0):
        raise RuntimeError("Some samples were never evaluated in repeated CV.")

    avg_prob_high = prob_sum / prob_count
    aggregate_metrics = calculate_metrics(y_array, avg_prob_high)

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    fold_means = fold_metrics_df.drop(columns=["model", "fold_id"]).mean(numeric_only=True)
    fold_stds = fold_metrics_df.drop(columns=["model", "fold_id"]).std(numeric_only=True)

    summary = {
        "model": model_name,
        "n_samples": n,
        "n_features_input": X.shape[1],
        "n_cv_predictions_per_sample": int(prob_count.min()),
    }

    for metric_name, metric_value in aggregate_metrics.items():
        summary[f"aggregate_{metric_name}"] = metric_value

    for metric_name, metric_value in fold_means.items():
        summary[f"fold_mean_{metric_name}"] = metric_value

    for metric_name, metric_value in fold_stds.items():
        summary[f"fold_std_{metric_name}"] = metric_value

    aggregate_pred_df = pd.DataFrame({
        "model": model_name,
        "case_submitter_id": X.index,
        "true_binary": y_array,
        "true_label": [TARGET_LABEL_NAMES[int(v)] for v in y_array],
        "mean_prob_high_risk": avg_prob_high,
        "pred_binary": (avg_prob_high >= 0.5).astype(int),
    })

    aggregate_pred_df["pred_label"] = aggregate_pred_df["pred_binary"].map(TARGET_LABEL_NAMES)

    return summary, fold_metrics_df, aggregate_pred_df


def run_model_comparison(
    models: List[Tuple[str, object, pd.DataFrame]],
    y: pd.Series,
    n_splits: int,
    n_repeats: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )

    summary_rows = []
    fold_metrics_all = []
    aggregate_predictions_all = []

    for model_name, estimator, X in models:
        print("\n" + "=" * 80)
        print(f"Evaluating model: {model_name}")
        print("=" * 80)

        summary, fold_metrics_df, aggregate_pred_df = evaluate_repeated_cv(
            model_name=model_name,
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
        )

        summary_rows.append(summary)
        fold_metrics_all.append(fold_metrics_df)
        aggregate_predictions_all.append(aggregate_pred_df)

        print(
            f"{model_name}: "
            f"ROC-AUC={summary['aggregate_roc_auc']:.3f}, "
            f"Balanced Acc={summary['aggregate_balanced_accuracy']:.3f}, "
            f"Macro F1={summary['aggregate_macro_f1']:.3f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    fold_metrics_df = pd.concat(fold_metrics_all, axis=0, ignore_index=True)
    aggregate_predictions_df = pd.concat(aggregate_predictions_all, axis=0, ignore_index=True)

    summary_df = summary_df.sort_values(
        ["aggregate_roc_auc", "aggregate_balanced_accuracy", "aggregate_macro_f1"],
        ascending=False,
    ).reset_index(drop=True)

    return summary_df, fold_metrics_df, aggregate_predictions_df


# =============================================================================
# Feature extraction from fitted final models
# =============================================================================

def extract_features_from_gene_pipeline(
    fitted_pipeline: Pipeline,
    original_feature_names: List[str],
) -> pd.DataFrame:
    feature_names = np.array(original_feature_names)

    if "select" in fitted_pipeline.named_steps:
        selector = fitted_pipeline.named_steps["select"]

        if hasattr(selector, "get_support"):
            support = selector.get_support()
            feature_names = feature_names[support]

    clf = fitted_pipeline.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
        value_name = "importance"
    elif hasattr(clf, "coef_"):
        values = clf.coef_.ravel()
        value_name = "coefficient_for_high_risk"
    else:
        return pd.DataFrame({"feature": feature_names})

    df = pd.DataFrame({
        "feature": feature_names,
        value_name: values,
    })

    if value_name == "coefficient_for_high_risk":
        df["abs_value"] = df[value_name].abs()
        df = df.sort_values("abs_value", ascending=False)
    else:
        df = df.sort_values(value_name, ascending=False)

    return df.reset_index(drop=True)


def extract_features_from_numeric_pipeline(
    fitted_pipeline: Pipeline,
    original_feature_names: List[str],
) -> pd.DataFrame:
    clf = fitted_pipeline.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
        value_name = "importance"
    elif hasattr(clf, "coef_"):
        values = clf.coef_.ravel()
        value_name = "coefficient_for_high_risk"
    else:
        return pd.DataFrame({"feature": original_feature_names})

    df = pd.DataFrame({
        "feature": original_feature_names,
        value_name: values,
    })

    if value_name == "coefficient_for_high_risk":
        df["abs_value"] = df[value_name].abs()
        df = df.sort_values("abs_value", ascending=False)
    else:
        df = df.sort_values(value_name, ascending=False)

    return df.reset_index(drop=True)


# =============================================================================
# Plotting
# =============================================================================

def plot_pca(X_genes: pd.DataFrame, y: pd.Series, out_path: Path) -> None:
    safe_mkdir(out_path.parent)

    X_scaled = StandardScaler().fit_transform(X_genes)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=np.asarray(y), alpha=0.85)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.title("PCA of immune-gene expression in TCGA-THYM")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_model_comparison(summary_df: pd.DataFrame, out_path: Path, metric: str = "aggregate_roc_auc") -> None:
    safe_mkdir(out_path.parent)

    top = summary_df.head(15).iloc[::-1].copy()

    plt.figure(figsize=(10, max(5, 0.35 * len(top))))
    plt.barh(top["model"], top[metric])
    plt.xlabel(metric)
    plt.title("Top model comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_feature_count_performance(summary_df: pd.DataFrame, out_path: Path) -> None:
    safe_mkdir(out_path.parent)

    rows = []
    pattern = re.compile(r"immune_rf_selectk_(\d+)$")

    for _, row in summary_df.iterrows():
        match = pattern.search(row["model"])
        if match:
            rows.append({
                "k": int(match.group(1)),
                "roc_auc": row["aggregate_roc_auc"],
                "balanced_accuracy": row["aggregate_balanced_accuracy"],
                "macro_f1": row["aggregate_macro_f1"],
            })

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("k")

    plt.figure(figsize=(8, 6))
    plt.plot(df["k"], df["roc_auc"], marker="o", label="ROC-AUC")
    plt.plot(df["k"], df["balanced_accuracy"], marker="o", label="Balanced accuracy")
    plt.plot(df["k"], df["macro_f1"], marker="o", label="Macro F1")
    plt.xlabel("Number of selected immune genes")
    plt.ylabel("Metric")
    plt.title("Random forest performance by selected feature count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_confusion_matrix_from_predictions(pred_df: pd.DataFrame, out_path: Path) -> None:
    safe_mkdir(out_path.parent)

    y_true = pred_df["true_binary"].astype(int).values
    y_pred = pred_df["pred_binary"].astype(int).values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = [LOW_RISK_LABEL, HIGH_RISK_LABEL]

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Best model aggregate confusion matrix")
    plt.colorbar()

    ticks = np.arange(2)
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_roc_from_predictions(pred_df: pd.DataFrame, out_path: Path) -> None:
    safe_mkdir(out_path.parent)

    y_true = pred_df["true_binary"].astype(int).values
    prob = pred_df["mean_prob_high_risk"].astype(float).values

    fpr, tpr, _ = roc_curve(y_true, prob)
    auc_value = roc_auc_score(y_true, prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Best model aggregate ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_top_features(feature_df: pd.DataFrame, out_path: Path, title: str, n: int = 20) -> None:
    safe_mkdir(out_path.parent)

    if feature_df.empty:
        return

    if "importance" in feature_df.columns:
        value_col = "importance"
    elif "abs_value" in feature_df.columns:
        value_col = "abs_value"
    elif "coefficient_for_high_risk" in feature_df.columns:
        value_col = "coefficient_for_high_risk"
    else:
        return

    top = feature_df.head(n).copy().iloc[::-1]

    plt.figure(figsize=(8, max(5, 0.3 * len(top))))
    plt.barh(top["feature"], top[value_col])
    plt.xlabel(value_col)
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pathway_score_boxplots(X_pathway: pd.DataFrame, y: pd.Series, out_path: Path) -> None:
    safe_mkdir(out_path.parent)

    pathways = list(X_pathway.columns)
    n = len(pathways)

    plt.figure(figsize=(12, max(6, 0.45 * n)))

    positions = []
    labels = []
    data = []

    pos = 1
    for pathway in pathways:
        low_values = X_pathway.loc[y[y == 0].index, pathway].values
        high_values = X_pathway.loc[y[y == 1].index, pathway].values

        data.extend([low_values, high_values])
        positions.extend([pos, pos + 0.35])
        labels.append(pathway)
        pos += 1.2

    plt.boxplot(data, positions=positions, widths=0.25, showfliers=False)

    tick_positions = [1 + i * 1.2 + 0.175 for i in range(n)]
    plt.xticks(tick_positions, labels, rotation=45, ha="right")
    plt.ylabel("Mean log2 expression score")
    plt.title("Immune pathway scores by thymoma histologic risk group")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Improved TCGA-THYM immune-gene ML classifier"
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(OUT_DIR),
        help="Output directory",
    )

    parser.add_argument(
        "--k-values",
        type=str,
        default="10,20,30,40,60,80,150",
        help="Comma-separated SelectKBest feature counts",
    )

    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of repeated CV repeats.",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of stratified CV folds.",
    )

    parser.add_argument(
        "--reuse-downloaded",
        action="store_true",
        help="Reuse saved expression/metadata files if available.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "gdc_cache"
    fig_dir = out_dir / "figures"

    safe_mkdir(out_dir)
    safe_mkdir(cache_dir)
    safe_mkdir(fig_dir)

    k_values = parse_k_values(args.k_values)

    print("=" * 90)
    print("Improved TCGA-THYM immune-gene explainable ML classifier")
    print("=" * 90)

    metadata_path = out_dir / "metadata_files.csv"
    labeled_metadata_path = out_dir / "metadata_samples.csv"
    expression_path = out_dir / "immune_expression_log2_tpm_all_genes.csv"
    pathway_path = out_dir / "immune_pathway_scores.csv"
    clinical_path = out_dir / "clinical_features.csv"

    # -------------------------------------------------------------------------
    # Load or download metadata
    # -------------------------------------------------------------------------
    if args.reuse_downloaded and metadata_path.exists():
        print(f"Loading metadata from {metadata_path}")
        meta = pd.read_csv(metadata_path)
    else:
        meta = query_tcga_thym_star_count_files()
        meta.to_csv(metadata_path, index=False)

    meta_labeled = create_target_labels(meta)
    meta_labeled.to_csv(labeled_metadata_path, index=False)

    # -------------------------------------------------------------------------
    # Load or download expression
    # -------------------------------------------------------------------------
    if args.reuse_downloaded and expression_path.exists():
        print(f"Loading saved immune expression matrix from {expression_path}")
        X_genes = pd.read_csv(expression_path, index_col=0)

        meta_indexed = meta_labeled.set_index("case_submitter_id").loc[X_genes.index].copy()
        y = meta_indexed["target_binary"].astype(int)
    else:
        expression = build_expression_matrix(meta, cache_dir)
        X_genes, y, meta_indexed = prepare_expression_features(expression, meta_labeled)
        X_genes.to_csv(expression_path)

    # Ensure y index matches X_genes.
    y = pd.Series(y.values, index=X_genes.index, name="target_binary").astype(int)

    # -------------------------------------------------------------------------
    # Create pathway and clinical features
    # -------------------------------------------------------------------------
    X_pathway = make_pathway_scores(X_genes)
    X_pathway.to_csv(pathway_path)

    X_clinical = make_clinical_features(meta_indexed)
    X_clinical = X_clinical.loc[X_genes.index]
    X_clinical.to_csv(clinical_path)

    X_gene_clinical = pd.concat([X_genes, X_clinical], axis=1)
    X_pathway_clinical = pd.concat([X_pathway, X_clinical], axis=1)

    print("\nFinal target distribution:")
    print(y.map(TARGET_LABEL_NAMES).value_counts())

    print("\nGene-expression feature matrix:")
    print(X_genes.shape)

    print("\nPathway-score feature matrix:")
    print(X_pathway.shape)

    print("\nClinical feature matrix:")
    print(X_clinical.shape)

    # Adjust folds if needed.
    smallest_class = y.value_counts().min()
    n_splits = min(args.n_splits, int(smallest_class))

    if n_splits < 2:
        raise RuntimeError("Not enough samples for cross-validation.")

    print(f"\nUsing repeated stratified CV: {n_splits} folds x {args.n_repeats} repeats")

    # -------------------------------------------------------------------------
    # Build model list
    # -------------------------------------------------------------------------
    models = []

    n_gene_features = X_genes.shape[1]
    valid_k_values = [k for k in k_values if k <= n_gene_features]

    # Gene-only models
    models.append(("immune_rf_all_genes", make_gene_rf_all(), X_genes))

    for k in valid_k_values:
        models.append((f"immune_rf_selectk_{k}", make_gene_rf_selectk(k), X_genes))
        models.append((f"immune_logistic_selectk_{k}", make_gene_logistic_selectk(k), X_genes))

    models.append(("immune_rf_selectfrommodel_median", make_gene_rf_selectfrommodel(), X_genes))

    # Pathway-only models
    models.append(("pathway_rf", make_numeric_rf_model(), X_pathway))
    models.append(("pathway_logistic", make_numeric_logistic_model(), X_pathway))

    # Clinical-only models
    models.append(("clinical_rf", make_clinical_rf_model(X_clinical), X_clinical))
    models.append(("clinical_logistic", make_clinical_logistic_model(X_clinical), X_clinical))

    # Pathway + clinical models
    models.append(("pathway_clinical_rf", make_clinical_rf_model(X_pathway_clinical), X_pathway_clinical))
    models.append(("pathway_clinical_logistic", make_clinical_logistic_model(X_pathway_clinical), X_pathway_clinical))

    # Gene + clinical models
    # To keep runtime reasonable, only use the most useful K values for combined models.
    combined_k_values = [k for k in [10, 20, 30, 40, 60, 80] if k <= n_gene_features]

    gene_cols = list(X_genes.columns)

    for k in combined_k_values:
        models.append((
            f"immune_clinical_rf_selectk_{k}",
            make_gene_clinical_rf_model(gene_cols, X_clinical, k),
            X_gene_clinical,
        ))

        models.append((
            f"immune_clinical_logistic_selectk_{k}",
            make_gene_clinical_logistic_model(gene_cols, X_clinical, k),
            X_gene_clinical,
        ))

    # -------------------------------------------------------------------------
    # Run model comparison
    # -------------------------------------------------------------------------
    summary_df, fold_metrics_df, aggregate_predictions_df = run_model_comparison(
        models=models,
        y=y,
        n_splits=n_splits,
        n_repeats=args.n_repeats,
    )

    summary_path = out_dir / "model_metrics_summary.csv"
    fold_metrics_path = out_dir / "model_fold_metrics.csv"
    aggregate_predictions_path = out_dir / "model_aggregate_predictions.csv"

    summary_df.to_csv(summary_path, index=False)
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    aggregate_predictions_df.to_csv(aggregate_predictions_path, index=False)

    print("\n" + "=" * 90)
    print("Top models by aggregate ROC-AUC")
    print("=" * 90)

    cols_to_print = [
        "model",
        "aggregate_roc_auc",
        "aggregate_balanced_accuracy",
        "aggregate_macro_f1",
        "aggregate_accuracy",
        "aggregate_recall_high_risk",
        "aggregate_precision_high_risk",
    ]

    print(summary_df[cols_to_print].head(15).to_string(index=False))

    # -------------------------------------------------------------------------
    # Pick best model
    # -------------------------------------------------------------------------
    best_model_name = summary_df.iloc[0]["model"]
    print("\nBest model:")
    print(best_model_name)

    best_predictions = aggregate_predictions_df[
        aggregate_predictions_df["model"] == best_model_name
    ].copy()

    best_predictions.to_csv(out_dir / "best_model_predictions.csv", index=False)

    # -------------------------------------------------------------------------
    # Fit final best model if it is easy to extract features from.
    # -------------------------------------------------------------------------
    model_lookup = {name: (estimator, X) for name, estimator, X in models}
    best_estimator, best_X = model_lookup[best_model_name]

    final_best = clone(best_estimator)
    final_best.fit(best_X, y.values)

    # Try to extract features if best model is a gene-only or pathway-only pipeline.
    best_feature_df = pd.DataFrame()

    if best_X is X_genes:
        best_feature_df = extract_features_from_gene_pipeline(final_best, list(X_genes.columns))
    elif best_X is X_pathway:
        best_feature_df = extract_features_from_numeric_pipeline(final_best, list(X_pathway.columns))

    if not best_feature_df.empty:
        best_feature_df.to_csv(out_dir / "final_best_model_features.csv", index=False)

    # Also extract features from the best immune-gene-only RF SelectK model.
    immune_rf_rows = summary_df[
        summary_df["model"].str.startswith("immune_rf_selectk_")
    ].copy()

    if not immune_rf_rows.empty:
        best_immune_rf_name = immune_rf_rows.iloc[0]["model"]
        best_immune_rf_estimator, _ = model_lookup[best_immune_rf_name]

        final_immune_rf = clone(best_immune_rf_estimator)
        final_immune_rf.fit(X_genes, y.values)

        immune_rf_features = extract_features_from_gene_pipeline(
            final_immune_rf,
            list(X_genes.columns),
        )

        immune_rf_features.to_csv(out_dir / "final_best_immune_rf_features.csv", index=False)

        print("\nBest immune-gene random forest model:")
        print(best_immune_rf_name)

        print("\nTop selected immune genes:")
        print(immune_rf_features.head(20).to_string(index=False))
    else:
        immune_rf_features = pd.DataFrame()

    # Pathway feature importance
    pathway_rf = make_numeric_rf_model()
    pathway_rf.fit(X_pathway, y.values)
    pathway_features = extract_features_from_numeric_pipeline(pathway_rf, list(X_pathway.columns))
    pathway_features.to_csv(out_dir / "final_pathway_feature_importance.csv", index=False)

    print("\nTop pathway features:")
    print(pathway_features.head(20).to_string(index=False))

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    plot_pca(
        X_genes=X_genes,
        y=y,
        out_path=fig_dir / "pca_immune_expression.png",
    )

    plot_model_comparison(
        summary_df=summary_df,
        out_path=fig_dir / "model_comparison_roc_auc.png",
        metric="aggregate_roc_auc",
    )

    plot_feature_count_performance(
        summary_df=summary_df,
        out_path=fig_dir / "rf_selectk_feature_count_performance.png",
    )

    plot_confusion_matrix_from_predictions(
        pred_df=best_predictions,
        out_path=fig_dir / "best_model_confusion_matrix.png",
    )

    plot_roc_from_predictions(
        pred_df=best_predictions,
        out_path=fig_dir / "best_model_roc_curve.png",
    )

    if not best_feature_df.empty:
        plot_top_features(
            feature_df=best_feature_df,
            out_path=fig_dir / "best_model_top_features.png",
            title="Top features from final best model",
            n=20,
        )
    elif not immune_rf_features.empty:
        plot_top_features(
            feature_df=immune_rf_features,
            out_path=fig_dir / "best_model_top_features.png",
            title="Top selected immune-gene features",
            n=20,
        )

    plot_top_features(
        feature_df=pathway_features,
        out_path=fig_dir / "pathway_feature_importance.png",
        title="Immune pathway feature importance",
        n=20,
    )

    plot_pathway_score_boxplots(
        X_pathway=X_pathway,
        y=y,
        out_path=fig_dir / "pathway_score_boxplots.png",
    )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("Done")
    print("=" * 90)

    print(f"Results saved to: {out_dir.resolve()}")

    print("\nMost important files:")
    print(f"  {summary_path}")
    print(f"  {aggregate_predictions_path}")
    print(f"  {out_dir / 'final_best_immune_rf_features.csv'}")
    print(f"  {out_dir / 'final_pathway_feature_importance.csv'}")
    print(f"  {fig_dir}")


if __name__ == "__main__":
    main()