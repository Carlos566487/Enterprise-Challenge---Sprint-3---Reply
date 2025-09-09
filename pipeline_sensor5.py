#!/usr/bin/env python3
# coding: utf-8
"""
pipeline_sensor4.py - Pipeline revisado com validação de integridade e enriquecimento de features.
Agora suporta múltiplos CSVs: leituras, maquinas, funcionarios e manutencao.
"""

import argparse
import os
import sys
import io
import logging
import unicodedata
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline_sensor4")

# --- utils ---
def slugify_colname(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip().replace(" ", "_").replace("-", "_")
    s = s.replace(".", "").replace(",", "")
    return s

CANONICAL_MAP = {
    "vibracao": "vibracao",
    "temperatura": "temperatura",
    "velocidade_motor": "velocidade_motor",
    "dias_ultima_manutencao": "dias_ultima_manutencao",
    "falha": "falha",
    "id_sensor": "id_sensor",
    "id_maquina": "id_maquina",
    "ts": "ts",
    "timestamp": "ts",
    "modelo": "modelo",
    "tipo": "tipo",
    "id_funcionario": "id_funcionario",
    "cargo": "cargo",
    "idade": "idade",
    "ultima_manutencao": "ultima_manutencao",
    "data_manutencao": "data_manutencao"
}

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        slug = slugify_colname(c)
        if slug in CANONICAL_MAP:
            new_cols[c] = CANONICAL_MAP[slug]
        else:
            new_cols[c] = slug
    return df.rename(columns=new_cols)

def robust_read_csv(path, sep=","):
    encodings = ["utf-8","utf-8-sig","utf-16","cp1252","latin-1"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, engine="python", sep=sep, on_bad_lines="warn")
            logger.info("Arquivo %s lido com encoding=%s shape=%s", path, enc, df.shape)
            return df
        except Exception:
            continue
    with open(path,"rb") as f:
        raw = f.read()
    for enc in encodings:
        try:
            text = raw.decode(enc)
        except Exception:
            continue
        try:
            df = pd.read_csv(io.StringIO(text), engine="python", sep=sep, on_bad_lines="warn")
            logger.info("Arquivo %s lido via decode %s shape=%s", path, enc, df.shape)
            return df
        except Exception:
            continue
    raise RuntimeError(f"Não foi possível ler o CSV {path}")

def map_falha_to_int(s):
    s = s.fillna(0)
    if s.dtype == bool:
        return s.astype(int)
    ss = s.astype(str).str.strip().str.lower()
    mapping = {"true":1,"false":0,"1":1,"0":0,"sim":1,"nao":0,"n":0}
    return ss.map(mapping).fillna(0).astype(int)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# --- validação e enriquecimento ---
def validate_and_enrich(df_leituras, df_maquinas=None, df_funcionarios=None, df_manutencao=None):
    # Validação de domínio
    if "vibracao" in df_leituras.columns:
        df_leituras = df_leituras[(df_leituras["vibracao"]>=0) & (df_leituras["vibracao"]<=100)]
    if "temperatura" in df_leituras.columns:
        df_leituras = df_leituras[(df_leituras["temperatura"]>=-50) & (df_leituras["temperatura"]<=150)]
    if "velocidade_motor" in df_leituras.columns:
        df_leituras = df_leituras[(df_leituras["velocidade_motor"]>=0) & (df_leituras["velocidade_motor"]<=10000)]
    if "falha" in df_leituras.columns:
        df_leituras = df_leituras[df_leituras["falha"].isin([0,1])]

    # Validação referencial: ID_MAQUINA em LEITURA deve existir em MAQUINA_AUTONOMA
    if df_maquinas is not None and "id_maquina" in df_leituras.columns:
        valid_ids = set(df_maquinas["id_maquina"].unique())
        df_leituras = df_leituras[df_leituras["id_maquina"].isna() | df_leituras["id_maquina"].isin(valid_ids)]

    # Enriquecimento: atributos da máquina
    if df_maquinas is not None and "id_maquina" in df_leituras.columns:
        df_leituras = df_leituras.merge(df_maquinas[["id_maquina","modelo","tipo","ultima_manutencao"]],
                                        on="id_maquina", how="left")
        # calcular dias desde ultima manutencao
        if "ts" in df_leituras.columns and "ultima_manutencao" in df_leituras.columns:
            df_leituras["ts"] = pd.to_datetime(df_leituras["ts"], errors="coerce")
            df_leituras["ultima_manutencao"] = pd.to_datetime(df_leituras["ultima_manutencao"], errors="coerce")
            df_leituras["dias_ultima_manutencao"] = (
                (df_leituras["ts"] - df_leituras["ultima_manutencao"]).dt.days
            ).fillna(df_leituras.get("dias_ultima_manutencao", 0))

    # Dados do funcionário via manutenções
    if df_manutencao is not None and df_funcionarios is not None:
        if "id_maquina" in df_manutencao.columns and df_maquinas is not None:
            valid_ids = set(df_maquinas["id_maquina"].unique())
            df_manutencao = df_manutencao[df_manutencao["id_maquina"].isin(valid_ids)]
        if "id_funcionario" in df_manutencao.columns:
            valid_func = set(df_funcionarios["id_funcionario"].unique())
            df_manutencao = df_manutencao[df_manutencao["id_funcionario"].isin(valid_func)]
        df_last_maint = df_manutencao.sort_values("data_manutencao").groupby("id_maquina").tail(1)
        df_last_maint = df_last_maint.merge(df_funcionarios[["id_funcionario","cargo","idade"]],
                                            on="id_funcionario", how="left")
        df_leituras = df_leituras.merge(df_last_maint[["id_maquina","cargo","idade"]], on="id_maquina", how="left")

    return df_leituras

# --- main ---
def main(args):
    outdir = args.outdir
    figs_dir = os.path.join(outdir,'figs')
    rel_dir = os.path.join(outdir,'relatorios')
    ensure_dir(figs_dir); ensure_dir(rel_dir)

    # Leitura do CSV principal (leituras sensores)
    df = robust_read_csv(args.csv)
    df = canonicalize_columns(df)
    if "falha" in df.columns:
        df["falha"] = map_falha_to_int(df["falha"])

    # Ler outros datasets se fornecidos
    df_maquinas = None
    df_funcionarios = None
    df_manutencao = None
    if args.maquinas:
        df_maquinas = canonicalize_columns(robust_read_csv(args.maquinas))
    if args.funcionarios:
        df_funcionarios = canonicalize_columns(robust_read_csv(args.funcionarios))
    if args.manutencao:
        df_manutencao = canonicalize_columns(robust_read_csv(args.manutencao))

    # aplicar validação e enriquecimento
    df = validate_and_enrich(df, df_maquinas, df_funcionarios, df_manutencao)

    # Features
    numeric_features = [c for c in ["vibracao","temperatura","velocidade_motor","dias_ultima_manutencao"] if c in df.columns]
    categorical_features = [c for c in ["modelo","tipo","cargo"] if c in df.columns]

    logger.info("Features numéricas: %s", numeric_features)
    logger.info("Features categóricas: %s", categorical_features)

    # Salvar dataset limpo
    df.to_csv(os.path.join(rel_dir,"dados_sensores_clean.csv"), index=False, encoding="utf-8")

    # Modelo supervisionado se falha existir
    if "falha" in df.columns and df["falha"].nunique()>1:
        X = df[numeric_features+categorical_features].copy()
        y = df["falha"].astype(int)

        # separa numéricas e categóricas
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),("sc", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical_features)
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        clf = Pipeline([("pre", pre),("rf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:,1] if hasattr(clf,"predict_proba") else None

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(os.path.join(rel_dir,"metricas_classificacao.csv"))
        logger.info("Métricas de classificação salvas.")

        cm = confusion_matrix(y_test, y_pred)
        logger.info("Matriz de confusão:\n%s", cm)
        if HAS_MPL:
            plt.figure(); plt.imshow(cm, cmap="Blues"); plt.title("Matriz de Confusão")
            for (i,j),val in np.ndenumerate(cm): plt.text(j,i,val,ha="center",va="center")
            plt.tight_layout(); plt.savefig(os.path.join(figs_dir,"confusion_matrix.png")); plt.close()
            if y_prob is not None:
                plt.figure(); RocCurveDisplay.from_predictions(y_test,y_prob); plt.title("ROC")
                plt.tight_layout(); plt.savefig(os.path.join(figs_dir,"roc_curve.png")); plt.close()
    else:
        logger.info("Coluna falha ausente ou sem variação suficiente.")

    # IsolationForest anomalias
    try:
        Xf = df[numeric_features].fillna(0)
        iso = IsolationForest(n_estimators=200, random_state=42, contamination=0.02)
        scores = -iso.fit(Xf).score_samples(Xf)
        df["anomalia_score"] = scores
        df["anomalia_rank_pct"] = pd.Series(scores).rank(pct=True)
        df["criticidade"] = pd.cut(df["anomalia_rank_pct"], bins=[0,0.75,0.9,0.98,1.0], labels=["Baixo","Médio","Alto","Crítico"])
    except Exception as e:
        logger.exception("Erro ao calcular anomalias: %s", e)

    df.to_csv(os.path.join(rel_dir,"dados_sensores_resultados.csv"), index=False, encoding="utf-8")
    logger.info("Resultados salvos em %s", rel_dir)

    print("Pipeline concluído. Resultados em:", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Arquivo CSV com leituras de sensores")
    parser.add_argument("--maquinas", help="Arquivo CSV com maquinas")
    parser.add_argument("--funcionarios", help="Arquivo CSV com funcionarios")
    parser.add_argument("--manutencao", help="Arquivo CSV com manutencoes")
    parser.add_argument("--outdir", default="saida", help="Diretório de saída")
    args = parser.parse_args()
    main(args)
