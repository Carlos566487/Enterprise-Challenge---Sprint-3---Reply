import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import textwrap

st.set_page_config(page_title="Dashboard Preditivo - Máquinas", layout="wide")

st.title("🎛️ Dashboard de Monitoramento & Manutenção Preditiva")

# Sidebar - inputs / uploads / params
st.sidebar.header("Dados & Filtros")

default_rel = "saida/relatorios/dados_sensores_resultados.csv"
default_metrics = "saida/relatorios/metricas_classificacao.csv"

uploaded_rel = st.sidebar.file_uploader("Upload: resultado (dados_sensores_resultados.csv)", type=["csv"])
uploaded_metrics = st.sidebar.file_uploader("Upload: métricas (metricas_classificacao.csv)", type=["csv"])
uploaded_maquinas = st.sidebar.file_uploader("Upload: maquinas (opcional)", type=["csv"])
uploaded_func = st.sidebar.file_uploader("Upload: funcionarios (opcional)", type=["csv"])
uploaded_manut = st.sidebar.file_uploader("Upload: manutencao (opcional)", type=["csv"])

outdir_input = st.sidebar.text_input("Pasta de saída (se não fizer upload)", value="saida")

# Load data (prefer uploaded files, then default paths)
@st.cache_data
def load_csv(maybe_uploaded, default_path):
    if maybe_uploaded is not None:
        try:
            return pd.read_csv(maybe_uploaded)
        except Exception as e:
            st.sidebar.error(f"Erro ao ler upload: {e}")
            return None
    p = Path(default_path)
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            st.sidebar.error(f"Erro ao ler {default_path}: {e}")
            return None
    return None

df = load_csv(uploaded_rel, f"{outdir_input}/relatorios/dados_sensores_resultados.csv" if outdir_input else default_rel)
metrics_df = load_csv(uploaded_metrics, f"{outdir_input}/relatorios/metricas_classificacao.csv" if outdir_input else default_metrics)
df_maquinas = load_csv(uploaded_maquinas, f"{outdir_input}/relatorios/maquinas.csv")
df_func = load_csv(uploaded_func, f"{outdir_input}/relatorios/funcionarios.csv")
df_manut = load_csv(uploaded_manut, f"{outdir_input}/relatorios/manutencao.csv")

if df is None:
    st.error("Nenhum arquivo de resultados encontrado. Execute o pipeline (pipeline_sensor5.py) ou faça upload de 'dados_sensores_resultados.csv'.")
    st.stop()

# Normalize columns (lowercase keys to be tolerant)
df.columns = [c.strip() for c in df.columns]
col_map = {c: c for c in df.columns}
# Common names mapping
aliases = {
    'ts': ['ts','timestamp','time','date','data'],
    'id_maquina': ['id_maquina','maquina','machine','idmachine'],
    'id_sensor': ['id_sensor','sensor','sensor_id'],
    'vibracao': ['vibracao','vibração','vibration'],
    'temperatura': ['temperatura','temperature'],
    'velocidade_motor': ['velocidade_motor','velocidade','rpm'],
    'dias_ultima_manutencao': ['dias_ultima_manutencao','dias_ultima','dias'],
    'falha': ['falha','failure','fault'],
    'anomalia_score': ['anomalia_score','anomaly_score','score'],
    'criticidade': ['criticidade','criticity','criticality'],
    'modelo': ['modelo','model'],
    'tipo': ['tipo','type'],
    'cargo': ['cargo','role'],
    'idade': ['idade','age']
}
# helper to find a column
def find_col(df, candidates):
    cols = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in cols:
            return df.columns[cols.index(cand.lower())]
    # fuzzy contains
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

col_ts = find_col(df, aliases['ts'])
col_machine = find_col(df, aliases['id_maquina'])
col_sensor = find_col(df, aliases['id_sensor'])
col_vib = find_col(df, aliases['vibracao'])
col_temp = find_col(df, aliases['temperatura'])
col_vel = find_col(df, aliases['velocidade_motor'])
col_dias = find_col(df, aliases['dias_ultima_manutencao'])
col_falha = find_col(df, aliases['falha'])
col_score = find_col(df, aliases['anomalia_score'])
col_crit = find_col(df, aliases['criticidade'])

# parse timestamp
if col_ts:
    df[col_ts] = pd.to_datetime(df[col_ts], errors='coerce')

# Sidebar filters
st.sidebar.markdown("## Filtros")
min_date = df[col_ts].min() if col_ts else None
max_date = df[col_ts].max() if col_ts else None
if col_ts and min_date is not None:
    start_date = st.sidebar.date_input("Data início", value=min_date.date(), min_value=min_date.date(), max_value=max_date.date())
    end_date = st.sidebar.date_input("Data fim", value=max_date.date(), min_value=min_date.date(), max_value=max_date.date())
else:
    start_date = None; end_date = None

machines = sorted(df[col_machine].unique().tolist()) if col_machine else []
sel_machine = st.sidebar.multiselect("Máquinas (vazio = todas)", options=machines, default=[])
crit_levels = sorted(df[col_crit].dropna().unique().tolist()) if col_crit else ['Baixo','Médio','Alto','Crítico']
sel_crit = st.sidebar.multiselect("Criticidade", options=crit_levels, default=crit_levels)
sel_cargo = None
if df_func is not None and 'cargo' in df_func.columns:
    cargos = sorted(df_func['cargo'].dropna().unique().tolist())
    sel_cargo = st.sidebar.multiselect("Cargo (responsável manutenção)", options=cargos, default=[])

# Apply filters
mask = pd.Series(True, index=df.index)
if col_ts and start_date is not None:
    mask = mask & (df[col_ts].dt.date >= start_date) & (df[col_ts].dt.date <= end_date)
if sel_machine:
    mask = mask & df[col_machine].isin(sel_machine)
if col_crit and sel_crit:
    mask = mask & df[col_crit].isin(sel_crit)

dff = df[mask].copy()

# ======= KPIs =======
st.header("Visão Geral (Painel de Controle)")
k1, k2, k3, k4 = st.columns(4)

with k1:
    pct_fail = (dff[col_falha].sum()/len(dff))*100 if (col_falha and len(dff)>0) else 0.0
    st.metric("Falhas no período (%)", f"{pct_fail:.2f}%")

with k2:
    crit_counts = dff[col_crit].value_counts(normalize=True).mul(100).round(2) if col_crit else pd.Series()
    st.metric("Criticidade dominante", crit_counts.idxmax() if not crit_counts.empty else "n/d", f"{crit_counts.max() if not crit_counts.empty else 0:.2f}%")

with k3:
    avg_days = dff[col_dias].mean() if col_dias else np.nan
    st.metric("Tempo médio desde última manutenção (dias)", f"{avg_days:.1f}" if not np.isnan(avg_days) else "n/d")

with k4:
    if metrics_df is not None:
        m = metrics_df.copy()
        if 'Unnamed: 0' in m.columns:
            m = m.set_index('Unnamed: 0')
        try:
            if '1' in m.index.astype(str):
                prec = float(m.loc['1','precision']); rec = float(m.loc['1','recall'])
            elif 'macro avg' in m.index:
                prec = float(m.loc['macro avg','precision']); rec = float(m.loc['macro avg','recall'])
            else:
                if 'accuracy' in m.index:
                    prec = rec = float(m.loc['accuracy','precision']) if 'precision' in m.columns else 0.0
                else:
                    prec = rec = 0.0
            st.metric("Modelo - Precision/Recall", f"{prec:.2f}", delta=f"Recall: {rec:.2f}")
        except Exception:
            st.metric("Modelo - Precision/Recall", "n/d")
    else:
        st.metric("Modelo - Precision/Recall", "n/d")

st.markdown("---")

# ======= Criticidade Donut =======
st.subheader("Distribuição de Criticidade")
if col_crit:
    crit_summary = dff[col_crit].value_counts().reset_index()
    crit_summary.columns = ['criticidade','qtd']
    fig = px.pie(crit_summary, names='criticidade', values='qtd', hole=0.45, title="Proporção por criticidade")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Coluna de criticidade não encontrada nos resultados.")

st.markdown("---")

# ======= Análise Temporal =======
st.subheader("Análise Temporal")
if col_ts and (col_vib or col_temp or col_vel):
    vars_plot = [c for c in [col_vib, col_temp, col_vel] if c]
    sel_var = st.selectbox("Variável para série temporal", options=vars_plot, format_func=lambda x: x)
    fig = px.line(dff.sort_values(col_ts), x=col_ts, y=sel_var, color=col_machine if col_machine else None, title=f"Série temporal - {sel_var}")
    if col_falha:
        failures = dff[dff[col_falha]==1]
        if not failures.empty:
            fig.add_trace(go.Scatter(x=failures[col_ts], y=failures[sel_var], mode='markers', marker=dict(color='red', size=6), name='Falha'))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Não há colunas de tempo ou variáveis para plotar séries temporais.")

# Dias desde última manutenção vs falha (scatter)
st.subheader("Dias desde última manutenção vs Falhas")
if col_dias and col_falha:
    samp = dff[[col_dias, col_falha, col_machine]].dropna()
    if not samp.empty:
        fig = px.box(samp, x=col_falha, y=col_dias, points='all', title="Dias desde última manutenção por ocorrência de falha (0=sem,1=com)")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.scatter(samp, x=col_dias, y=col_falha, color=col_machine if col_machine else None, title="Scatter: dias vs falha (0/1)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Sem dados suficientes para análise.")
else:
    st.info("Coluna 'dias desde última manutenção' ou 'falha' ausente.")

st.markdown("---")

# ======= Comparativo entre Máquinas =======
st.subheader("Comparativo entre Máquinas")

if col_machine:
    agg_cols = {}
    if col_vib: agg_cols[col_vib] = 'mean'
    if col_temp: agg_cols[col_temp] = 'mean'
    if col_vel: agg_cols[col_vel] = 'mean'
    if col_falha: agg_cols[col_falha] = 'sum'
    if col_score: agg_cols[col_score] = 'mean'

    if agg_cols:
        if col_score:
            sort_col = col_score
        elif col_falha:
            sort_col = col_falha
        else:
            sort_col = list(agg_cols.keys())[0]

        summary = (
            dff.groupby(col_machine)
               .agg(agg_cols)
               .reset_index()
               .sort_values(by=sort_col, ascending=False)
        )

        st.dataframe(summary.style.background_gradient(subset=summary.columns[1:], cmap='Reds'))

        top5 = summary.sort_values(sort_col, ascending=False).head(5)

        st.markdown("**Top 5 máquinas mais críticas**")
        st.table(top5)

        pivot = dff.pivot_table(
            index=col_machine,
            values=[col_vib, col_temp, col_vel, col_score],
            aggfunc='mean'
        )
        st.write("Heatmap (médias por máquina)")
        fig_h = px.imshow(
            pivot.fillna(0).T,
            labels=dict(x="Máquina", y="Variável", color="Valor"),
            x=pivot.index,
            y=pivot.columns
        )
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.warning("Nenhuma métrica selecionada para agregação.")
else:
    st.info("Coluna de id da máquina ausente.")

st.markdown("---")

# ======= Manutenção & Recursos Humanos =======
st.subheader("Manutenção & Responsáveis (quando disponível)")
if df_manut is not None:
    st.markdown("### Histórico de manutenções")
    try:
        df_manut['data_manutencao'] = pd.to_datetime(df_manut['data_manutencao'], errors='coerce')
        manut_by_machine = df_manut.groupby('id_maquina').apply(lambda x: x.sort_values('data_manutencao')[['data_manutencao','id_funcionario']].to_dict('records')).to_dict()
        for m,v in manut_by_machine.items():
            st.markdown(f"**Máquina {m}**")
            for r in v:
                st.write(f"- {r['data_manutencao'].date()} — Responsável: {r.get('id_funcionario')}")
    except Exception as e:
        st.info("Erro processando histórico de manutenção: "+str(e))
else:
    st.info("Arquivo de manutenções não carregado. Faça upload para ver o histórico.")

if df_func is not None:
    st.markdown("### Análise de responsáveis")
    if 'cargo' in df_func.columns:
        cargos = df_func['cargo'].value_counts().reset_index()
        cargos.columns = ['cargo','qtd']
        figc = px.bar(cargos, x='cargo', y='qtd', title='Manutenções por cargo (approx)')
        st.plotly_chart(figc, use_container_width=True)
    if 'idade' in df_func.columns:
        fig_age = px.histogram(df_func, x='idade', nbins=20, title='Distribuição de idade dos responsáveis')
        st.plotly_chart(fig_age, use_container_width=True)

st.markdown("---")

# ======= Predição & Anomalias =======
st.subheader("Predição & Anomalias")
if col_score:
    fig_score = px.histogram(dff, x=col_score, nbins=60, title='Distribuição do anomalia_score')
    st.plotly_chart(fig_score, use_container_width=True)
    if col_crit:
        st.write(dff[col_crit].value_counts())

# Download links for filtered data
st.markdown("---")
st.subheader("Exportar dados filtrados")
csv_export = dff.to_csv(index=False)
st.download_button("Download CSV filtrado", data=csv_export, file_name="dados_filtrados.csv", mime="text/csv")

st.sidebar.markdown("---")
st.sidebar.markdown(textwrap.dedent("""
**Observações & Ações recomendadas**

- Ajustar políticas de manutenção para máquinas com criticidade 'Alto' ou 'Crítico'.
- Definir limites de alarme por variável (vibração/temperatura/RPM).
- Reforçar treinamento para cargos que apresentarem maior reincidência de falhas.
"""))

# ======= Auto-rewrite do script =======
from pathlib import Path

# caminho do script atual
file_path = Path(__file__).parent / "dashboard_avancado.py"

# lê o próprio arquivo
with open(__file__, "r", encoding="utf-8") as original:
    script_content = original.read()

# sobrescreve/salva
with open(file_path, "w", encoding="utf-8") as f:
    f.write(script_content)

print(f"Saved {file_path}")
