from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


## ─────────────────────────────────────────────
## Constantes globais utilizadas em todo o dashboard
## ─────────────────────────────────────────────

## Seed fixa para reprodutibilidade dos modelos de ML
RANDOM_STATE = 42

## Colunas que causam data leakage (contêm informação do resultado)
## e por isso são excluídas das features numéricas
LEAKAGE_COLS = ["year", "wins", "draws", "loses"]

## Colunas brutas do PPDA que foram substituídas por ppda_coef / oppda_coef
RAW_COLS = ["ppda_att", "ppda_def", "oppda_att", "oppda_def"]

## Variável-alvo principal (pontos por jogo)
TARGET_COL = "pts"

## Features selecionadas para os modelos de Machine Learning
ML_FEATURES = [
    "xG",
    "xGA",
    "npxG",
    "npxGA",
    "deep",
    "deep_allowed",
    "npxGD",
    "ppda_coef",
    "oppda_coef",
]


## ─────────────────────────────────────────────
## Configuração da página Streamlit
## ─────────────────────────────────────────────
st.set_page_config(
    page_title="Football Performance Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


## ─────────────────────────────────────────────
## Funções de carregamento de dados
## ─────────────────────────────────────────────


@st.cache_data(show_spinner="A carregar dados…")
def load_data_from_path(path: str) -> pd.DataFrame:
    """Lê um CSV a partir de um caminho local.
    Resultado é guardado em cache pelo Streamlit para evitar
    releituras desnecessárias do disco entre interações."""
    return pd.read_csv(path)


def resolve_default_data_path() -> Optional[Path]:
    """Procura o ficheiro CSV padrão em vários locais possíveis.
    Devolve o primeiro caminho que existir, ou None se nenhum for encontrado."""

    ## Resolve o diretório onde este script se encontra
    base = Path(__file__).resolve().parent

    ## Lista de localizações candidatas, por ordem de prioridade
    candidates = [
        base / "data" / "understat_per_game.csv",
        Path("data/understat_per_game.csv"),
        Path("/mnt/data/data/understat_per_game.csv"),
        Path("/mnt/data/understat_per_game.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@st.cache_data(show_spinner="A carregar ficheiro…")
def load_uploaded_data(_content_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Lê um CSV a partir dos bytes enviados pelo utilizador.
    O underscore em _content_bytes indica ao Streamlit que deve
    usar o hash dos bytes como chave de cache (evita re-parse)."""
    return pd.read_csv(StringIO(_content_bytes.decode("utf-8")))


## ─────────────────────────────────────────────
## Preparação e limpeza dos dados
## ─────────────────────────────────────────────


@st.cache_data(show_spinner="A preparar dados…")
def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame, List[str]]:
    """Limpa e prepara o DataFrame original para análise.

    Passos principais:
    1. Valida a existência das colunas obrigatórias.
    2. Converte a coluna 'date' para datetime, extrai 'month' e 'season_half'.
    3. Substitui zeros em ppda_coef / oppda_coef pela mediana dos valores positivos.
    4. Identifica colunas numéricas e categóricas.
    5. Calcula a matriz de correlação e seleciona as 4 variáveis
       numéricas mais correlacionadas com a variável-alvo (pts).

    Devolve:
        - df preparado
        - lista de colunas numéricas
        - lista de colunas categóricas
        - matriz de correlação
        - lista das top 4 colunas numéricas mais correlacionadas com pts
    """
    df = df.copy()

    ## Verificar se as colunas obrigatórias existem no dataset
    required_cols = {"date", "ppda_coef", "oppda_coef", TARGET_COL}
    missing_required = sorted(required_cols - set(df.columns))
    if missing_required:
        raise ValueError(
            "O ficheiro não contém as colunas obrigatórias: " + ", ".join(missing_required)
        )

    ## Converter datas e remover linhas com datas inválidas
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    ## Criar features temporais derivadas da data
    df["month"] = df["date"].dt.month
    ## season_half = 0 para jan-jun (1ª metade), 1 para jul-dez (2ª metade)
    df["season_half"] = (df["date"].dt.month > 6).astype(int)

    ## Substituir zeros em ppda_coef e oppda_coef pela mediana dos valores > 0
    ## Zeros nestes coeficientes indicam dados em falta e distorcem a análise
    positive_ppda = df.loc[df["ppda_coef"] > 0, "ppda_coef"]
    positive_oppda = df.loc[df["oppda_coef"] > 0, "oppda_coef"]

    if not positive_ppda.empty:
        ppda_median = positive_ppda.median()
        df["ppda_coef"] = df["ppda_coef"].replace(0, ppda_median)
    if not positive_oppda.empty:
        oppda_median = positive_oppda.median()
        df["oppda_coef"] = df["oppda_coef"].replace(0, oppda_median)

    ## Identificar colunas numéricas, excluindo as que causam data leakage
    ## e as colunas brutas de PPDA (já substituídas pelos coeficientes)
    numerical_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in LEAKAGE_COLS + RAW_COLS
    ]

    ## Identificar colunas categóricas, excluindo 'date' (já convertida)
    categorical_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns if c != "date"
    ]

    ## Calcular a matriz de correlação entre todas as variáveis numéricas
    corr_matrix = df[numerical_cols].corr(numeric_only=True)
    if TARGET_COL not in corr_matrix.columns:
        raise ValueError("Não foi possível calcular a matriz de correlação com a variável-alvo 'pts'.")

    ## Selecionar as 4 variáveis numéricas com maior correlação absoluta com pts
    ## (excluindo a própria variável-alvo da lista)
    top_num_cols = corr_matrix[TARGET_COL].abs().nlargest(5).index.tolist()
    top_num_cols = [c for c in top_num_cols if c != TARGET_COL][:4]

    return df, numerical_cols, categorical_cols, corr_matrix, top_num_cols


## ─────────────────────────────────────────────
## Agregação ao nível equipa / época
## ─────────────────────────────────────────────


@st.cache_data(show_spinner="A agregar dados por equipa/época…")
def build_team_season(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega os dados ao nível do jogo para o nível equipa/época.

    Para cada combinação (team, year, league) calcula:
    - Totais: pontos, vitórias, empates, derrotas, jogos disputados
    - Médias: xG, xGA, npxG, ppda_coef, oppda_coef, deep
    - Métricas derivadas: win_rate, xG_diff, pts_per_game
    """

    ## Verificar que todas as colunas necessárias para a agregação existem
    required = {
        "team",
        "year",
        "league",
        "pts",
        "xG",
        "xGA",
        "npxG",
        "ppda_coef",
        "oppda_coef",
        "deep",
        "wins",
        "draws",
        "loses",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Faltam colunas necessárias para a agregação por equipa/época: " + ", ".join(missing)
        )

    ## Agrupar por equipa, ano e liga, e calcular as estatísticas agregadas
    team_season = (
        df.groupby(["team", "year", "league"])
        .agg(
            total_pts=("pts", "sum"),
            avg_xG=("xG", "mean"),
            avg_xGA=("xGA", "mean"),
            avg_npxG=("npxG", "mean"),
            avg_ppda=("ppda_coef", "mean"),
            avg_oppda=("oppda_coef", "mean"),
            avg_deep=("deep", "mean"),
            total_wins=("wins", "sum"),
            total_draws=("draws", "sum"),
            total_losses=("loses", "sum"),
            games_played=("pts", "count"),
        )
        .reset_index()
    )

    ## Calcular métricas derivadas a partir dos agregados
    team_season["win_rate"] = team_season["total_wins"] / team_season["games_played"]
    team_season["xG_diff"] = team_season["avg_xG"] - team_season["avg_xGA"]
    team_season["pts_per_game"] = team_season["total_pts"] / team_season["games_played"]
    return team_season


## ─────────────────────────────────────────────
## Pipeline de Machine Learning
## ─────────────────────────────────────────────


def run_ml_pipeline(df: pd.DataFrame) -> Dict[str, object]:
    """Treina e avalia dois modelos de classificação (Logistic Regression e Random Forest)
    para prever se a equipa ganhou pontos (pts >= 1) ou não.

    Pipeline:
    1. Seleciona apenas as features disponíveis no dataset.
    2. Cria a variável-alvo binária: 1 se pts >= 1, 0 se pts == 0.
    3. Trata valores em falta com a mediana.
    4. Divide os dados em treino (80%) e teste (20%) com estratificação.
    5. Normaliza as features para Logistic Regression (StandardScaler).
    6. Treina ambos os modelos e calcula métricas de avaliação.

    Devolve um dicionário com todas as métricas, previsões e objetos necessários
    para gerar os gráficos de avaliação.
    """

    ## Filtrar apenas as features que existem no dataset atual
    available_features = [c for c in ML_FEATURES if c in df.columns]
    if len(available_features) < 2:
        raise ValueError("Não existem features suficientes para treinar os modelos.")

    X = df[available_features].copy()

    ## Variável-alvo binária: 1 se a equipa ganhou pelo menos 1 ponto (empate ou vitória)
    y = (df[TARGET_COL] >= 1).astype(int)

    ## Preencher valores em falta com a mediana de cada feature
    if X.isnull().any().any():
        X = X.fillna(X.median(numeric_only=True))

    ## Validações mínimas antes de treinar
    if len(np.unique(y)) < 2:
        raise ValueError("Os dados atuais só têm uma classe na variável-alvo. Não é possível treinar os modelos.")

    if len(df) < 50:
        raise ValueError("O subconjunto de dados é demasiado pequeno para avaliar os modelos de forma fiável.")

    ## Dividir em treino/teste com estratificação para manter a proporção das classes
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    ## StandardScaler para a Logistic Regression (necessita features normalizadas)
    ## O Random Forest não precisa de normalização
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    ## Instanciar e treinar os dois modelos
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=500)
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

    ## LR usa dados normalizados; RF usa dados originais
    lr.fit(X_train_sc, y_train)
    rf.fit(X_train, y_train)

    ## Gerar previsões e probabilidades para o conjunto de teste
    lr_preds = lr.predict(X_test_sc)
    rf_preds = rf.predict(X_test)
    ## Probabilidades da classe positiva (pts >= 1) para as curvas ROC
    lr_proba = lr.predict_proba(X_test_sc)[:, 1]
    rf_proba = rf.predict_proba(X_test)[:, 1]

    ## Relatórios de classificação em formato DataFrame para apresentação
    lr_report = pd.DataFrame(classification_report(y_test, lr_preds, output_dict=True)).T
    rf_report = pd.DataFrame(classification_report(y_test, rf_preds, output_dict=True)).T

    ## Curvas ROC: False Positive Rate vs True Positive Rate
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
    auc_lr = auc(fpr_lr, tpr_lr)
    auc_rf = auc(fpr_rf, tpr_rf)

    ## Importância das features do Random Forest (ordenadas para visualização)
    importance_df = pd.DataFrame(
        {"feature": available_features, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=True)

    return {
        "features": available_features,
        "class_balance": {
            "points": int(y.sum()),
            "loss": int((1 - y).sum()),
            "points_pct": float(y.mean()),
            "loss_pct": float((1 - y).mean()),
        },
        "split_sizes": {"train": len(X_train), "test": len(X_test)},
        "y_test": y_test,
        "lr_preds": lr_preds,
        "rf_preds": rf_preds,
        "lr_proba": lr_proba,
        "rf_proba": rf_proba,
        "lr_report": lr_report,
        "rf_report": rf_report,
        "auc_lr": auc_lr,
        "auc_rf": auc_rf,
        "roc_lr": (fpr_lr, tpr_lr),
        "roc_rf": (fpr_rf, tpr_rf),
        "importance_df": importance_df,
        "accuracy_lr": accuracy_score(y_test, lr_preds),
        "accuracy_rf": accuracy_score(y_test, rf_preds),
    }


## ─────────────────────────────────────────────
## Filtros interativos da barra lateral
## ─────────────────────────────────────────────


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Cria widgets multiselect na barra lateral para filtrar por liga, ano e equipa.
    Por defeito, todos os valores ficam selecionados para replicar o comportamento
    do notebook original. Devolve o DataFrame filtrado."""

    ## Extrair valores únicos e ordená-los para os widgets
    leagues = sorted(df["league"].dropna().unique().tolist()) if "league" in df.columns else []
    years = sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else []
    teams = sorted(df["team"].dropna().unique().tolist()) if "team" in df.columns else []

    st.sidebar.header("Filtros")
    selected_leagues = st.sidebar.multiselect("Liga", leagues, default=leagues)
    selected_years = st.sidebar.multiselect("Época / Ano", years, default=years)
    selected_teams = st.sidebar.multiselect("Equipa", teams, default=teams)

    ## Aplicar cada filtro apenas se houver seleção e a coluna existir
    filtered = df.copy()
    if selected_leagues and "league" in filtered.columns:
        filtered = filtered[filtered["league"].isin(selected_leagues)]
    if selected_years and "year" in filtered.columns:
        filtered = filtered[filtered["year"].isin(selected_years)]
    if selected_teams and "team" in filtered.columns:
        filtered = filtered[filtered["team"].isin(selected_teams)]

    st.sidebar.divider()
    st.sidebar.caption(
        "Por defeito, o dashboard replica a lógica do notebook. Os filtros servem para exploração adicional."
    )
    return filtered


## ─────────────────────────────────────────────
## Funções de visualização — gráficos Plotly
## ─────────────────────────────────────────────


def plot_missing_values(df: pd.DataFrame) -> Optional[go.Figure]:
    """Gera um gráfico de barras com a contagem de valores em falta por coluna.
    Devolve None se não houver valores em falta."""

    ## Contar NaN por coluna e filtrar apenas as que têm valores em falta
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        return None

    fig = go.Figure(
        data=[
            go.Bar(
                x=missing.index,
                y=missing.values,
                text=missing.values,
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="Valores em Falta por Coluna",
        xaxis_title="Coluna",
        yaxis_title="Quantidade",
        xaxis_tickangle=45,
        height=380,
        showlegend=False,
    )
    return fig


def plot_categorical_vs_target(df: pd.DataFrame) -> Optional[go.Figure]:
    """Gera subplots de barras que mostram a média de pontos (pts) para cada
    valor das variáveis categóricas mais relevantes (league, h_a, result).
    Devolve None se nenhuma dessas colunas existir no dataset."""

    ## Selecionar apenas as colunas categóricas de interesse que existem
    top_cat_cols = [c for c in ["league", "h_a", "result"] if c in df.columns]
    if not top_cat_cols:
        return None

    fig = make_subplots(
        rows=1,
        cols=len(top_cat_cols),
        subplot_titles=[f"{col} (média de {TARGET_COL})" for col in top_cat_cols],
        specs=[[{"type": "bar"}] * len(top_cat_cols)],
    )

    for i, col in enumerate(top_cat_cols, start=1):
        ## Calcular contagem e média de pts por categoria
        tab = df.groupby(col)[TARGET_COL].agg(["count", "mean"]).round(2)
        tab = tab.sort_values("mean", ascending=False)
        fig.add_trace(
            go.Bar(
                x=tab.index,
                y=tab["mean"],
                text=tab["mean"].round(2),
                textposition="auto",
                marker_color=px.colors.sequential.Viridis[: len(tab)],
                showlegend=False,
            ),
            row=1,
            col=i,
        )
        fig.update_xaxes(tickangle=45, row=1, col=i)

    fig.update_layout(height=460, title_text="Variáveis Categóricas vs Pontos Médios")
    return fig


def plot_numerical_vs_target(df: pd.DataFrame, top_num_cols: List[str]) -> Optional[go.Figure]:
    """Gera uma grelha 2×2 com boxplots (linha 1) e violin plots (linha 2)
    para as 4 variáveis numéricas mais correlacionadas com pts.

    A posição na grelha é calculada com divisão inteira e módulo:
    - subplot_row = (i // 2) + 1  → alterna entre linha 1 e 2
    - subplot_col = (i % 2) + 1   → alterna entre coluna 1 e 2
    """
    if len(top_num_cols) < 1:
        return None

    ## Usar no máximo 4 variáveis para a grelha 2×2
    cols_to_use = top_num_cols[:4]
    rows = 2
    cols = 2
    subplot_titles = [f"{col} vs {TARGET_COL}" for col in cols_to_use]

    ## Linha 1 = boxplots, Linha 2 = violin plots
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        specs=[
            [{"type": "box"}, {"type": "box"}],
            [{"type": "violin"}, {"type": "violin"}],
        ],
    )

    colors = px.colors.qualitative.Set2
    for i, col in enumerate(cols_to_use):
        ## Calcular posição na grelha (divisão inteira para linha, módulo para coluna)
        subplot_row = (i // 2) + 1
        subplot_col = (i % 2) + 1
        is_box_row = subplot_row == 1

        if is_box_row:
            fig.add_trace(
                go.Box(
                    y=df[col],
                    x=df[TARGET_COL].astype(str),
                    name=col,
                    marker_color=colors[i % len(colors)],
                    boxpoints="outliers",
                    showlegend=False,
                ),
                row=subplot_row,
                col=subplot_col,
            )
        else:
            ## Violin plot com mini-boxplot embutido para informação extra
            fig.add_trace(
                go.Violin(
                    y=df[col],
                    x=df[TARGET_COL].astype(str),
                    name=col,
                    marker_color=colors[i % len(colors)],
                    box_visible=True,
                    points="outliers",
                    showlegend=False,
                ),
                row=subplot_row,
                col=subplot_col,
            )

    fig.update_layout(
        height=720,
        title_text="Top Variáveis Numéricas vs Pontos — Boxplot e Violin",
    )
    fig.update_xaxes(tickangle=45)
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Gera um heatmap da matriz de correlação completa.
    Usa a escala RdBu_r centrada em 0 para distinguir correlações
    positivas (vermelho) de negativas (azul)."""

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title=f"Matriz de Correlação — Variável-alvo: {TARGET_COL}",
    )
    fig.update_layout(height=720, font_size=10)
    return fig


def plot_key_metric_distributions(df: pd.DataFrame) -> Optional[go.Figure]:
    """Gera histogramas (linha 1) e boxplots (linha 2) para as métricas
    principais: xG, xGA, pts. Adiciona uma linha vertical a tracejado
    na média (μ) de cada histograma para referência rápida."""

    ## Usar apenas as métricas-chave que existem no dataset
    key_cols = [c for c in ["xG", "xGA", "pts"] if c in df.columns]
    if not key_cols:
        return None

    fig = make_subplots(
        rows=2,
        cols=len(key_cols),
        subplot_titles=key_cols,
        specs=[[{"type": "histogram"}] * len(key_cols), [{"type": "box"}] * len(key_cols)],
        vertical_spacing=0.12,
    )

    colors = px.colors.qualitative.Set3
    for i, col in enumerate(key_cols):
        col_pos = i + 1
        mean_val = df[col].mean()

        ## Histograma na linha 1
        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=col,
                nbinsx=25,
                marker_color=colors[i % len(colors)],
                opacity=0.8,
                showlegend=False,
            ),
            row=1,
            col=col_pos,
        )

        ## Boxplot na linha 2 para ver outliers e quartis
        fig.add_trace(
            go.Box(
                y=df[col],
                name=f"{col}_box",
                marker_color=colors[i % len(colors)],
                showlegend=False,
            ),
            row=2,
            col=col_pos,
        )

        ## Linha vertical a tracejado na média do histograma
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"μ={mean_val:.2f}",
            row=1,
            col=col_pos,
        )

    fig.update_layout(height=620, title_text="Distribuição e Boxplot das Métricas Principais")
    return fig


def plot_top_features_heatmap(df: pd.DataFrame, top_num_cols: List[str]) -> Optional[go.Figure]:
    """Gera um heatmap de correlação reduzido, contendo apenas as variáveis
    numéricas mais correlacionadas com pts e a própria variável-alvo.
    Devolve None se nenhuma das top features existir no dataset."""

    ## Filtrar apenas as colunas que realmente existem
    available = [c for c in top_num_cols if c in df.columns]
    if not available:
        return None

    ## Calcular correlação apenas entre as top features + pts
    top_features_corr = df[available + [TARGET_COL]].corr(numeric_only=True)
    fig = px.imshow(
        top_features_corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlação das Variáveis Mais Relevantes com a Variável-alvo",
    )
    fig.update_layout(height=500)
    return fig


def plot_top_clubs(team_season: pd.DataFrame, top_n: int) -> go.Figure:
    """Gera um gráfico de barras com os top N clubes ordenados pela
    média de pontos por época. A cor das barras reflete o valor (Viridis)."""

    ## Calcular média de pontos entre todas as épocas, por equipa
    top_clubs = (
        team_season.groupby("team")["total_pts"].mean().sort_values(ascending=False).head(top_n)
    )
    fig = px.bar(
        x=top_clubs.index,
        y=top_clubs.values,
        labels={"x": "Equipa", "y": "Média de pontos por época"},
        title=f"Top {top_n} clubes por média de pontos por época",
        color=top_clubs.values,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(xaxis_tickangle=45, showlegend=False, coloraxis_showscale=False, height=500)
    return fig


def plot_xg_vs_points(team_season: pd.DataFrame) -> go.Figure:
    """Gera um scatter plot de xG médio vs pontos totais por época,
    com cores por liga e uma linha de tendência OLS.
    Permite identificar quais equipas sobre/sub-performam face ao xG."""

    fig = px.scatter(
        team_season,
        x="avg_xG",
        y="total_pts",
        color="league",
        hover_name="team",
        hover_data={"year": True, "win_rate": ":.2f"},
        trendline="ols",
        title="xG médio vs pontos totais por época e por liga",
        labels={"avg_xG": "xG médio por jogo", "total_pts": "Pontos totais na época"},
    )
    fig.update_layout(height=560)
    return fig


def plot_confusion_matrices(y_test, lr_preds, rf_preds) -> go.Figure:
    """Gera duas matrizes de confusão lado a lado (Logistic Regression e Random Forest).
    Cada célula mostra o número de previsões corretas/incorretas."""

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Logistic Regression", "Random Forest"],
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]],
    )

    ## Iterar sobre os dois conjuntos de previsões
    for col_pos, preds in enumerate([lr_preds, rf_preds], start=1):
        cm = confusion_matrix(y_test, preds)
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=["Derrota", "Pontos"],
                y=["Derrota", "Pontos"],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            ),
            row=1,
            col=col_pos,
        )

    fig.update_layout(height=420, title_text="Matrizes de Confusão — Conjunto de Teste")
    return fig


def plot_roc_curves(results: Dict[str, object]) -> go.Figure:
    """Gera as curvas ROC para ambos os modelos e uma baseline aleatória (diagonal).
    A AUC (área sob a curva) é indicada na legenda de cada modelo.
    Quanto mais próxima de 1.0, melhor o modelo discrimina as classes."""

    fpr_lr, tpr_lr = results["roc_lr"]
    fpr_rf, tpr_rf = results["roc_rf"]

    fig = go.Figure()

    ## Curva ROC da Logistic Regression
    fig.add_trace(
        go.Scatter(
            x=fpr_lr,
            y=tpr_lr,
            mode="lines",
            name=f"Logistic Regression (AUC={results['auc_lr']:.3f})",
            line=dict(width=2),
        )
    )

    ## Curva ROC do Random Forest
    fig.add_trace(
        go.Scatter(
            x=fpr_rf,
            y=tpr_rf,
            mode="lines",
            name=f"Random Forest (AUC={results['auc_rf']:.3f})",
            line=dict(width=2),
        )
    )

    ## Baseline aleatória (diagonal): um modelo aleatório teria AUC ≈ 0.5
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Baseline aleatória",
        )
    )
    fig.update_layout(
        title="Curva ROC — Previsão do resultado do jogo",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=520,
    )
    return fig


def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    """Gera um gráfico de barras horizontal com a importância de cada feature
    segundo o Random Forest. Features no topo são as mais influentes."""

    fig = go.Figure(
        go.Bar(
            x=importance_df["importance"],
            y=importance_df["feature"],
            orientation="h",
            marker_color=px.colors.sequential.Viridis_r[: len(importance_df)],
        )
    )
    fig.update_layout(
        title="Importância das Features — Random Forest",
        xaxis_title="Importância",
        yaxis_title="Feature",
        height=460,
    )
    return fig


## ─────────────────────────────────────────────
## Dicionário de variáveis e utilitários
## ─────────────────────────────────────────────


def build_variable_dictionary() -> pd.DataFrame:
    """Constrói um DataFrame com o nome e a explicação de cada variável
    utilizada no dataset. Serve como referência rápida no separador 'Variáveis'."""

    rows = [
        ("date", "Data do jogo."),
        ("league", "Liga em que o jogo foi disputado."),
        ("team", "Equipa observada no registo."),
        ("year", "Ano/época associado ao jogo."),
        ("h_a", "Indica se a equipa jogou em casa (home) ou fora (away)."),
        ("result", "Resultado do jogo: vitória, empate ou derrota."),
        ("pts", "Pontos obtidos no jogo (3 vitória, 1 empate, 0 derrota)."),
        ("wins", "Número de vitórias agregadas no registo."),
        ("draws", "Número de empates agregados no registo."),
        ("loses", "Número de derrotas agregadas no registo."),
        ("xG", "Expected Goals da equipa."),
        ("xGA", "Expected Goals Against, ou seja, expected goals concedidos."),
        ("npxG", "Expected Goals sem penáltis."),
        ("npxGA", "Expected Goals Against sem penáltis."),
        ("deep", "Entradas ofensivas em zonas profundas."),
        ("deep_allowed", "Entradas profundas permitidas ao adversário."),
        ("npxGD", "Diferença entre criação e concessão de npxG."),
        ("ppda_coef", "Intensidade de pressão da equipa (passes permitidos por ação defensiva)."),
        ("oppda_coef", "Indicador de pressão do adversário."),
        ("month", "Mês extraído da data do jogo."),
        ("season_half", "Metade da época: 0 para primeira metade, 1 para segunda metade."),
    ]
    return pd.DataFrame(rows, columns=["Variável", "Explicação"])


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Converte um DataFrame em bytes CSV codificados em UTF-8.
    Usado pelo botão de download do Streamlit."""
    return df.to_csv(index=False).encode("utf-8")


## ─────────────────────────────────────────────
## Função principal — composição do dashboard
## ─────────────────────────────────────────────


def main() -> None:
    """Função principal que monta todo o dashboard Streamlit.
    Organiza o conteúdo em 6 separadores:
    1. Visão Geral — resumo, estatísticas, valores em falta
    2. EDA — variáveis categóricas e numéricas vs pts
    3. Correlações — heatmaps e distribuições
    4. Equipas/Épocas — agregações e scatter xG vs pts
    5. Machine Learning — treino e avaliação de modelos
    6. Variáveis — dicionário de dados
    """

    st.title("⚽ Dashboard de Desempenho no Futebol")
    st.markdown(
        "Este dashboard foi gerado a partir do ficheiro `main.ipynb`, mantendo a mesma lógica de preparação, análise e modelação."
    )

    st.sidebar.header("Fonte de dados")

    ## Usar session_state para controlar se o CSV padrão está ativo ou foi removido pelo utilizador
    if "use_default_csv" not in st.session_state:
        st.session_state["use_default_csv"] = True

    raw_df: Optional[pd.DataFrame] = None
    data_origin = ""

    ## Tentar localizar o CSV padrão no sistema de ficheiros
    default_path = resolve_default_data_path()

    if default_path is not None and st.session_state["use_default_csv"]:
        ## CSV padrão encontrado e ativo — carregar (com cache) e mostrar info bloqueada
        raw_df = load_data_from_path(str(default_path))
        data_origin = f"Ficheiro local: {default_path.name}"
        st.sidebar.text_input(
            "Dataset carregado",
            value=str(default_path.name),
            disabled=True,
            help="O dataset original é carregado automaticamente a partir da pasta `data/`.",
        )
        ## Botão para remover o dataset padrão e permitir upload de outro
        if st.sidebar.button("Remover dataset e carregar outro", type="secondary"):
            st.session_state["use_default_csv"] = False
            st.rerun()
    else:
        ## Sem CSV padrão ou o utilizador optou por removê-lo
        if default_path is not None:
            ## O CSV existe mas foi removido — oferecer botão para restaurar
            if st.sidebar.button("Restaurar dataset original", type="primary"):
                st.session_state["use_default_csv"] = True
                st.rerun()

        ## File uploader como fallback para carregar outro CSV
        uploaded_file = st.sidebar.file_uploader(
            "Carregar CSV",
            type=["csv"],
            help="Carregue um ficheiro CSV com a mesma estrutura do dataset original.",
        )
        if uploaded_file is not None:
            ## Ler os bytes e passar para a função cached
            content = uploaded_file.getvalue()
            raw_df = load_uploaded_data(content, uploaded_file.name)
            data_origin = f"Ficheiro carregado: {uploaded_file.name}"

    if raw_df is None:
        st.warning(
            "Nenhum dataset ativo. Carregue um ficheiro CSV na barra lateral ou restaure o dataset original."
        )
        st.stop()

    st.sidebar.success(data_origin)

    ## Preparar os dados (limpeza, features derivadas, correlações)
    try:
        prepared_df, numerical_cols, categorical_cols, corr_matrix, top_num_cols = prepare_data(raw_df)
    except Exception as exc:
        st.error(f"Erro ao preparar os dados: {exc}")
        st.stop()

    ## Aplicar filtros interativos da barra lateral
    filtered_df = apply_filters(prepared_df)

    if filtered_df.empty:
        st.error("Os filtros escolhidos não devolvem registos. Ajuste a seleção na barra lateral.")
        st.stop()

    ## Agregar dados ao nível equipa/época para o separador 4
    try:
        team_season_filtered = build_team_season(filtered_df)
    except Exception as exc:
        st.error(f"Erro na agregação por equipa/época: {exc}")
        st.stop()

    st.caption(f"Origem dos dados: {data_origin}")

    ## ── Métricas de resumo no topo ──
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Registos", f"{len(filtered_df):,}")
    col2.metric("Variáveis", f"{filtered_df.shape[1]}")
    col3.metric("Equipas", f"{filtered_df['team'].nunique() if 'team' in filtered_df.columns else 0}")
    col4.metric("Ligas", f"{filtered_df['league'].nunique() if 'league' in filtered_df.columns else 0}")
    col5.metric(
        "Período",
        f"{filtered_df['date'].min().date()} → {filtered_df['date'].max().date()}",
    )

    ## ── Separadores principais do dashboard ──
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Visão Geral",
            "EDA",
            "Correlações",
            "Equipas / Épocas",
            "Machine Learning",
            "Variáveis",
        ]
    )

    ## ── Tab 1: Visão Geral ──
    with tab1:
        st.subheader("Resumo do dataset")
        st.markdown(
            "- O notebook original trabalha com observações por jogo.\n"
            "- A aplicação preserva a limpeza inicial: parsing de datas, criação de `month` e `season_half`, e correção dos zeros em `ppda_coef` e `oppda_coef`.\n"
            "- A coluna-alvo mantém-se como `pts` para análise e como alvo binário na secção de Machine Learning."
        )

        info_col1, info_col2 = st.columns([1.25, 1])
        with info_col1:
            st.dataframe(filtered_df.head(15), use_container_width=True)
        with info_col2:
            ## Todos os valores são strings para evitar ArrowTypeError (tipos mistos)
            summary_df = pd.DataFrame(
                {
                    "Indicador": [
                        "Total de observações",
                        "Total de features",
                        "Total de equipas únicas",
                        "Intervalo temporal",
                    ],
                    "Valor": [
                        f"{len(filtered_df):,}",
                        str(filtered_df.shape[1]),
                        str(filtered_df['team'].nunique() if 'team' in filtered_df.columns else 0),
                        f"{filtered_df['date'].min().date()} → {filtered_df['date'].max().date()}",
                    ],
                }
            )
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

        st.subheader("Estatísticas numéricas")
        st.dataframe(filtered_df[numerical_cols].describe().T.round(3), use_container_width=True)

        if categorical_cols:
            st.subheader("Estatísticas categóricas")
            st.dataframe(filtered_df[categorical_cols].describe().T, use_container_width=True)

        ## Gráfico de valores em falta (ou mensagem de sucesso se não houver)
        missing_fig = plot_missing_values(filtered_df)
        st.subheader("Valores em falta")
        if missing_fig is None:
            st.success("Não foram encontrados valores em falta no subconjunto atual.")
        else:
            st.plotly_chart(missing_fig, use_container_width=True)

        ## Botão para descarregar os dados filtrados em CSV
        st.download_button(
            "Descarregar dados filtrados (CSV)",
            data=dataframe_to_csv_bytes(filtered_df),
            file_name="football_filtered_data.csv",
            mime="text/csv",
        )

    ## ── Tab 2: Análise Exploratória (EDA) ──
    with tab2:
        st.subheader("Variáveis categóricas vs pontos")
        cat_fig = plot_categorical_vs_target(filtered_df)
        if cat_fig is not None:
            st.plotly_chart(cat_fig, use_container_width=True)
            st.markdown(
                "**Leitura rápida:** em geral, jogar em casa tende a gerar mais pontos, enquanto o resultado do jogo explica diretamente a pontuação obtida."
            )
        else:
            st.info("Não existem colunas categóricas suficientes para este gráfico.")

        st.subheader("Variáveis numéricas com maior relação com `pts`")
        num_fig = plot_numerical_vs_target(filtered_df, top_num_cols)
        if num_fig is not None:
            st.plotly_chart(num_fig, use_container_width=True)
            st.markdown(
                "**Leitura rápida:** métricas como xG, xGA, xpts ou golos marcados/concedidos tendem a separar bem vitórias, empates e derrotas."
            )
        else:
            st.info("Não foi possível calcular as variáveis numéricas mais relevantes.")

    ## ── Tab 3: Correlações ──
    with tab3:
        st.subheader("Matriz de correlação completa")
        st.plotly_chart(plot_correlation_heatmap(corr_matrix), use_container_width=True)

        ## Tabela com as 5 variáveis mais correlacionadas (excluindo pts consigo mesma)
        strong_corr = corr_matrix[TARGET_COL].abs().sort_values(ascending=False)
        top_corr_table = strong_corr.iloc[1:6].reset_index()
        top_corr_table.columns = ["Variável", "Correlação absoluta com pts"]
        st.dataframe(top_corr_table, hide_index=True, use_container_width=True)

        dist_fig = plot_key_metric_distributions(filtered_df)
        if dist_fig is not None:
            st.subheader("Distribuição das métricas principais")
            st.plotly_chart(dist_fig, use_container_width=True)

        top_heatmap = plot_top_features_heatmap(filtered_df, top_num_cols)
        if top_heatmap is not None:
            st.subheader("Heatmap das features mais relevantes")
            st.plotly_chart(top_heatmap, use_container_width=True)

    ## ── Tab 4: Equipas / Épocas ──
    with tab4:
        st.subheader("Agregação por equipa e época")
        st.markdown(
            "Nesta secção, os dados ao nível do jogo são agregados ao nível da época por clube, tal como no notebook original."
        )

        ## Slider para escolher quantos clubes mostrar no ranking
        top_n = st.slider("Número de clubes a apresentar", min_value=5, max_value=20, value=15)
        st.plotly_chart(plot_top_clubs(team_season_filtered, top_n), use_container_width=True)

        st.subheader("xG médio vs pontos totais")
        st.plotly_chart(plot_xg_vs_points(team_season_filtered), use_container_width=True)

        st.subheader("Melhores épocas por pontos totais")
        st.dataframe(
            team_season_filtered.sort_values("total_pts", ascending=False).head(20),
            use_container_width=True,
        )

    ## ── Tab 5: Machine Learning ──
    with tab5:
        st.subheader("Previsão do resultado do jogo")

        ## Opção para treinar com dados filtrados ou com o dataset completo
        use_filtered_for_ml = st.checkbox(
            "Treinar modelos apenas com os dados filtrados",
            value=False,
            help="Desativado por defeito para manter o comportamento mais próximo do notebook original.",
        )
        ml_df = filtered_df if use_filtered_for_ml else prepared_df

        try:
            ml_results = run_ml_pipeline(ml_df)
        except Exception as exc:
            st.warning(f"Não foi possível executar a secção de Machine Learning: {exc}")
        else:
            ## Métricas de resumo dos modelos
            bal = ml_results["class_balance"]
            met1, met2, met3, met4 = st.columns(4)
            met1.metric("Train", f"{ml_results['split_sizes']['train']:,}")
            met2.metric("Test", f"{ml_results['split_sizes']['test']:,}")
            met3.metric("Accuracy LR", f"{ml_results['accuracy_lr']:.3f}")
            met4.metric("Accuracy RF", f"{ml_results['accuracy_rf']:.3f}")

            st.caption(
                f"Balanceamento das classes — Pontos: {bal['points']:,} ({bal['points_pct']:.1%}) | Derrotas: {bal['loss']:,} ({bal['loss_pct']:.1%})"
            )
            st.caption("Features utilizadas: " + ", ".join(ml_results["features"]))

            ## Gráficos de avaliação: matrizes de confusão, curvas ROC, importância
            st.plotly_chart(
                plot_confusion_matrices(
                    ml_results["y_test"], ml_results["lr_preds"], ml_results["rf_preds"]
                ),
                use_container_width=True,
            )
            st.plotly_chart(plot_roc_curves(ml_results), use_container_width=True)
            st.plotly_chart(
                plot_feature_importance(ml_results["importance_df"]), use_container_width=True
            )

            ## Relatórios de classificação detalhados lado a lado
            rep_col1, rep_col2 = st.columns(2)
            with rep_col1:
                st.markdown("#### Classification Report — Logistic Regression")
                st.dataframe(ml_results["lr_report"].round(3), use_container_width=True)
            with rep_col2:
                st.markdown("#### Classification Report — Random Forest")
                st.dataframe(ml_results["rf_report"].round(3), use_container_width=True)

    ## ── Tab 6: Dicionário de Variáveis ──
    with tab6:
        st.subheader("Explicação das variáveis utilizadas")
        var_df = build_variable_dictionary()
        ## Filtrar para mostrar apenas as variáveis presentes no dataset atual
        available_var_df = var_df[var_df["Variável"].isin(filtered_df.columns)].reset_index(drop=True)
        st.dataframe(available_var_df, use_container_width=True, hide_index=True)
        st.info(
            "A tabela apresenta as principais variáveis usadas no notebook e no dashboard. Se o ficheiro tiver colunas adicionais, estas podem ser exploradas nas tabelas e gráficos interativos."
        )

    ## ── Notas técnicas (expander no fundo da página) ──
    with st.expander("Notas técnicas sobre a conversão do notebook para Streamlit"):
        st.markdown(
            "- A lógica do notebook foi preservada: mesmas features, mesma variável-alvo e os mesmos modelos (`LogisticRegression` e `RandomForestClassifier`).\n"
            "- A aplicação foi organizada por separadores para tornar a navegação mais simples.\n"
            "- O dataset principal (`data/understat_per_game.csv`) é carregado automaticamente e apresentado de forma fixa na barra lateral. Se não for encontrado, é possível carregar um CSV manualmente.\n"
            "- Todas as funções de carregamento e preparação de dados utilizam `@st.cache_data` para evitar re-processamento desnecessário entre interações."
        )


if __name__ == "__main__":
    main()
