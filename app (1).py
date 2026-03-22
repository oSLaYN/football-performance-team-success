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


RANDOM_STATE = 42
LEAKAGE_COLS = ["year", "wins", "draws", "loses"]
RAW_COLS = ["ppda_att", "ppda_def", "oppda_att", "oppda_def"]
TARGET_COL = "pts"
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


st.set_page_config(
    page_title="Football Performance Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def resolve_default_data_path() -> Optional[Path]:
    base = Path(__file__).resolve().parent
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


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame, List[str]]:
    df = df.copy()

    required_cols = {"date", "ppda_coef", "oppda_coef", TARGET_COL}
    missing_required = sorted(required_cols - set(df.columns))
    if missing_required:
        raise ValueError(
            "O ficheiro não contém as colunas obrigatórias: " + ", ".join(missing_required)
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["month"] = df["date"].dt.month
    df["season_half"] = (df["date"].dt.month > 6).astype(int)

    positive_ppda = df.loc[df["ppda_coef"] > 0, "ppda_coef"]
    positive_oppda = df.loc[df["oppda_coef"] > 0, "oppda_coef"]

    if not positive_ppda.empty:
        ppda_median = positive_ppda.median()
        df["ppda_coef"] = df["ppda_coef"].replace(0, ppda_median)
    if not positive_oppda.empty:
        oppda_median = positive_oppda.median()
        df["oppda_coef"] = df["oppda_coef"].replace(0, oppda_median)

    numerical_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in LEAKAGE_COLS + RAW_COLS
    ]
    categorical_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns if c != "date"
    ]

    corr_matrix = df[numerical_cols].corr(numeric_only=True)
    if TARGET_COL not in corr_matrix.columns:
        raise ValueError("Não foi possível calcular a matriz de correlação com a variável-alvo 'pts'.")

    top_num_cols = corr_matrix[TARGET_COL].abs().nlargest(5).index.tolist()
    top_num_cols = [c for c in top_num_cols if c != TARGET_COL][:4]

    return df, numerical_cols, categorical_cols, corr_matrix, top_num_cols


def build_team_season(df: pd.DataFrame) -> pd.DataFrame:
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
    team_season["win_rate"] = team_season["total_wins"] / team_season["games_played"]
    team_season["xG_diff"] = team_season["avg_xG"] - team_season["avg_xGA"]
    team_season["pts_per_game"] = team_season["total_pts"] / team_season["games_played"]
    return team_season


def run_ml_pipeline(df: pd.DataFrame) -> Dict[str, object]:
    available_features = [c for c in ML_FEATURES if c in df.columns]
    if len(available_features) < 2:
        raise ValueError("Não existem features suficientes para treinar os modelos.")

    X = df[available_features].copy()
    y = (df[TARGET_COL] >= 1).astype(int)

    if X.isnull().any().any():
        X = X.fillna(X.median(numeric_only=True))

    if len(np.unique(y)) < 2:
        raise ValueError("Os dados atuais só têm uma classe na variável-alvo. Não é possível treinar os modelos.")

    if len(df) < 50:
        raise ValueError("O subconjunto de dados é demasiado pequeno para avaliar os modelos de forma fiável.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=500)
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

    lr.fit(X_train_sc, y_train)
    rf.fit(X_train, y_train)

    lr_preds = lr.predict(X_test_sc)
    rf_preds = rf.predict(X_test)
    lr_proba = lr.predict_proba(X_test_sc)[:, 1]
    rf_proba = rf.predict_proba(X_test)[:, 1]

    lr_report = pd.DataFrame(classification_report(y_test, lr_preds, output_dict=True)).T
    rf_report = pd.DataFrame(classification_report(y_test, rf_preds, output_dict=True)).T

    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
    auc_lr = auc(fpr_lr, tpr_lr)
    auc_rf = auc(fpr_rf, tpr_rf)

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


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    leagues = sorted(df["league"].dropna().unique().tolist()) if "league" in df.columns else []
    years = sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else []
    teams = sorted(df["team"].dropna().unique().tolist()) if "team" in df.columns else []

    st.sidebar.header("Filtros")
    selected_leagues = st.sidebar.multiselect("Liga", leagues, default=leagues)
    selected_years = st.sidebar.multiselect("Época / Ano", years, default=years)
    selected_teams = st.sidebar.multiselect("Equipa", teams, default=teams)

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


def plot_missing_values(df: pd.DataFrame) -> Optional[go.Figure]:
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
    if len(top_num_cols) < 1:
        return None

    cols_to_use = top_num_cols[:4]
    rows = 2
    cols = 2
    subplot_titles = [f"{col} vs {TARGET_COL}" for col in cols_to_use]

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
    available = [c for c in top_num_cols if c in df.columns]
    if not available:
        return None

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
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Logistic Regression", "Random Forest"],
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]],
    )

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
    fpr_lr, tpr_lr = results["roc_lr"]
    fpr_rf, tpr_rf = results["roc_rf"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr_lr,
            y=tpr_lr,
            mode="lines",
            name=f"Logistic Regression (AUC={results['auc_lr']:.3f})",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fpr_rf,
            y=tpr_rf,
            mode="lines",
            name=f"Random Forest (AUC={results['auc_rf']:.3f})",
            line=dict(width=2),
        )
    )
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


def build_variable_dictionary() -> pd.DataFrame:
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
    return df.to_csv(index=False).encode("utf-8")


def main() -> None:
    st.title("⚽ Dashboard de Desempenho no Futebol")
    st.markdown(
        "Este dashboard foi gerado a partir do ficheiro `main.ipynb`, mantendo a mesma lógica de preparação, análise e modelação."
    )

    st.sidebar.header("Fonte de dados")
    uploaded_file = st.sidebar.file_uploader(
        "Carregar CSV (opcional)",
        type=["csv"],
        help="Se não carregar um ficheiro, a aplicação tenta usar `data/understat_per_game.csv`.",
    )

    raw_df: Optional[pd.DataFrame] = None
    data_origin = ""

    if uploaded_file is not None:
        raw_df = load_uploaded_data(uploaded_file)
        data_origin = f"Ficheiro carregado: {uploaded_file.name}"
    else:
        default_path = resolve_default_data_path()
        if default_path is not None:
            raw_df = load_data_from_path(str(default_path))
            data_origin = f"Ficheiro local: {default_path}"

    if raw_df is None:
        st.warning(
            "Não foi encontrado o ficheiro `understat_per_game.csv`. Coloque-o na pasta `data/` ou carregue-o manualmente na barra lateral."
        )
        st.stop()

    st.sidebar.success(data_origin)

    try:
        prepared_df, numerical_cols, categorical_cols, corr_matrix, top_num_cols = prepare_data(raw_df)
    except Exception as exc:
        st.error(f"Erro ao preparar os dados: {exc}")
        st.stop()

    filtered_df = apply_filters(prepared_df)

    if filtered_df.empty:
        st.error("Os filtros escolhidos não devolvem registos. Ajuste a seleção na barra lateral.")
        st.stop()

    try:
        team_season_filtered = build_team_season(filtered_df)
    except Exception as exc:
        st.error(f"Erro na agregação por equipa/época: {exc}")
        st.stop()

    st.caption(f"Origem dos dados: {data_origin}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Registos", f"{len(filtered_df):,}")
    col2.metric("Variáveis", f"{filtered_df.shape[1]}")
    col3.metric("Equipas", f"{filtered_df['team'].nunique() if 'team' in filtered_df.columns else 0}")
    col4.metric("Ligas", f"{filtered_df['league'].nunique() if 'league' in filtered_df.columns else 0}")
    col5.metric(
        "Período",
        f"{filtered_df['date'].min().date()} → {filtered_df['date'].max().date()}",
    )

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
                        filtered_df.shape[1],
                        filtered_df['team'].nunique() if 'team' in filtered_df.columns else 0,
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

        missing_fig = plot_missing_values(filtered_df)
        st.subheader("Valores em falta")
        if missing_fig is None:
            st.success("Não foram encontrados valores em falta no subconjunto atual.")
        else:
            st.plotly_chart(missing_fig, use_container_width=True)

        st.download_button(
            "Descarregar dados filtrados (CSV)",
            data=dataframe_to_csv_bytes(filtered_df),
            file_name="football_filtered_data.csv",
            mime="text/csv",
        )

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

    with tab3:
        st.subheader("Matriz de correlação completa")
        st.plotly_chart(plot_correlation_heatmap(corr_matrix), use_container_width=True)

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

    with tab4:
        st.subheader("Agregação por equipa e época")
        st.markdown(
            "Nesta secção, os dados ao nível do jogo são agregados ao nível da época por clube, tal como no notebook original."
        )

        top_n = st.slider("Número de clubes a apresentar", min_value=5, max_value=20, value=15)
        st.plotly_chart(plot_top_clubs(team_season_filtered, top_n), use_container_width=True)

        st.subheader("xG médio vs pontos totais")
        st.plotly_chart(plot_xg_vs_points(team_season_filtered), use_container_width=True)

        st.subheader("Melhores épocas por pontos totais")
        st.dataframe(
            team_season_filtered.sort_values("total_pts", ascending=False).head(20),
            use_container_width=True,
        )

    with tab5:
        st.subheader("Previsão do resultado do jogo")
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

            rep_col1, rep_col2 = st.columns(2)
            with rep_col1:
                st.markdown("#### Classification Report — Logistic Regression")
                st.dataframe(ml_results["lr_report"].round(3), use_container_width=True)
            with rep_col2:
                st.markdown("#### Classification Report — Random Forest")
                st.dataframe(ml_results["rf_report"].round(3), use_container_width=True)

    with tab6:
        st.subheader("Explicação das variáveis utilizadas")
        var_df = build_variable_dictionary()
        available_var_df = var_df[var_df["Variável"].isin(filtered_df.columns)].reset_index(drop=True)
        st.dataframe(available_var_df, use_container_width=True, hide_index=True)
        st.info(
            "A tabela apresenta as principais variáveis usadas no notebook e no dashboard. Se o ficheiro tiver colunas adicionais, estas podem ser exploradas nas tabelas e gráficos interativos."
        )

    with st.expander("Notas técnicas sobre a conversão do notebook para Streamlit"):
        st.markdown(
            "- A lógica do notebook foi preservada: mesmas features, mesma variável-alvo e os mesmos modelos (`LogisticRegression` e `RandomForestClassifier`).\n"
            "- A aplicação foi organizada por separadores para tornar a navegação mais simples.\n"
            "- O projeto já inclui o dataset principal em `data/understat_per_game.csv`, mas pode carregar outro CSV manualmente na barra lateral."
        )


if __name__ == "__main__":
    main()
