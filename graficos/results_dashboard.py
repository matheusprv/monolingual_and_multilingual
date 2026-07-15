"""
Peixoto Results Dashboard

Interactive Streamlit dashboard for comparing causal language model metrics across:
- grammar swap
- lexical swap
- translation

Run:
    streamlit run results_dashboard.py

Expected CSV files in the same folder as this script:
    grammar_swap_causalLM_results_compiled.csv
    lexical_swap_causalLM_results_compiled.csv
    translation_causalLM_results_compiled.csv

The app also lets you upload replacement CSVs from the sidebar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_TITLE = "Interactive Results Dashboard"

PB_LABEL = "PB NATIVO"

DEFAULT_FILES = {
    "Grammar swap": "grammar_swap_causalLM_results_compiled.csv",
    "Lexical swap": "lexical_swap_causalLM_results_compiled.csv",
    "Translation": "translation_causalLM_results_compiled.csv",
}

REQUIRED_COLUMNS = ["Model", "Task", "Language"]

PREFERRED_METRIC_ORDER = [
    "Perplexity",
    "Pseudo-Perplexity",
    "BPB",
    "Tokens/Word",
]

LANGUAGE_ORDER = [
    "PB NATIVO",
    "GALEGO",
    "ESPANHOL",
    "FRANCÊS",
    "INGLÊS",
    "ÁRABE",
    "MANDARIM",
    "RUSSO",
]


@dataclass
class PlotConfig:
    metric: str
    value_col: str
    value_label: str
    data_mode: str
    aggregation: str
    show_std: bool
    log_scale: bool


def clean_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("—", "-")
        .replace("–", "-")
        .replace("/", "_")
        .replace("%", "pct")
    )


@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str, scenario: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Scenario"] = scenario
    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Keep labels consistent even if future files have extra spaces.
    for col in ["Model", "Task", "Language", "Scenario"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Convert metric-like columns to numeric when possible.
    for col in df.columns:
        if col not in ["Model", "Task", "Language", "Scenario"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def available_metrics(df: pd.DataFrame) -> list[str]:
    numeric_cols = [
        col
        for col in df.columns
        if col not in ["Scenario", "Model", "Task", "Language"]
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].notna().any()
    ]

    preferred = [m for m in PREFERRED_METRIC_ORDER if m in numeric_cols]
    remaining = [m for m in numeric_cols if m not in preferred]
    return preferred + remaining


def sort_key(values: Iterable[str], preferred_order: list[str] | None = None) -> list[str]:
    values = list(dict.fromkeys([str(v) for v in values if pd.notna(v)]))
    if preferred_order is None:
        return sorted(values)

    known = [v for v in preferred_order if v in values]
    unknown = sorted([v for v in values if v not in preferred_order])
    return known + unknown


def make_order_maps(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "Scenario": sort_key(df["Scenario"].unique(), list(DEFAULT_FILES.keys())),
        "Language": sort_key(df["Language"].unique(), LANGUAGE_ORDER),
        "Model": sort_key(df["Model"].unique()),
        "Task": sort_key(df["Task"].unique()),
    }


def add_pb_comparisons(df: pd.DataFrame, metrics: list[str], pb_label: str = PB_LABEL) -> pd.DataFrame:
    """
    Adds one column per metric with percentage change against PB NATIVO.

    Baseline grain:
        Scenario + Model + Task

    Formula:
        (value / PB_NATIVO_value - 1) * 100

    This keeps the comparison fair because each row is compared to the PB NATIVO
    result for the same scenario, model, and task.
    """
    out = df.copy()

    key_cols = ["Scenario", "Model", "Task"]
    pb = out[out["Language"] == pb_label][key_cols + metrics].copy()

    # If duplicate PB rows appear in future datasets, average them at the same grain.
    pb = pb.groupby(key_cols, as_index=False)[metrics].mean()

    for metric in metrics:
        pb_metric_col = f"{metric}__PB_BASELINE"
        delta_col = f"{metric}__DELTA_PCT_VS_PB"

        baseline = pb[key_cols + [metric]].rename(columns={metric: pb_metric_col})
        out = out.merge(baseline, on=key_cols, how="left")

        denominator = out[pb_metric_col].replace(0, np.nan)
        out[delta_col] = (out[metric] / denominator - 1.0) * 100.0

    return out


def aggregate_values(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    aggregation: str,
) -> pd.DataFrame:
    working = df.dropna(subset=[value_col]).copy()

    if not group_cols:
        group_cols = ["Scenario"]

    grouped = working.groupby(group_cols, dropna=False)[value_col]

    if aggregation == "Mean":
        center = grouped.mean()
    elif aggregation == "Median":
        center = grouped.median()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    std = grouped.std(ddof=1)
    n = grouped.count()
    min_v = grouped.min()
    max_v = grouped.max()

    result = (
        pd.concat(
            [
                center.rename("Value"),
                std.rename("Std"),
                n.rename("N"),
                min_v.rename("Min"),
                max_v.rename("Max"),
            ],
            axis=1,
        )
        .reset_index()
    )
    result["Std"] = result["Std"].fillna(0)
    return result


def filter_for_log_scale(
    plot_df: pd.DataFrame,
    value_col: str,
    chart_name: str,
) -> pd.DataFrame:
    if value_col not in plot_df.columns:
        return plot_df

    non_positive = plot_df[value_col].le(0) | plot_df[value_col].isna()
    count = int(non_positive.sum())

    if count:
        st.warning(
            f"Log scale needs positive values. {count} non-positive or missing points "
            f"were hidden in the {chart_name}."
        )
        plot_df = plot_df.loc[~non_positive].copy()

    return plot_df


def set_log_axis(fig: go.Figure, axis: str = "y") -> go.Figure:
    if axis == "y":
        fig.update_yaxes(type="log")
    elif axis == "x":
        fig.update_xaxes(type="log")
    return fig


def apply_category_orders(fig: go.Figure, order_maps: dict[str, list[str]]) -> go.Figure:
    for axis_name, axis in [("xaxis", "x"), ("yaxis", "y")]:
        pass

    # Plotly Express already receives category_orders. This helper is kept
    # for future manual figures.
    return fig


def format_figure(fig: go.Figure, title: str, y_label: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        legend_title_text="Group",
        hovermode="closest",
        margin=dict(l=20, r=20, t=70, b=20),
    )

    if y_label:
        fig.update_yaxes(title_text=y_label)

    return fig


def display_downloads(fig: go.Figure, filtered_df: pd.DataFrame, aggregate_df: pd.DataFrame | None, prefix: str) -> None:
    st.divider()
    st.subheader("Export")

    html = fig.to_html(include_plotlyjs="cdn")
    st.download_button(
        "Download current chart as HTML",
        data=html,
        file_name=f"{prefix}.html",
        mime="text/html",
    )

    try:
        pdf_bytes = fig.to_image(format="pdf")
        st.download_button(
            "Download current chart as PDF",
            data=pdf_bytes,
            file_name=f"{prefix}.pdf",
            mime="application/pdf",
        )
    except Exception:
        st.caption(
            "PDF export needs the `kaleido` package. It is included in the requirements file; "
            "install it if this button does not appear in your environment."
        )

    st.download_button(
        "Download filtered data as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{prefix}_filtered_data.csv",
        mime="text/csv",
    )

    if aggregate_df is not None:
        st.download_button(
            "Download aggregated values as CSV",
            data=aggregate_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{prefix}_aggregated.csv",
            mime="text/csv",
        )


def load_data_ui() -> pd.DataFrame:
    st.sidebar.header("Data")

    script_dir = Path(__file__).resolve().parent
    use_uploads = st.sidebar.checkbox("Upload CSVs manually", value=False)

    frames: list[pd.DataFrame] = []

    if not use_uploads:
        missing = []
        for scenario, filename in DEFAULT_FILES.items():
            path = script_dir / filename
            if path.exists():
                frames.append(load_csv_from_path(str(path), scenario))
            else:
                missing.append(filename)

        if missing:
            st.sidebar.warning(
                "Some default files were not found next to the script: "
                + ", ".join(missing)
                + ". Use manual upload or place the files in the same folder."
            )

    if use_uploads or not frames:
        st.sidebar.caption("Upload each scenario CSV.")
        uploaded_items = {
            "Grammar swap": st.sidebar.file_uploader("Grammar swap CSV", type="csv", key="grammar_csv"),
            "Lexical swap": st.sidebar.file_uploader("Lexical swap CSV", type="csv", key="lexical_csv"),
            "Translation": st.sidebar.file_uploader("Translation CSV", type="csv", key="translation_csv"),
        }

        frames = []
        for scenario, uploaded_file in uploaded_items.items():
            if uploaded_file is not None:
                temp = pd.read_csv(uploaded_file)
                temp["Scenario"] = scenario
                frames.append(temp)

    if not frames:
        st.stop()

    df = pd.concat(frames, ignore_index=True)
    return normalize_dataframe(df)


def build_sidebar_filters(df: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, PlotConfig, dict[str, list[str]]]:
    order_maps = make_order_maps(df)

    st.sidebar.header("Filters")

    selected_scenarios = st.sidebar.multiselect(
        "Scenarios",
        options=order_maps["Scenario"],
        default=order_maps["Scenario"],
    )
    selected_models = st.sidebar.multiselect(
        "Models",
        options=order_maps["Model"],
        default=order_maps["Model"],
    )
    selected_languages = st.sidebar.multiselect(
        "Languages",
        options=order_maps["Language"],
        default=order_maps["Language"],
    )
    selected_tasks = st.sidebar.multiselect(
        "Tasks",
        options=order_maps["Task"],
        default=order_maps["Task"],
    )

    metric = st.sidebar.selectbox("Metric", options=metrics, index=0)

    data_mode = st.sidebar.radio(
        "Display values as",
        options=["Raw metric", "Δ% against PB NATIVO"],
        index=0,
        help="Δ% is calculated as (Language value / PB NATIVO value - 1) × 100 for the same Scenario + Model + Task.",
    )

    aggregation = st.sidebar.radio(
        "Aggregation",
        options=["Mean", "Median"],
        horizontal=True,
        help="Whenever values are aggregated, the app also computes standard deviation and sample count.",
    )

    show_std = st.sidebar.checkbox(
        "Show standard deviation where possible",
        value=True,
        help="Bar and line charts show std as error bars. Heatmaps include std in hover and can be switched to a std heatmap.",
    )

    log_scale = st.sidebar.checkbox(
        "Use log scale",
        value=False,
        help="Log scale is only meaningful for positive values. Non-positive values are hidden if log scale is enabled.",
    )

    if data_mode == "Raw metric":
        value_col = metric
        value_label = metric
    else:
        value_col = f"{metric}__DELTA_PCT_VS_PB"
        value_label = f"Δ {metric} vs PB NATIVO (%)"

    filtered = df[
        df["Scenario"].isin(selected_scenarios)
        & df["Model"].isin(selected_models)
        & df["Language"].isin(selected_languages)
        & df["Task"].isin(selected_tasks)
    ].copy()

    config = PlotConfig(
        metric=metric,
        value_col=value_col,
        value_label=value_label,
        data_mode=data_mode,
        aggregation=aggregation,
        show_std=show_std,
        log_scale=log_scale,
    )

    return filtered, config, order_maps


def chart_bar(filtered: pd.DataFrame, config: PlotConfig, order_maps: dict[str, list[str]]) -> tuple[go.Figure, pd.DataFrame]:
    st.subheader("Bar chart")

    c1, c2, c3 = st.columns(3)
    with c1:
        x_axis = st.selectbox("X-axis", ["Language", "Model", "Scenario", "Task"], index=0)
    with c2:
        color_by = st.selectbox("Group / color by", ["Scenario", "Model", "Language", "Task", "None"], index=0)
    with c3:
        facet_by = st.selectbox("Facet by", ["None", "Scenario", "Model", "Language", "Task"], index=0)

    group_cols = [x_axis]
    if color_by != "None" and color_by not in group_cols:
        group_cols.append(color_by)
    if facet_by != "None" and facet_by not in group_cols:
        group_cols.append(facet_by)

    agg = aggregate_values(filtered, group_cols, config.value_col, config.aggregation)
    plot_df = agg.rename(columns={config.value_col: "Value"})

    if config.log_scale:
        plot_df = filter_for_log_scale(plot_df, "Value", "bar chart")

    error_y = "Std" if config.show_std else None

    fig = px.bar(
        plot_df,
        x=x_axis,
        y="Value",
        color=None if color_by == "None" else color_by,
        facet_col=None if facet_by == "None" else facet_by,
        barmode="group",
        error_y=error_y,
        hover_data=["Std", "N", "Min", "Max"],
        category_orders=order_maps,
        labels={"Value": config.value_label},
    )

    if config.log_scale:
        set_log_axis(fig, "y")

    fig = format_figure(
        fig,
        title=f"{config.aggregation} {config.value_label} by {x_axis}",
        y_label=config.value_label,
    )
    return fig, agg


def chart_grouped_bar(filtered: pd.DataFrame, config: PlotConfig, order_maps: dict[str, list[str]]) -> tuple[go.Figure, pd.DataFrame]:
    st.subheader("Grouped bar chart")

    c1, c2, c3 = st.columns(3)
    with c1:
        x_axis = st.selectbox("X-axis", ["Scenario", "Language", "Model", "Task"], index=0, key="gb_x")
    with c2:
        group_by = st.selectbox("Grouped by", ["Model", "Language", "Scenario", "Task"], index=0, key="gb_group")
    with c3:
        facet_by = st.selectbox("Facet by", ["None", "Scenario", "Model", "Language", "Task"], index=0, key="gb_facet")

    group_cols = [x_axis]
    if group_by not in group_cols:
        group_cols.append(group_by)
    if facet_by != "None" and facet_by not in group_cols:
        group_cols.append(facet_by)

    agg = aggregate_values(filtered, group_cols, config.value_col, config.aggregation)
    plot_df = agg.copy()

    if config.log_scale:
        plot_df = filter_for_log_scale(plot_df, "Value", "grouped bar chart")

    fig = px.bar(
        plot_df,
        x=x_axis,
        y="Value",
        color=group_by,
        facet_col=None if facet_by == "None" else facet_by,
        barmode="group",
        error_y="Std" if config.show_std else None,
        hover_data=["Std", "N", "Min", "Max"],
        category_orders=order_maps,
        labels={"Value": config.value_label},
    )

    if config.log_scale:
        set_log_axis(fig, "y")

    fig = format_figure(
        fig,
        title=f"{config.aggregation} {config.value_label}: {x_axis} grouped by {group_by}",
        y_label=config.value_label,
    )
    return fig, agg


def chart_heatmap(filtered: pd.DataFrame, config: PlotConfig, order_maps: dict[str, list[str]]) -> tuple[go.Figure, pd.DataFrame]:
    st.subheader("Heatmap")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x_axis = st.selectbox("X-axis", ["Language", "Model", "Scenario", "Task"], index=0, key="hm_x")
    with c2:
        y_axis = st.selectbox("Y-axis", ["Model", "Language", "Scenario", "Task"], index=0, key="hm_y")
    with c3:
        heat_value = st.selectbox(
            "Heatmap value",
            ["Aggregated value", "Standard deviation", "Sample count"],
            index=0,
            key="hm_value",
        )
    with c4:
        show_cell_values = st.checkbox(
            "Show numbers in cells",
            value=True,
            key="hm_show_cell_values",
            help="Writes the selected heatmap value inside each non-empty cell.",
        )

    if x_axis == y_axis:
        st.warning("Choose different fields for the X and Y axes.")
        st.stop()

    group_cols = [x_axis, y_axis]
    agg = aggregate_values(filtered, group_cols, config.value_col, config.aggregation)

    z_col = {
        "Aggregated value": "Value",
        "Standard deviation": "Std",
        "Sample count": "N",
    }[heat_value]

    heat_df = agg.copy()

    if config.log_scale and z_col != "N":
        heat_df = filter_for_log_scale(heat_df, z_col, "heatmap")
        heat_df[f"log10_{z_col}"] = np.log10(heat_df[z_col])
        z_col_for_plot = f"log10_{z_col}"
        color_title = f"log10({config.value_label if z_col == 'Value' else z_col})"
    else:
        z_col_for_plot = z_col
        color_title = config.value_label if z_col == "Value" else z_col

    x_order = order_maps.get(x_axis, sorted(heat_df[x_axis].dropna().unique()))
    y_order = order_maps.get(y_axis, sorted(heat_df[y_axis].dropna().unique()))

    matrix = heat_df.pivot(index=y_axis, columns=x_axis, values=z_col_for_plot).reindex(index=y_order, columns=x_order)
    display_value_matrix = heat_df.pivot(index=y_axis, columns=x_axis, values=z_col).reindex(index=y_order, columns=x_order)
    value_matrix = heat_df.pivot(index=y_axis, columns=x_axis, values="Value").reindex(index=y_order, columns=x_order)
    std_matrix = heat_df.pivot(index=y_axis, columns=x_axis, values="Std").reindex(index=y_order, columns=x_order)
    n_matrix = heat_df.pivot(index=y_axis, columns=x_axis, values="N").reindex(index=y_order, columns=x_order)

    if z_col == "N":
        text_matrix = display_value_matrix.map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
    else:
        text_matrix = display_value_matrix.map(lambda v: "" if pd.isna(v) else f"{v:.3g}")

    customdata = np.dstack(
        [
            value_matrix.to_numpy(dtype=float),
            std_matrix.to_numpy(dtype=float),
            n_matrix.to_numpy(dtype=float),
        ]
    )

    heatmap_kwargs = dict(
        z=matrix.to_numpy(dtype=float),
        x=list(matrix.columns),
        y=list(matrix.index),
        colorbar=dict(title=color_title),
        customdata=customdata,
        hovertemplate=(
            f"{x_axis}: %{{x}}<br>"
            f"{y_axis}: %{{y}}<br>"
            f"{config.aggregation}: %{{customdata[0]:.4g}}<br>"
            "Std: %{customdata[1]:.4g}<br>"
            "N: %{customdata[2]:.0f}<extra></extra>"
        ),
    )

    if show_cell_values:
        heatmap_kwargs.update(
            text=text_matrix.to_numpy(dtype=str),
            texttemplate="%{text}",
            textfont=dict(size=12),
        )

    fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))

    fig.update_layout(
        title=f"{heat_value}: {config.value_label} aggregated over selected rows",
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return fig, agg


def chart_box(filtered: pd.DataFrame, config: PlotConfig, order_maps: dict[str, list[str]]) -> tuple[go.Figure, pd.DataFrame | None]:
    st.subheader("Distribution / box plot")

    c1, c2 = st.columns(2)
    with c1:
        x_axis = st.selectbox("X-axis", ["Language", "Model", "Scenario", "Task"], index=0, key="box_x")
    with c2:
        color_by = st.selectbox("Color by", ["Scenario", "Model", "Language", "Task", "None"], index=0, key="box_color")

    plot_df = filtered.dropna(subset=[config.value_col]).copy()
    plot_df = plot_df.rename(columns={config.value_col: "Value"})

    if config.log_scale:
        plot_df = filter_for_log_scale(plot_df, "Value", "box plot")

    fig = px.box(
        plot_df,
        x=x_axis,
        y="Value",
        color=None if color_by == "None" else color_by,
        points="all",
        hover_data=["Scenario", "Model", "Task", "Language"],
        category_orders=order_maps,
        labels={"Value": config.value_label},
    )

    if config.log_scale:
        set_log_axis(fig, "y")

    fig = format_figure(
        fig,
        title=f"Distribution of {config.value_label} by {x_axis}",
        y_label=config.value_label,
    )

    return fig, None


def chart_line_profile(filtered: pd.DataFrame, config: PlotConfig, order_maps: dict[str, list[str]]) -> tuple[go.Figure, pd.DataFrame]:
    st.subheader("Profile line chart")

    c1, c2, c3 = st.columns(3)
    with c1:
        x_axis = st.selectbox("X-axis", ["Language", "Task", "Scenario", "Model"], index=0, key="line_x")
    with c2:
        color_by = st.selectbox("Line/color by", ["Model", "Language", "Scenario", "Task"], index=0, key="line_color")
    with c3:
        facet_by = st.selectbox("Facet by", ["None", "Scenario", "Model", "Language", "Task"], index=0, key="line_facet")

    group_cols = [x_axis]
    if color_by not in group_cols:
        group_cols.append(color_by)
    if facet_by != "None" and facet_by not in group_cols:
        group_cols.append(facet_by)

    agg = aggregate_values(filtered, group_cols, config.value_col, config.aggregation)
    plot_df = agg.copy()

    if config.log_scale:
        plot_df = filter_for_log_scale(plot_df, "Value", "line chart")

    fig = px.line(
        plot_df,
        x=x_axis,
        y="Value",
        color=color_by,
        facet_col=None if facet_by == "None" else facet_by,
        markers=True,
        error_y="Std" if config.show_std else None,
        hover_data=["Std", "N", "Min", "Max"],
        category_orders=order_maps,
        labels={"Value": config.value_label},
    )

    if config.log_scale:
        set_log_axis(fig, "y")

    fig = format_figure(
        fig,
        title=f"{config.aggregation} {config.value_label} profile by {x_axis}",
        y_label=config.value_label,
    )

    return fig, agg


def metric_value_series(df: pd.DataFrame, metric: str, mode: str) -> pd.Series:
    if mode == "Raw metric":
        return df[metric]
    return df[f"{metric}__DELTA_PCT_VS_PB"]


def chart_scatter(filtered: pd.DataFrame, metrics: list[str], config: PlotConfig, order_maps: dict[str, list[str]]) -> tuple[go.Figure, pd.DataFrame | None]:
    st.subheader("Metric relationship / scatter plot")

    c1, c2, c3 = st.columns(3)
    with c1:
        x_metric = st.selectbox("X metric", metrics, index=min(2, len(metrics) - 1), key="scatter_x_metric")
    with c2:
        y_metric = st.selectbox("Y metric", metrics, index=0, key="scatter_y_metric")
    with c3:
        color_by = st.selectbox("Color by", ["Scenario", "Model", "Language", "Task"], index=0, key="scatter_color")

    size_by = st.selectbox("Point size by", ["None"] + metrics, index=0, key="scatter_size")
    symbol_by = st.selectbox("Symbol by", ["None", "Scenario", "Model", "Language", "Task"], index=0, key="scatter_symbol")
    show_trendline = st.checkbox(
        "Add OLS trendline",
        value=False,
        help="Requires the optional `statsmodels` package. If unavailable, the chart is drawn without the trendline.",
        key="scatter_trendline",
    )

    plot_df = filtered.copy()
    plot_df["X Value"] = metric_value_series(plot_df, x_metric, config.data_mode)
    plot_df["Y Value"] = metric_value_series(plot_df, y_metric, config.data_mode)

    size_col = None
    if size_by != "None":
        plot_df["Size Value"] = metric_value_series(plot_df, size_by, config.data_mode)
        # Plotly cannot use negative sizes. Use absolute values for comparison mode.
        plot_df["Size Value"] = plot_df["Size Value"].abs()
        size_col = "Size Value"

    plot_df = plot_df.dropna(subset=["X Value", "Y Value"]).copy()

    if config.log_scale:
        plot_df = filter_for_log_scale(plot_df, "X Value", "scatter x-axis")
        plot_df = filter_for_log_scale(plot_df, "Y Value", "scatter y-axis")

    x_label = x_metric if config.data_mode == "Raw metric" else f"Δ {x_metric} vs PB NATIVO (%)"
    y_label = y_metric if config.data_mode == "Raw metric" else f"Δ {y_metric} vs PB NATIVO (%)"

    trendline = "ols" if show_trendline and len(plot_df) >= 3 else None

    try:
        fig = px.scatter(
            plot_df,
            x="X Value",
            y="Y Value",
            color=color_by,
            symbol=None if symbol_by == "None" else symbol_by,
            size=size_col,
            hover_data=["Scenario", "Model", "Task", "Language"],
            category_orders=order_maps,
            labels={"X Value": x_label, "Y Value": y_label},
            trendline=trendline,
        )
    except ModuleNotFoundError:
        st.warning("`statsmodels` is not installed, so the OLS trendline was skipped.")
        fig = px.scatter(
            plot_df,
            x="X Value",
            y="Y Value",
            color=color_by,
            symbol=None if symbol_by == "None" else symbol_by,
            size=size_col,
            hover_data=["Scenario", "Model", "Task", "Language"],
            category_orders=order_maps,
            labels={"X Value": x_label, "Y Value": y_label},
        )

    if config.log_scale:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")

    fig = format_figure(
        fig,
        title=f"{y_label} vs {x_label}",
        y_label=y_label,
    )
    fig.update_xaxes(title=x_label)

    return fig, None


def chart_rank_table(filtered: pd.DataFrame, config: PlotConfig) -> tuple[go.Figure, pd.DataFrame]:
    st.subheader("Ranking table")

    c1, c2 = st.columns(2)
    with c1:
        group_by = st.multiselect(
            "Group ranking by",
            ["Scenario", "Model", "Language", "Task"],
            default=["Scenario", "Model", "Language"],
            key="rank_group",
        )
    with c2:
        sort_order = st.radio("Sort order", ["Highest first", "Lowest first"], horizontal=True, key="rank_sort")

    if not group_by:
        group_by = ["Scenario"]

    agg = aggregate_values(filtered, group_by, config.value_col, config.aggregation)
    ascending = sort_order == "Lowest first"
    table = agg.sort_values("Value", ascending=ascending).reset_index(drop=True)
    table.insert(0, "Rank", np.arange(1, len(table) + 1))

    st.dataframe(table, use_container_width=True, hide_index=True)

    top = table.head(min(25, len(table))).copy()
    label_col = top[group_by].astype(str).agg(" | ".join, axis=1)

    plot_df = pd.DataFrame(
        {
            "Group": label_col,
            "Value": top["Value"],
            "Std": top["Std"],
            "N": top["N"],
        }
    )

    if config.log_scale:
        plot_df = filter_for_log_scale(plot_df, "Value", "ranking chart")

    fig = px.bar(
        plot_df,
        x="Value",
        y="Group",
        orientation="h",
        error_x="Std" if config.show_std else None,
        hover_data=["Std", "N"],
        labels={"Value": config.value_label},
    )

    if config.log_scale:
        fig.update_xaxes(type="log")

    fig.update_layout(
        title=f"Top {len(top)} ranked groups",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return fig, table


def curiosity_panel(filtered: pd.DataFrame, config: PlotConfig) -> None:
    st.subheader("Data summary")

    non_missing = filtered[config.value_col].notna().sum() if config.value_col in filtered else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows selected", f"{len(filtered):,}")
    c2.metric("Non-missing metric rows", f"{non_missing:,}")
    c3.metric("Models", f"{filtered['Model'].nunique():,}")
    c4.metric("Languages", f"{filtered['Language'].nunique():,}")

    if non_missing == 0:
        st.warning("No rows have values for this metric/mode after filtering.")
        return

    if config.data_mode == "Δ% against PB NATIVO":
        values = filtered.dropna(subset=[config.value_col]).copy()

        if len(values):
            biggest_increase = values.loc[values[config.value_col].idxmax()]
            biggest_decrease = values.loc[values[config.value_col].idxmin()]

            st.caption(
                "PB comparison uses the same Scenario + Model + Task as baseline. "
                "Positive values mean the selected language had a larger metric than PB NATIVO; "
                "negative values mean it had a smaller metric."
            )

            ci1, ci2 = st.columns(2)
            ci1.info(
                "Largest increase: "
                f"{biggest_increase[config.value_col]:.2f}% — "
                f"{biggest_increase['Scenario']} | {biggest_increase['Model']} | "
                f"{biggest_increase['Task']} | {biggest_increase['Language']}"
            )
            ci2.info(
                "Largest decrease: "
                f"{biggest_decrease[config.value_col]:.2f}% — "
                f"{biggest_decrease['Scenario']} | {biggest_decrease['Model']} | "
                f"{biggest_decrease['Task']} | {biggest_decrease['Language']}"
            )


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
    )

    st.title(APP_TITLE)
    st.caption(
        "Explore raw metrics or percentage change against PB NATIVO across scenarios, models, languages, and tasks."
    )

    base_df = load_data_ui()
    metrics = available_metrics(base_df)

    if not metrics:
        st.error("No numeric metric columns were found.")
        st.stop()

    enriched = add_pb_comparisons(base_df, metrics)
    filtered, config, order_maps = build_sidebar_filters(enriched, metrics)

    curiosity_panel(filtered, config)

    chart_type = st.selectbox(
        "Chart type",
        [
            "Bar chart",
            "Grouped bar chart",
            "Heatmap",
            "Distribution / box plot",
            "Profile line chart",
            "Metric scatter plot",
            "Ranking table",
        ],
        index=0,
    )

    fig: go.Figure
    aggregate_df: pd.DataFrame | None

    if chart_type == "Bar chart":
        fig, aggregate_df = chart_bar(filtered, config, order_maps)
    elif chart_type == "Grouped bar chart":
        fig, aggregate_df = chart_grouped_bar(filtered, config, order_maps)
    elif chart_type == "Heatmap":
        fig, aggregate_df = chart_heatmap(filtered, config, order_maps)
    elif chart_type == "Distribution / box plot":
        fig, aggregate_df = chart_box(filtered, config, order_maps)
    elif chart_type == "Profile line chart":
        fig, aggregate_df = chart_line_profile(filtered, config, order_maps)
    elif chart_type == "Metric scatter plot":
        fig, aggregate_df = chart_scatter(filtered, metrics, config, order_maps)
    elif chart_type == "Ranking table":
        fig, aggregate_df = chart_rank_table(filtered, config)
    else:
        st.stop()

    st.plotly_chart(fig, use_container_width=True)

    prefix = clean_name(f"{chart_type}_{config.data_mode}_{config.metric}")
    display_downloads(fig, filtered, aggregate_df, prefix)

    with st.expander("How the PB NATIVO comparison is calculated"):
        st.write(
            """
            For each selected metric, the app finds the PB NATIVO value with the same
            `Scenario`, `Model`, and `Task`, then computes:

            `Δ% vs PB NATIVO = (row_value / pb_nativo_value - 1) × 100`

            This means comparisons are local to each model and task instead of using a
            single global PB baseline.
            """
        )

    with st.expander("Notes about standard deviation"):
        st.write(
            """
            Bar charts and line charts display standard deviation as error bars when
            aggregation is enabled. Heatmaps display standard deviation and sample size
            in the hover tooltip, and can also be switched to a standard-deviation heatmap.
            Box plots show the full distribution of the selected rows.
            """
        )


if __name__ == "__main__":
    main()
