"""
Interactive web viewer for reduced embeddings.

Launches a Plotly Dash app to visualize 2D/3D projections of embeddings
with color-coding by dataset and interactive hover text.
"""

import argparse
import json
import logging
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

logger = logging.getLogger(__name__)


def load_reduced_embeddings(parquet_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load reduced embeddings and metadata."""
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    metadata_path = parquet_path.with_suffix(".json")
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    logger.info(f"Loaded {len(df)} points from {parquet_path}")
    logger.info(f"Datasets: {df['dataset'].unique()}")

    return df, metadata


def create_app(df: pd.DataFrame, metadata: dict) -> Dash:
    """Create Dash app for interactive visualization."""

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)

    preview_chars = metadata.get("hover_preview_chars", 160)

    def make_preview(text: str) -> str:
        text = text.strip()
        if len(text) <= preview_chars:
            return text
        return text[:preview_chars].rstrip() + "…"

    df["preview"] = df["text"].apply(make_preview)

    def make_hover_label(text: str) -> str:
        if not text:
            return "No text available"
        wrapped = textwrap.wrap(text, width=80)
        if not wrapped:
            return text
        return "<br>".join(wrapped)

    df["hover_label"] = df["preview"].apply(make_hover_label)

    # Get unique datasets
    datasets = sorted(df["dataset"].unique())

    # Create color mapping (use distinct colors)
    color_map = {
        "allenai__WildChat-4.8M": "#FF6B6B",  # Red
        "HuggingFaceFW__fineweb__sample-10BT": "#4ECDC4",  # Teal
    }
    # Fallback colors if we have other datasets
    fallback_colors = ["#95E1D3", "#F38181", "#AA96DA", "#FCBAD3", "#FFFFD2"]
    for i, ds in enumerate(datasets):
        if ds not in color_map:
            color_map[ds] = fallback_colors[i % len(fallback_colors)]

    # Create app with Bootstrap theme
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Create initial figure
    def create_figure(visible_datasets):
        """Create plotly figure with specified datasets visible."""
        filtered_df = df[df["dataset"].isin(visible_datasets)]

        if len(filtered_df) == 0:
            # Empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No data selected",
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2"
            )
            return fig

        # Determine if 2D or 3D
        is_3d = metadata.get("n_components", 2) == 3 and (df["z"] != 0).any()

        if is_3d:
            fig = px.scatter_3d(
                filtered_df,
                x="x",
                y="y",
                z="z",
                color="dataset",
                custom_data=["text", "dataset", "hover_label"],
                color_discrete_map=color_map,
                opacity=0.7
            )
        else:
            fig = px.scatter(
                filtered_df,
                x="x",
                y="y",
                color="dataset",
                custom_data=["text", "dataset", "hover_label"],
                color_discrete_map=color_map,
                opacity=0.7
            )

        # Update layout
        fig.update_traces(
            marker=dict(size=3),
            hovertemplate="<b>%{customdata[1]}</b><br>%{customdata[2]}<extra></extra>"
        )
        fig.update_layout(
            title=f"{metadata.get('method', 'Dimensionality Reduction').upper()} Projection "
                  f"({len(filtered_df):,} points)",
            hovermode="closest",
            height=800,
            template="plotly_white",
            clickmode="event+select",
            hoverlabel=dict(bgcolor="white", font_family="monospace", font_size=12)
        )

        return fig

    # Layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Embedding Space Visualization"),
                html.P([
                    f"Method: {metadata.get('method', 'unknown').upper()}, ",
                    f"Total points: {len(df):,}, ",
                    f"Dimensions: {metadata.get('n_components', 2)}D"
                ])
            ])
        ], className="mb-3 mt-3"),

        dbc.Row([
            dbc.Col([
                html.H5("Dataset Visibility"),
                dbc.Checklist(
                    id="dataset-checklist",
                    options=[{"label": ds, "value": ds} for ds in datasets],
                    value=datasets,  # All visible by default
                    inline=False,
                    switch=True
                )
            ], width=3),

            dbc.Col([
                dcc.Graph(id="embedding-plot", style={"height": "800px"})
            ], width=9)
        ]),

        dbc.Row([
            dbc.Col(width=3),  # Empty spacer to match sidebar
            dbc.Col([
                html.Div(id="selection-info", className="mt-3")
            ], width=9)
        ])
    ], fluid=False, style={"max-width": "1400px"})

    # Callback to update figure
    @callback(
        Output("embedding-plot", "figure"),
        Input("dataset-checklist", "value")
    )
    def update_figure(visible_datasets):
        return create_figure(visible_datasets)

    # Callback to show selection info
    @callback(
        Output("selection-info", "children"),
        Input("embedding-plot", "clickData")
    )
    def display_click_data(clickData):
        if clickData is None:
            return html.P("Click on a point to see its text", className="text-muted")

        point = clickData["points"][0]

        # Extract data from the point
        customdata = point.get("customdata") or []
        text = customdata[0] if len(customdata) > 0 else "No text available"
        dataset_name = customdata[1] if len(customdata) > 1 else point.get("fullData", {}).get("name", "unknown")

        x = point.get("x", 0)
        y = point.get("y", 0)

        return dbc.Card([
            dbc.CardHeader("Selected Point"),
            dbc.CardBody([
                html.P([html.Strong("Dataset: "), dataset_name]),
                html.P([html.Strong("Coordinates: "), f"({x:.3f}, {y:.3f})"]),
                html.Hr(),
                html.P(html.Strong("Text:"), className="mb-2"),
                html.Div(
                    text,
                    style={
                        "font-family": "monospace",
                        "font-size": "14px",
                        "white-space": "pre-wrap",
                        "word-break": "break-word",
                        "overflow-wrap": "anywhere",
                        "background-color": "#f8f9fa",
                        "padding": "10px",
                        "border-radius": "4px"
                    }
                )
            ])
        ], className="mt-3")

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Interactive web viewer for reduced embeddings"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to reduced embeddings parquet file"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to bind to"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    # Load data
    df, metadata = load_reduced_embeddings(Path(args.input))

    # Create and run app
    app = create_app(df, metadata)

    logger.info(f"Starting server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
