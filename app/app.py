import os
import sys
import time
from datetime import timedelta

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.hopsworks_utils import get_feature_store, get_or_create_price_fg  # noqa: E402
from inference_pipeline.daily_inference import get_or_create_predictions_fg  # noqa: E402

CACHE_TTL_SECONDS = 900
_cache = {"data": None, "ts": 0}


def load_data():
    if _cache["data"] is not None and time.time() - _cache["ts"] < CACHE_TTL_SECONDS:
        return _cache["data"]
    _, fs = get_feature_store()
    price_df = get_or_create_price_fg(fs).read().sort_values("datetime")
    pred_df = get_or_create_predictions_fg(fs).read().sort_values("datetime")
    _cache["data"] = (price_df, pred_df)
    _cache["ts"] = time.time()
    return price_df, pred_df


def build_dashboard():
    price_df, pred_df = load_data()
    now = pd.Timestamp.utcnow()

    history_window = price_df[price_df["datetime"] >= now - timedelta(days=7)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_window["datetime"], y=history_window["price_eur_mwh"],
        name="Actual price", line=dict(color="#1f77b4"),
    ))
    fig.add_trace(go.Scatter(
        x=pred_df["datetime"], y=pred_df["predicted_price_eur_mwh"],
        name="Forecast", line=dict(color="#ff7f0e", dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="Time (UTC)", yaxis_title="EUR / MWh",
        legend=dict(orientation="h", yanchor="bottom", y=1.02), height=450,
    )

    upcoming = pred_df[pred_df["datetime"] >= now].head(24)
    if upcoming.empty:
        best_times_md = "No forecast available yet - check back after the next inference run."
    else:
        cheapest = upcoming.nsmallest(3, "predicted_price_eur_mwh")
        best_times_md = "\n\n".join(
            f"**{row['datetime'].strftime('%a %H:%M')}** — {row['predicted_price_eur_mwh']:.1f} EUR/MWh"
            for _, row in cheapest.iterrows()
        )

    merged = pd.merge(
        pred_df[pred_df["datetime"] < now],
        price_df[["datetime", "price_eur_mwh"]],
        on="datetime", how="inner",
    ).sort_values("datetime")
    merged = merged[merged["datetime"] >= now - timedelta(days=7)]

    fig2 = go.Figure()
    if merged.empty:
        mae_md = "Not enough overlapping history yet to compare predictions vs actuals."
    else:
        mae = (merged["predicted_price_eur_mwh"] - merged["price_eur_mwh"]).abs().mean()
        mae_md = f"MAE over shown window: {mae:.2f} EUR/MWh"
        fig2.add_trace(go.Scatter(x=merged["datetime"], y=merged["price_eur_mwh"], name="Actual"))
        fig2.add_trace(go.Scatter(x=merged["datetime"], y=merged["predicted_price_eur_mwh"], name="Predicted"))
    fig2.update_layout(xaxis_title="Time (UTC)", yaxis_title="EUR / MWh", height=350)

    return fig, best_times_md, fig2, mae_md


with gr.Blocks(title="SE3 Electricity Price Forecast") as demo:
    gr.Markdown("# Swedish Electricity Price Forecast (SE3 - Stockholm)")
    refresh_btn = gr.Button("Refresh")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Price trend & 24h forecast")
            price_plot = gr.Plot()
        with gr.Column(scale=1):
            gr.Markdown("### Best times to use electricity")
            best_times = gr.Markdown()

    gr.Markdown("### Predicted vs actual (last 7 days)")
    mae_text = gr.Markdown()
    compare_plot = gr.Plot()

    outputs = [price_plot, best_times, compare_plot, mae_text]
    demo.load(build_dashboard, outputs=outputs)
    refresh_btn.click(build_dashboard, outputs=outputs)

if __name__ == "__main__":
    demo.launch()
