# SE3 Electricity Price Forecast

ID2223 project (group_888): predicts next-24h hourly day-ahead electricity
prices for the Stockholm region (SE3) using weather + calendar features,
served through a Streamlit dashboard.

## Live dashboard

- **Primary**: https://id2223finalproject-tdkxcqvvre8kraotgy3ffg.streamlit.app (Streamlit Community Cloud)
- **Backup**: https://huggingface.co/spaces/kkkkkkatherine/se3-electricity-forecast
  (Gradio, on Hugging Face Spaces - kept running as a fallback; see "Deploying
  the dashboard" below for why it's not the primary)

## Architecture

Feature-Training-Inference pipeline on Hopsworks:

- **Feature pipeline** (`feature_pipeline/`, daily via GitHub Actions):
  pulls SE3 day-ahead prices (ENTSO-E Transparency Platform) and
  OpenWeatherMap forecasts, engineers lag/rolling price features and
  calendar features, upserts into two Hopsworks Feature Groups
  (`electricity_prices`, `weather`).
- **Training pipeline** (`training_pipeline/`, weekly): reads the joined
  Feature View, trains an XGBoost regressor with a time-based
  train/test split, registers the model in the Hopsworks Model Registry.
- **Inference pipeline** (`inference_pipeline/`, daily): builds feature
  rows for the next 24 hours from the latest forecast weather + most recent
  known price lags, predicts, writes to the `price_predictions` Feature
  Group.
- **Dashboard** (`app/app.py`): Streamlit app, deployed on Streamlit
  Community Cloud, reads directly from the Feature Store.

## One-time setup

1. `pip install -r requirements.txt` (Linux/Mac). **On Windows**, this
   fails building `twofish` - run `powershell -File
   scripts\install_windows.ps1` instead (see "Known rough edges" below for
   why).
2. Copy `.env.example` to `.env` and fill in:
   - `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT` (create a project at
     app.hopsworks.ai first)
   - `OPENWEATHER_API_KEY` (free tier at openweathermap.org; new keys take
     up to ~2h to activate)
   - `ENTSOE_API_KEY` (register at transparency.entsoe.eu, then generate a
     "Web Api Security Token" from account settings, or email
     transparency@entsoe.eu with subject "Restful API access" if that
     option isn't visible on your account)
3. Backfill history (only needs to run once):
   ```bash
   python -m feature_pipeline.backfill_prices
   python -m feature_pipeline.backfill_weather
   ```
4. Train the first model:
   ```bash
   python -m training_pipeline.train_model
   ```
5. Run inference once to populate predictions:
   ```bash
   python -m inference_pipeline.daily_inference
   ```

## Automation (GitHub Actions)

Add these repo secrets: `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`,
`OPENWEATHER_API_KEY`, `ENTSOE_API_KEY`. The three workflows in
`.github/workflows/` then run
on their own schedule (daily feature+inference, weekly training) - see each
file for the cron expression, and use "Run workflow" to trigger manually.

## Deploying the dashboard

**Primary - Streamlit Community Cloud** (share.streamlit.io): sign in with
GitHub, connect this repo, set the main file path to `app/app.py`, and add
`HOPSWORKS_API_KEY` / `HOPSWORKS_PROJECT` under the app's Secrets (TOML
format). Every push to `main` redeploys automatically. Uses
`app/requirements.txt` (not the root one) - deliberately excludes
`hopsworks[python]`'s `confluent-kafka` dependency, since Streamlit Cloud's
Debian image ships a `librdkafka` too old to build it against; the
dashboard only ever reads from the Feature Store, so plain `hopsworks` +
an explicit `pyarrow` is enough.
