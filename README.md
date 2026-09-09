# SE3 Electricity Price Forecast

ID2223 project (group_888): predicts next-24h hourly day-ahead electricity
prices for the Stockholm region (SE3) using weather + calendar features,
served through a Gradio dashboard.

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
- **Dashboard** (`app/app.py`): Gradio app, deployed as a Hugging Face
  Space, reads directly from the Feature Store.

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

Create a Hugging Face Space (Gradio SDK), add it as a second git remote,
and push this repo to it. Edit the Space's generated README.md frontmatter
so `app_file: app/app.py` (it defaults to `app.py` at the repo root).
Add `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT` as Space secrets.

## Known rough edges to verify before relying on this

- **Prices come from ENTSO-E, not Nordpool directly**: an earlier version
  of this pipeline used the `nordpool` PyPI package, which scrapes
  Nordpool's public data-portal API. That endpoint turned out to only
  serve the last ~60 days of history to unauthenticated requests (older
  dates return 401), which isn't enough to train a seasonal model on. The
  ENTSO-E Transparency Platform publishes the same day-ahead auction
  results officially and allows pulling years of history with a free
  registered token - see `feature_pipeline/entsoe_client.py`.
- **`pyjks`/`twofish` on Windows**: `hopsworks` depends on `pyjks` (for
  Kafka client-cert auth used when writing to a Feature Group), which in
  turn lists `twofish` as a dependency - a C extension with no prebuilt
  Windows wheel. In practice the standard JKS keystore path Hopsworks
  actually uses never imports `twofish` (it's only needed for the
  BKS/PKCS12 code paths), so `scripts/install_windows.ps1` installs
  `hopsworks` and `pyjks` with `--no-deps` and supplies their real
  dependencies by hand, skipping `twofish`.
- **Feature Group storage format**: the Hopsworks SDK defaults new Feature
  Groups to `time_travel_format="DELTA"`, which needs the `deltalake`
  extra - not published for Windows at all. All FG creation helpers in
  `common/hopsworks_utils.py` explicitly pass `time_travel_format="HUDI"`
  instead, which works cross-platform.
- **OpenWeatherMap free tier**: forecast is 3-hourly, interpolated to
  hourly here - fine for temperature/wind but coarser for cloud cover.
- **DST**: all timestamps are kept in UTC throughout to sidestep the
  23h/25h CET/CEST transition days; the dashboard should probably convert
  to Europe/Stockholm only at display time.
