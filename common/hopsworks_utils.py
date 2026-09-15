import hopsworks

from . import config


def get_project():
    return hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT,
    )


def get_feature_store():
    project = get_project()
    return project, project.get_feature_store()


def get_or_create_price_fg(fs):
    return fs.get_or_create_feature_group(
        name=config.PRICE_FG_NAME,
        version=config.FG_VERSION,
        description="Hourly day-ahead electricity price with lag/rolling features",
        primary_key=["area", "datetime"],
        event_time="datetime",
        online_enabled=False,
        time_travel_format="HUDI",  # DELTA needs deltalake, not available on Windows
    )


def get_or_create_weather_fg(fs):
    return fs.get_or_create_feature_group(
        name=config.WEATHER_FG_NAME,
        version=config.FG_VERSION,
        description="Hourly weather (actual/forecast) and calendar features for Stockholm",
        primary_key=["datetime"],
        event_time="datetime",
        online_enabled=False,
        time_travel_format="HUDI",
    )


def get_or_create_predictions_fg(fs):
    return fs.get_or_create_feature_group(
        name=config.PREDICTIONS_FG_NAME,
        version=config.PREDICTIONS_FG_VERSION,
        description="Model predictions for future hourly electricity prices, one row per "
        "(target hour, time the forecast was made) so every forecast horizon is retained",
        primary_key=["datetime", "prediction_made_at"],
        event_time="datetime",
        online_enabled=True,
        time_travel_format="HUDI",
    )


def get_champion_model(project):
    """The current champion is simply the highest registered version.

    training_pipeline/train_model.py only ever registers a new version when it
    beats the reigning champion on a freshly re-run, identical backtest, so
    version number alone is a safe way to pick "current best" - unlike
    comparing raw stored 'mae' values across versions (mr.get_best_model),
    which silently breaks the moment the evaluation methodology changes (as it
    did going from a single-shot oracle eval to a recursive backtest: older
    versions' mae is on a different, incomparable scale).
    """
    mr = project.get_model_registry()
    models = mr.get_models(config.MODEL_NAME)
    if not models:
        return None
    return max(models, key=lambda m: m.version)


def get_or_create_feature_view(fs):
    price_fg = get_or_create_price_fg(fs)
    weather_fg = get_or_create_weather_fg(fs)

    # exclude weather's datetime to avoid the "weather_"-prefixed column names
    # Hopsworks generates when both sides of a join share a column name
    query = price_fg.select_all().join(weather_fg.select_except(["area", "datetime"]), on=["datetime"])

    return fs.get_or_create_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION,
        description="Joined price + weather + calendar features for electricity price forecasting",
        query=query,
    )
