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
