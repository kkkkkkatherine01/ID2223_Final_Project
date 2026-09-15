from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from common import config

_STOCKHOLM_TZ = ZoneInfo(config.STOCKHOLM_TZ)


def next_delivery_day_window(reference_time: datetime) -> tuple[datetime, datetime]:
    """Tomorrow's full Stockholm calendar day, as (start, end) in UTC.
    Not always 24h - 23h/25h on DST transition days.
    """
    local_now = reference_time.astimezone(_STOCKHOLM_TZ)
    tomorrow = (local_now + timedelta(days=1)).date()
    start = datetime.combine(tomorrow, time(0, 0), tzinfo=_STOCKHOLM_TZ)
    end = datetime.combine(tomorrow, time(23, 0), tzinfo=_STOCKHOLM_TZ)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)
