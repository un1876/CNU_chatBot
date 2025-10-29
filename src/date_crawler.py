import re
from datetime import datetime, timedelta, date
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

def _today(tz: str | None = "Asia/Seoul") -> date:
    if tz and ZoneInfo is not None:
        return datetime.now(ZoneInfo(tz)).date()
    return datetime.now().date()

def _fmt_dot(d: date) -> str:
    return f"{d.year:04d}.{d.month:02d}.{d.day:02d}"

def _parse_relative(msg: str, base: date) -> date | None:
    # 긴 표현을 먼저 매칭 (겹침 방지)
    rules: list[tuple[str, int]] = [
        (r"(내일\s*모레|내일모레)", 2),
        (r"(그그저께)", -3),
        (r"(그저께|전전일)", -2),
        (r"(어제|전일)", -1),
        (r"(오늘|금일|당일)", 0),
        (r"(내일|명일|익일)", 1),
        (r"(모레)", 2),
        (r"(글피)", 3),
        (r"(그글피)", 4),
    ]
    for pat, offset in rules:
        if re.search(pat, msg):
            return base + timedelta(days=offset)
    return None

def _parse_explicit(msg: str, base_year: int) -> date | None:
    # 1) 연도 포함 (예: 2025.10.21 / 2025-10-21 / 2025년 10월 21일)
    m = re.search(
        r"(?P<y>\d{4})\s*(?:년|\.|\/|-|\s)\s*"
        r"(?P<m>\d{1,2})\s*(?:월|\.|\/|-|\s)\s*"
        r"(?P<d>\d{1,2})\s*일?",
        msg
    )
    if m:
        return date(int(m["y"]), int(m["m"]), int(m["d"]))

    # 2) 연도 없이 월/일 (예: 10월 21일 / 10.21 / 10/21 / 10-21)
    m = re.search(
        r"(?P<m>\d{1,2})\s*(?:월|\.|\/|-|\s)\s*(?P<d>\d{1,2})\s*일?",
        msg
    )
    if m:
        return date(base_year, int(m["m"]), int(m["d"]))

    return None

def get_date(message: str, tz: str | None = "Asia/Seoul") -> str | None:
    """
    message에서 날짜를 해석해 'YYYY.MM.DD' 한 줄 문자열로 반환.
    - 숫자 날짜(예: 2025.10.21, 10월 21일 등)가 있으면 우선 사용
    - 없으면 오늘/어제/내일/모레/글피/그글피/내일모레/금일/전일/익일 등 상대 날짜 사용
    - 둘 다 없으면 None 반환
    """
    base = _today(tz)

    # 숫자 날짜 우선
    d = _parse_explicit(message, base_year=base.year)
    if d is None:
        d = _parse_relative(message, base)

    return _fmt_dot(d) if d else None
