import re, json, time
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0"}

def _expand_table_to_grid(table):
    """HTML <table>을 rowspan/colspan까지 펼친 2D 리스트로 변환"""
    grid = []
    spans = {}  # (r,c)->남은 row span
    rows = table.find_all("tr")
    for r_idx, tr in enumerate(rows):
        grid.append([])
        c_idx = 0
        # 이미 위에서 내려온 rowspan 차지 칸들 메우기
        while (r_idx, c_idx) in spans and spans[(r_idx, c_idx)] > 0:
            grid[r_idx].append("")  # 자리만 차지
            spans[(r_idx, c_idx)] -= 1
            c_idx += 1

        for cell in tr.find_all(["th", "td"]):
            # 다음 빈 칸 인덱스 찾기(위에서 내려온 span 자리 건너뛰기)
            while (r_idx, c_idx) in spans and spans[(r_idx, c_idx)] > 0:
                grid[r_idx].append("")
                spans[(r_idx, c_idx)] -= 1
                c_idx += 1

            txt = " ".join(cell.stripped_strings)
            rs = int(cell.get("rowspan", 1))
            cs = int(cell.get("colspan", 1))

            # 현재 행에 colspan 만큼 채우기
            for _ in range(cs):
                grid[r_idx].append(txt)
                c_idx += 1

            # 아래 행들에 rowspan-1 표시
            if rs > 1:
                for rr in range(1, rs):
                    for cc in range(cs):
                        spans[(r_idx + rr, c_idx - cs + cc)] = spans.get((r_idx + rr, c_idx - cs + cc), 0) + 1
    return grid

def _clean_lines(text):
    lines = re.split(r"[\n\r]+|(?<!\S)[•·\-–]\s*", text)
    out = []
    for ln in lines:
        ln = re.sub(r"\s+", " ", ln).strip()

        if ln and ln != "메뉴운영내역":
            out.append(ln)
    seen, uniq = set(), []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq

def _normalize_url(url, date=None, cafeteria=None, lang=None):
    """질문에서 준 URL 그대로 써도 되고, 파라미터만 바꿔서 재구성도 가능"""
    u = urlparse(url)
    q = parse_qs(u.query)
    if date is not None:       q["searchYmd"] = [date]
    if cafeteria is not None:  q["searchCafeteria"] = [cafeteria]
    if lang is not None:
        q["searchLang"] = [lang]
        q["Language_gb"] = [lang]
    new_q = urlencode({k: v[0] for k, v in q.items()})
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_q, u.fragment))

def get_menu(url, date=None, cafeteria=None, lang=None):
    """
    url: 예) https://mobileadmin.cnu.ac.kr/food/index.jsp?... (질문에서 준 URL 그대로 사용 가능)
    date/cafeteria/lang을 넘기면 해당 쿼리로 덮어써서 호출
    반환: { date, columns:[제1학생회관...], data[컬럼][meal][audience]: [메뉴...] }
    """
    url = _normalize_url(url, date=date, cafeteria=cafeteria, lang=lang)
    r = requests.get(url, headers=HEADERS, timeout=15)
    if not r.encoding or r.encoding.lower() == "iso-8859-1":
        r.encoding = r.apparent_encoding or "utf-8"

    soup = BeautifulSoup(r.text, "lxml")

    # 식단 테이블 찾기: '구분' 헤더가 있는 표를 선택
    table = None
    for tb in soup.find_all("table"):
        if tb.find(string=re.compile(r"구분")):
            table = tb
            break
    if table is None:
        raise RuntimeError("식단 테이블을 찾지 못했습니다. 페이지 구조가 변경되었을 수 있어요.")

    grid = _expand_table_to_grid(table)
    # 헤더 행(열 머리글) 찾기: '구분' 포함된 행
    header_row = None
    for row in grid:
        if any("구분" in (cell or "") for cell in row):
            header_row = row
            break
    if header_row is None:
        raise RuntimeError("헤더 행을 찾지 못했습니다.")

    # 컬럼 이름: 0열=구분, 1열=직원/학생, 2열~ = 각 식당(제1학생회관 등)
    columns = [c.strip() for c in header_row]
    # 뒤쪽 빈칸 제거
    while columns and not columns[-1]:
        columns.pop()
    cafeteria_cols = columns[2:]

    # 본문 시작 위치(헤더 다음 행부터)
    start_idx = grid.index(header_row) + 1

    result = {
        "date": parse_qs(urlparse(url).query).get("searchYmd", [""])[0],
        "columns": cafeteria_cols,
        "data": {col: {"조식": {"직원": [], "학생": []},
                       "중식": {"직원": [], "학생": []},
                       "석식": {"직원": [], "학생": []}} for col in cafeteria_cols},
        "source_url": url,
        "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    current_meal = None  # 조식/중식/석식 forward-fill
    for row in grid[start_idx:]:
        # 길이가 짧은 행 보정
        if len(row) < 2:
            continue
        if row[0]:
            current_meal = row[0].strip()  # 조식/중식/석식
        audience = (row[1] or "").strip()  # 직원/학생
        if current_meal not in ("조식", "중식", "석식"):
            continue
        if audience not in ("직원", "학생"):
            continue

        # 각 식당 칸을 분리
        for idx, col_name in enumerate(cafeteria_cols, start=2):
            if idx >= len(row):
                continue
            cell = (row[idx] or "").strip()
            if not cell:
                continue
            lines = _clean_lines(cell)
            if lines:
                result["data"][col_name][current_meal][audience].extend(lines)

    return result