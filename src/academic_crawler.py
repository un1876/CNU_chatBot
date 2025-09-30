import json
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
from collections import defaultdict


def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def extract_month_day(text):
    # MM.DD 형식에서 월과 일을 추출
    match = re.search(r'(\d{1,2})\.(\d{1,2})', text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def crawl_academic_calendar():
    url = "https://plus.cnu.ac.kr/_prog/academic_calendar/?site_dvs_cd=kr&menu_dvs_cd=05020101&year=2025"
    driver = get_driver()
    try:
        driver.get(url)
    except Exception as e:
        print("❗ WebDriver 오류 발생:", e)
        driver.quit()
        return
    time.sleep(3)

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # 기준 연도 추출
    year_text = soup.select_one('strong.year')
    if not year_text:
        print("기준 연도를 찾을 수 없습니다.")
        driver.quit()
        return

    current_year = int(year_text.get_text(strip=True))
    driver.quit()

    data_by_month = defaultdict(list)
    is_first = True

    calen_boxes = soup.select('div.calen_box')
    for calen in calen_boxes:
        li_tags = calen.select('div.fr_list ul li')
        for li in li_tags:
            text = li.get_text(strip=True)
            month, _ = extract_month_day(text)
            if month is None:
                continue

            # 첫 번째 12월은 이전 해
            if is_first and month == 12:
                event_year = current_year - 1
            else:
                event_year = current_year

            is_first = False

            key = f"{event_year}년 {month}월"
            data_by_month[key].append({
                "내용": text,
                "분류": "학사일정"
            })

    # 리스트로 변환
    result = []
    for month_year, schedules in sorted(data_by_month.items()):
        result.append({
            "month": month_year,
            "schedules": schedules
        })

    # JSON 저장
    with open("academic_calendar.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


