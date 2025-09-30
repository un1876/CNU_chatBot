# chatbot_core.py
import json, re, urllib.parse, requests
from bs4 import BeautifulSoup
from datetime import datetime
import torch
from transformers import AutoTokenizer, ElectraForSequenceClassification, PreTrainedTokenizerFast, GPT2LMHeadModel
from notice_crawler import CNUNoticeCrawler

# 전역 모델 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained("../kogpt2-finetuned")
model = GPT2LMHeadModel.from_pretrained("../kogpt2-finetuned").eval()

# 데이터 로딩
with open("menu_1.json", "r", encoding="utf-8") as f:
    fixed_menu_1 = json.load(f)
with open("menu_other.json", "r", encoding="utf-8") as f:
    daily_menu = json.load(f)
with open('bus.json', 'r', encoding='utf-8') as f:
    bus_stops = json.load(f)

from academic_crawler import crawl_academic_calendar




# --- 필요한 함수들 복사 ---
# 2) get_from_graduate
# 3) rag_answer_for_notices

# 5) answer_from_daily_menu
# 6) get_bus_arrival_by_number
# 7) chatbot_pipeline
# ...
#---------------------------식단--------------------------------------------------------------------------------------
def extract_cafeteria_from_message(message):
    """
    메시지에서 식당명 추출: '2학', '3학', '4학', '생과대' 등의 표현도 처리
    """
    if re.search(r"(2학|2학생회관)", message):
        return "제2학생회관"
    elif re.search(r"(3학|3학생회관)", message):
        return "제3학생회관"
    elif re.search(r"(4학|4학생회관)", message):
        return "제4학생회관"
    elif re.search(r"(생과대|생활과학대학)", message):
        return "생활과학대학"
    return None
def get_meal_types_from_message_or_time(message):
    meal_keywords = {
        "조식": ["조식", "아침"],
        "중식": ["중식", "점심"],
        "석식": ["석식", "저녁"]
    }
    detected_meals = []

    # 메시지에 식사명 키워드가 포함되어 있는지 확인
    for meal, keywords in meal_keywords.items():
        if any(word in message for word in keywords):
            detected_meals.append(meal)

    # 아무것도 없다면 현재 시간 기준 1개만 반환
    if not detected_meals:
        now = datetime.now().time()
        if now < datetime.strptime("10:00", "%H:%M").time():
            return ["조식"]
        elif now < datetime.strptime("14:00", "%H:%M").time():
            return ["중식"]
        elif now < datetime.strptime("20:00", "%H:%M").time():
            return ["석식"]
        else:
            return []

    return detected_meals

def make_rag_context_from_fixed_menu(menu_dict):
    context_lines = []
    for category, items in menu_dict.items():
        context_lines.append(f"📂 {category}")
        for item, price in items.items():
            context_lines.append(f"{item}: {price}원")
        context_lines.append("---")
    return "\n".join(context_lines)

def rag_answer_from_fixed_menu(message):
    # (1) 메뉴 로드
    menu_dict = fixed_menu_1

    # (2) Retrieval context 생성
    retrieval_context = make_rag_context_from_fixed_menu(menu_dict)

    # (3) 프롬프트 구성
    prompt = (
        "[SYSTEM] 다음 1학생회관 상시메뉴 정보를 참고하여 자연스럽게 답변해 주세요.\n"
        "[TOPIC] 식단\n"
        "[RETRIEVAL]\n"
        f"{retrieval_context}\n"
        f"[USER] {message} [SEP]"
    )

    # (4) 생성
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids,
        max_length=input_ids.shape[-1] + 100,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        do_sample=False,
        temperature=0.5,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # (5) 디코딩
    raw = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return raw.strip()

def make_rag_context_from_menu(message):
    cafeteria = extract_cafeteria_from_message(message)
    if cafeteria is None:
        return "식당 위치를 알 수 없습니다."

    대상 = "학생"
    if ("직원" in message or "교수" in message):
        대상 = "직원"

    식사명_리스트 = get_meal_types_from_message_or_time(message)

    context_lines = [f"질문: {message}", f"대상: {대상}", f"식당: {cafeteria}"]
    for meal_type in 식사명_리스트:
        try:
            meals = daily_menu[대상][meal_type]
            if meals:
                today_meal = meals[0]
                if cafeteria in today_meal:
                    menu_info = today_meal[cafeteria]
                    메뉴 = menu_info.get("메뉴", "").strip()
                    가격 = menu_info.get("가격", "").strip()
                    context_lines.append(f"{meal_type} - {cafeteria}: 메뉴: {메뉴} / 가격: {가격}")
                else:
                    context_lines.append(f"{meal_type} - {cafeteria}: 정보 없음")
            else:
                context_lines.append(f"{meal_type} - 정보 없음")
        except:
            context_lines.append(f"{meal_type} - 오류 발생")

    return "\n".join(context_lines)

def rag_answer_from_menu(message):
    retrieval_context = make_rag_context_from_menu(message)

    prompt = (
        "[SYSTEM] 아래 식단 정보를 참고해서 자연스럽고 정확하게 답변하세요.\n"
        "[TOPIC] 식단\n"
        f"[RETRIEVAL]\n{retrieval_context}\n"
        f"[USER] {message} [SEP] [BOT]"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids,
        max_length=input_ids.shape[-1] + 100,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.5
    )

    return tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
#--------------------------------------------식단 끝-------------------------------------------------------------



#--------------------------------------------통학/버스-------------------------------------------------------------
def extract_bus_info_for_rag(message):
    """
    메시지에서 버스 번호와 정류장명 추출 후, 관련 도착 정보를 문자열로 반환
    """
    import re
    bus_match = re.search(r'(\d{2,4})\s*번?', message)
    target_bus = bus_match.group(1) if bus_match else None

    stop_name = None
    for stop in bus_stops:
        if stop["정류장명"] in message:
            stop_name = stop["정류장명"]
            stop_id = stop["정류장ID"]
            break

    if not target_bus or not stop_name:
        return None, None, None, "버스 번호 또는 정류장을 찾을 수 없습니다."

    # API 요청
    url = f"http://openapitraffic.daejeon.go.kr/api/rest/busposinfo/getBusPosByRtid"
    params = {
        "serviceKey": "B%2FCPbINKFaAiuYyxiX216Mwr%2F%2Ff4O%2FTySlCctcTrjW%2BsxNef73j3ahB8ZERTr6jSbj5tBF6a0S5EQ6%2F%2FmNfYOg%3D%3D",  # 이 부분 채워줘야 함
        "busRouteId": target_bus
    }

    parsed_url = url + "?" + urllib.parse.urlencode(params, encoding="utf-8")
    response = requests.get(parsed_url)
    soup = BeautifulSoup(response.text, "xml")
    items = soup.find_all("item")

    info_lines = []
    for item in items:
        if item.find("stopFlag").text == "0":
            continue
        car_number = item.find("carNo").text
        remain_stops = item.find("stopCount").text
        predict_time = item.find("predictTime").text
        info_lines.append(f"- [{predict_time}분 후] 차량번호 {car_number}, 남은 정류장 {remain_stops}개")

    if not info_lines:
        info_lines.append("버스 도착 정보가 없습니다.")

    context = "\n".join(info_lines)
    return target_bus, stop_name, context, None

def rag_answer_from_bus(message):
    bus_number, stop_name, arrival_info, error = extract_bus_info_for_rag(message)

    if error:
        return error

    prompt = (
        "[SYSTEM] 다음 버스 도착 정보를 참고하여 자연스럽게 대답해 주세요.\n"
        "[TOPIC] 버스/통학\n"
        "[RETRIEVAL]\n"
        f"버스번호: {bus_number}\n"
        f"정류장명: {stop_name}\n"
        f"도착 정보:\n{arrival_info}\n"
        f"[USER] {message} [SEP] [BOT]"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids,
        max_length=input_ids.shape[-1] + 100,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
#-------------------------------------------------버스 수정필요-------------------------------------------------



#-------------------------------------------------졸업 요건----------------------------------------------------
def get_from_graduate(message,topic):
    old_prompt=f"[TOPIC] {topic} [USER] {message} [SEP] [BOT]"
    with open("졸업요건_RAG.json", encoding="utf-8") as f:
        rag = json.load(f)

    # (2) dept_alias 자동 생성 (JSON key 기반)
    departments = list(rag.get(topic, {}).keys())
    suffixes = ["학과", "학부", "대학", "교육과"]

    dept_alias = {}
    for name in departments:
        aliases = {name}
        for suf in suffixes:
            if name.endswith(suf) and len(name) > len(suf):
                base = name[:-len(suf)]
                aliases.update({base, base + "학", base + "학과"})
        dept_alias[name] = list(aliases)

    # (3) old_prompt에서 dept_key 찾기
    dept_key = next(
        (k for k, als in dept_alias.items() if any(a in old_prompt for a in als)),
        departments[0] if departments else None
    )

    # (4) 해당 학과 JSON 추출
    topic_data = rag.get(topic, {})
    dept_data = topic_data.get(dept_key, {})

    # (5) flatten into (path, text)
    chunks = []

    def walk(obj, path=[]):
        if isinstance(obj, dict):
            for k, v in obj.items(): walk(v, path + [k])
        elif isinstance(obj, list):
            for i, item in enumerate(obj): walk(item, path + [str(i)])
        else:
            chunks.append((path, str(obj)))

    walk(dept_data)

    # (6) old_prompt에서 어절 키워드 추출
    keywords = re.findall(r"[가-힣0-9]+", old_prompt)

    # (7) 부분문자열 매칭으로 관련 청크만 필터
    selected = []
    for path, text in chunks:
        if any(kw in key for kw in keywords for key in path) or any(kw in text for kw in keywords):
            selected.append((path, text))

    # (8) Retrieval 컨텍스트 생성
    retrieval = "\n".join(f"{'/'.join(p)}: {t}" for p, t in selected)

    # (9) **SYSTEM** 지시문 + RAG 프롬프트
    final_prompt = (
        f"[SYSTEM] 다음 정보를 참고하여 정확하게 답변하세요.\n"
        f"[TOPIC] {topic}\n"
        f"[RETRIEVAL]\n{retrieval}\n"
        f"[USER] {old_prompt} [SEP]"
    )

    # (10) 토크나이저·모델 로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained("../kogpt2-finetuned")
    model = GPT2LMHeadModel.from_pretrained("../kogpt2-finetuned").eval()

    # (11) 생성: beam search + 반복 방지 + 낮은 temperature
    input_ids = tokenizer.encode(final_prompt, return_tensors="pt")
    outs = model.generate(
        input_ids,
        max_length=input_ids.shape[-1] + 100,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        do_sample=False,
        temperature=0.5,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # (12) 디코딩 & 메타토큰 제거
    raw = tokenizer.decode(outs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    answer = raw.strip()
    return answer
#---------------------------------------------졸업요건끝-------------------------------------------------



#---------------------------------------------학사일정---------------------------------------------------
# 일단 학사일정 갱신

def make_rag_context_from_academic_calendar(message):
    with open("academic_calendar.json", encoding="utf-8") as f:
        calendar_data = json.load(f)

    context_lines = []
    for month_info in calendar_data:
        month = month_info["month"]
        for event in month_info["schedules"]:
            text = event["내용"]
            if any(keyword in message for keyword in [month] + re.findall(r"\d{1,2}월", message)):
                context_lines.append(f"{month}: {text}")
            elif re.search(r"\d{1,2}\.\d{1,2}", message) and re.search(r"\d{1,2}\.\d{1,2}", text):
                context_lines.append(f"{month}: {text}")

    if not context_lines:
        context_lines.append("관련된 학사일정을 찾을 수 없습니다.")

    return "\n".join(context_lines)

def rag_answer_from_academic_calendar(message):
    context = make_rag_context_from_academic_calendar(message)

    prompt = (
        "[SYSTEM] 아래 학사일정 정보를 참고해서 자연스럽고 정확하게 답변하세요.\n"
        "[TOPIC] 학사일정\n"
        f"[RETRIEVAL]\n{context}\n"
        f"[USER] {message} [SEP] [BOT]"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids,
        max_length=input_ids.shape[-1] + 100,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.5
    )

    return tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
#----------------------------------------------학사일정끝--------------------------------------------------
#----------------------------------------------공지사항----------------------------------------------------

def update_notices():
    crawler = CNUNoticeCrawler()
    notices = crawler.crawl_notices(max_pages=10)
    crawler.save_to_json(notices, filename="notices.json")

import json
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

# 전역 모델 불러오기
tokenizer = PreTrainedTokenizerFast.from_pretrained("../kogpt2-finetuned")
model = GPT2LMHeadModel.from_pretrained("../kogpt2-finetuned").eval()

def make_rag_context_from_notices(user_message, top_k=3):
    with open("notices.json", encoding="utf-8") as f:
        notice_data = json.load(f)["data"]

    user_message_lower = user_message.lower()

    # 간단한 keyword matching 기반 검색 (추후 BM25나 FAISS로 교체 가능)
    matches = []
    for item in notice_data:
        score = 0
        title = item["title"].lower()
        content = item["content"].lower()

        if any(word in title for word in user_message_lower.split()):
            score += 1
        if any(word in content for word in user_message_lower.split()):
            score += 2

        if score > 0:
            matches.append((score, item))

    # 상위 K개 선택
    matches = sorted(matches, key=lambda x: x[0], reverse=True)[:top_k]

    context_lines = []
    for _, item in matches:
        context_lines.append(f"📌 {item['title']} ({item['date']})")
        context_lines.append(item['content'][:300].replace('\n', ' ') + "...")
        context_lines.append("")

    if not context_lines:
        context_lines.append("관련된 공지사항을 찾지 못했습니다.")

    return "\n".join(context_lines)

def rag_answer_for_notices(user_message):
    retrieval_context = make_rag_context_from_notices(user_message)

    prompt = (
        "[SYSTEM] 아래 공지사항 정보를 참고해서 사용자의 질문에 답변하세요.\n"
        "[TOPIC] 학교공지사항\n"
        "[RETRIEVAL]\n"
        f"{retrieval_context}\n"
        f"[USER] {user_message} [SEP] [BOT]"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=input_ids.shape[-1] + 100,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.8,
    )

    return tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
#---------------------------------------------공지사항 끝-------------------------------------------

#---------------------------------------------답변하기--------------------------------------------
def update_all_data_once():
    global data_updated
    if not data_updated:
        print("📌 데이터 최초 갱신 중...")
        crawl_academic_calendar()
        update_notices()
        data_updated = True

#분류기를 사용하여 topic을 우선 추출
def extract_topic_from_message(message):

    model_dir = "../roberta"

    # 자동으로 저장된 토크나이저 타입을 불러옴
    tokenizer_classification = AutoTokenizer.from_pretrained(model_dir)
    model_classification = ElectraForSequenceClassification.from_pretrained(model_dir)
    inputs = tokenizer_classification(message, return_tensors="pt", truncation=True, padding=True)

    # 모델 예측
    with torch.no_grad():
        outputs = model_classification(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    id2label = {0: '버스/통학', 1: '식단', 2: '졸업요건', 3: '학교공지사항', 4: '학사일정'}
    print(f"예측 라벨: {id2label[predicted_class]}")
    topic = id2label[predicted_class]

    #이후 예측한 토픽을 return
    if topic:
        return topic
    else:
        return None

# respond 함수만 이 파일에 최종적으로 노출
def respond(message, history=None):
    update_all_data_once()

    if history is None:
        history = []

    topic = extract_topic_from_message(message)

    if topic == "식단":
        if "1학" in message or "1학생회관" in message:
            response = rag_answer_from_fixed_menu(message)
        else:
            response = rag_answer_from_menu(message)            # ✅
    elif topic == "버스/통학":
        response = rag_answer_from_bus(message)                 # ✅
    elif topic == "졸업요건":
        response = get_from_graduate(message, topic)            # ✅
    elif topic == "학사일정":
        response = rag_answer_from_academic_calendar(message)   # ✅
    elif topic == "공지사항":
        response = rag_answer_for_notices(message)              # ✅

    else:
        response = "지원되지 않는 주제입니다."

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history

