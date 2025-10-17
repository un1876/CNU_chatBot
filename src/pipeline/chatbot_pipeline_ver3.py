import json, re, urllib.parse ,requests,os,torch
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from bs4 import BeautifulSoup
from textwrap import dedent
from huggingface_hub import InferenceClient

data_updated = False


load_dotenv()
model_dir = "spidyun/chatbot-roberta"
BUS_KEY = os.getenv("BUS_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # .env에 HF_TOKEN=hf_***** 로 저장
PROVIDER = os.getenv("HF_PROVIDER", None)  # 'fireworks-ai' | 'together' | 'hf-inference' | None
MODEL_ID = os.getenv("MODEL_ID", "openai/gpt-oss-120b")


#분류 모델 - RobertA
tokenizer_classification = AutoTokenizer.from_pretrained(model_dir, token=HF_TOKEN, use_fast=True)
model_classification = AutoModelForSequenceClassification.from_pretrained(model_dir, token=HF_TOKEN).eval()


# 데이터 로딩
with open("../../rag_data/restaurant/menu_1.json", "r", encoding="utf-8") as f:
    fixed_menu_1 = json.load(f)
with open("../../rag_data/restaurant/menu_other.json", "r", encoding="utf-8") as f:
    daily_menu = json.load(f)
with open('../../rag_data/bus/bus.json', 'r', encoding='utf-8') as f:
    bus_stops = json.load(f)



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
    if "1학" in message or "1학생회관" in message:
        # (1) 메뉴 로드
        menu_dict = fixed_menu_1

        # (2) Retrieval context 생성
        retrieval_context = make_rag_context_from_fixed_menu(menu_dict)

    else:
        retrieval_context = make_rag_context_from_menu(message)

    return retrieval_context
#--------------------------------------------식단 끝-------------------------------------------------------------



#--------------------------------------------통학/버스-------------------------------------------------------------
def extract_bus_info_for_rag(message, service_key):
    import re
    bus_match = re.search(r'(\d{2,4})\s*번?', message)
    target_bus = bus_match.group(1) if bus_match else None

    if not target_bus:
        return None, None, None, "❗ 버스 번호를 찾을 수 없습니다."

    # 1. 정류장 ID 검색
    find_stops_url = "http://apis.data.go.kr/1613000/BusSttnInfoInqireService/getSttnNoList"
    params = {
        "serviceKey": service_key,
        "cityCode": 25,
        "nodeNm": message,  # 사용자가 말한 문장 전체에서 정류장 이름을 찾을 수 있도록
        "numOfRows": 10,
        "pageNo": 1,
        "_type": "xml"
    }

    stop_url = find_stops_url + "?" + urllib.parse.urlencode(params, encoding="utf-8")
    stop_response = requests.get(stop_url)
    stop_soup = BeautifulSoup(stop_response.text, "xml")
    stop_items = stop_soup.find_all("item")

    if not stop_items:
        return None, None, None, "❗ 정류장 이름을 찾을 수 없습니다."

    arrival_info_list = []
    matched_stop = None

    # 2. 각 정류소 ID에 대해 도착 정보 요청
    for stop in stop_items:
        stop_name = stop.find("nodenm").text
        stop_id = stop.find("nodeid").text

        arrival_url = "http://apis.data.go.kr/1613000/ArvlInfoInqireService/getSttnAcctoArvlPrearngeInfoList"
        params = {
            "serviceKey": service_key,
            "cityCode": 25,
            "nodeId": stop_id,
            "numOfRows": 10,
            "pageNo": 1,
            "_type": "xml"
        }

        arrival_response = requests.get(arrival_url + "?" + urllib.parse.urlencode(params, encoding="utf-8"))
        arrival_soup = BeautifulSoup(arrival_response.text, "xml")
        items = arrival_soup.find_all("item")

        for item in items:
            routeno = item.find("routeno").text
            if routeno == target_bus:
                arrtime = int(item.find("arrtime").text)
                remain_stops = item.find("arrprevstationcnt").text
                arrival_info_list.append(
                    f"- ⏱ {arrtime // 60}분 후 도착 예정 ({remain_stops}개 전 정류장) [정류장: {stop_name}]"
                )
                matched_stop = stop_name

    if not arrival_info_list:
        return target_bus, None, "❗ 해당 버스는 현재 도착 예정이 없습니다.", None

    return target_bus, matched_stop, "\n".join(arrival_info_list), None

def rag_answer_from_bus(message, service_key):
    bus_number, stop_name, arrival_info, error = extract_bus_info_for_rag(message, service_key)

    if error:
        return error

    prompt = (
        "[RETRIEVAL]\n"
        f"버스번호: {bus_number}\n"
        f"정류장명: {stop_name}\n"
        f"도착 정보:\n{arrival_info}\n"
    )


    return prompt

#-------------------------------------------------버스 수정필요-------------------------------------------------



#-------------------------------------------------졸업 요건----------------------------------------------------
def get_from_graduate(message,topic):
    old_prompt=f"[TOPIC] {topic} [USER] {message} [SEP] [BOT]"
    with open("../../rag_data/graduation_requirements/graduation_RAG.json", encoding="utf-8") as f:
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


    return retrieval
#---------------------------------------------졸업요건끝-------------------------------------------------





#---------------------------------------------학사일정---------------------------------------------------
# 일단 학사일정 갱신

def make_rag_context_from_academic_calendar(message):
    with open("../../rag_data/canlendar/academic_calendar.json", encoding="utf-8") as f:
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
    retrieval = make_rag_context_from_academic_calendar(message)

    return retrieval
#----------------------------------------------학사일정끝--------------------------------------------------
#----------------------------------------------공지사항----------------------------------------------------


import json
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

# 전역 모델 불러오기

def make_rag_context_from_notices(user_message, top_k=3):
    with open("../../rag_data/notice/notices.json", encoding="utf-8") as f:
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
    retrieval = make_rag_context_from_notices(user_message)

    return retrieval
#---------------------------------------------공지사항 끝-------------------------------------------

#---------------------------------------------답변하기--------------------------------------------

#분류기를 사용하여 topic을 우선 추출
def extract_topic_from_message(message):

    # 자동으로 저장된 토크나이저 타입을 불러옴
    inputs = tokenizer_classification(message, return_tensors="pt", truncation=False, padding=True)

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
def Chatmodel(message,rag,topic):
    # 1) 토큰/프로바이더 설정

    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN, provider=PROVIDER)
    m=(message or "").strip()
    t=(topic or "").strip()
    r=(rag or "").strip()
    system_promt=dedent(f"""
        너는 학교 infromation 데스크에 있는 안내원처럼 친절하고 상세히 설명해주는 AI야
        첫 문장에는 ""{m}(이)라고 물으셨다면 👊{t}👊에 대해 궁금하시군요 😁!"를 꼭 넣어줘
        {m}에서 사용자가 표현을 어떻게 해서 답을 해다랄고 하면 그에 맞는 표현으로 최종 출력해주고 맨 마지막줄에 그에 맞는 표현도 언급해줘
        !중요 {r}의 내용을 최우선으로 참고하고 해당 내용이 완전하지 않으면 근본적인 내용은 바뀌지 않는 선에서 조금 더 유추해 보강해줘
        !중요 전달해야하는 부분이 끝나면 더이상 출력하지 말고 멈춰
        !문의 관련 문구를 할 때는 https://plus.cnu.ac.kr/html/kr/ 해당 사이트 또는 해당 학과에 문의해라고 말해줘
        """)

    messages = [
        {"role": "system", "content": system_promt},
        {"role": "user", "content": f"질문{m}+참고데이터:{r}"}
    ]

    # 5) API 호출(출력)
    resp = client.chat_completion(
        messages=messages,
        max_tokens=2048,
        temperature=0.2,
    )

    # 6) 응답 반환
    return resp.choices[0].message.content


# respond 함수만 이 파일에 최종적으로 노출
def respond(message, history=None):

    if history is None:
        history = []

    topic = extract_topic_from_message(message)

    if topic == "식단":
        rag = rag_answer_from_menu(message)

    elif topic == "버스/통학":
        rag = rag_answer_from_bus(message, BUS_KEY)

    elif topic == "졸업요건":
        rag = get_from_graduate(message, topic)

    elif topic == "학사일정":
        rag = rag_answer_from_academic_calendar(message)

    elif topic == "공지사항":
        rag = rag_answer_for_notices(message)              #

    print(rag)

    response=Chatmodel(message,rag,topic)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history