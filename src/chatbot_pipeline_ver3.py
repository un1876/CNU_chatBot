import json, re,requests,os,torch,xml.etree.ElementTree as ET
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from textwrap import dedent
from huggingface_hub import InferenceClient
from pathlib import Path
from date_crawler import get_date
from menu_crawler import get_menu
data_updated = False


load_dotenv()
model_dir = "spidyun/chatbot-roberta"
BUS_KEY = os.getenv("BUS_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
PROVIDER = os.getenv("HF_PROVIDER", None)
MODEL_ID = os.getenv("HF_MODEL", "openai/gpt-oss-120b")
ROUTER_URL=os.getenv("HF_ROUTER_URL")


#분류 모델 - RobertA
tokenizer_classification = AutoTokenizer.from_pretrained(model_dir, token=HF_TOKEN, use_fast=True)
model_classification = AutoModelForSequenceClassification.from_pretrained(model_dir, token=HF_TOKEN).eval()

BASE_DIR = Path(__file__).resolve().parents[1]   # 프로젝트 루트(/chatbot)
RAG_PATH = BASE_DIR / "rag_data"

# 데이터 로딩
with open(RAG_PATH/ "restaurant" / "menu_1.json", "r", encoding="utf-8") as f:
    fixed_menu_1 = json.load(f)
with open(RAG_PATH/  "bus" / "bus_route.json", 'r', encoding='utf-8') as f:
    bus_stops = json.load(f)
with open(RAG_PATH/  "bus" / "Daejon_BUS_all_stops.json", encoding="utf-8") as f:
    Daejon_BUS_all_stops = json.load(f)


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
            if item=="운영시간":
                context_lines.append(f"{category} {item}은 {price}입니다.")
            else:
                context_lines.append(f"{item}: {price}원")
        context_lines.append("---")
    return "\n".join(context_lines)



def make_rag_context_from_menu(message):
    cafeteria = extract_cafeteria_from_message(message) # 위치 추출
    if cafeteria is None:
        return "식당 위치를 알 수 없습니다."

    target = "전체" # default=학생
    if ("직원" in message or "교수" in message):
        target = "직원"
    elif ("학생"in message):
        target= "학생"

    식사명_리스트 = get_meal_types_from_message_or_time(message)# 중식, 조식 , 석식

    date=str(get_date(message))

    url = f"https://mobileadmin.cnu.ac.kr/food/index.jsp?searchYmd={date}&searchLang=OCL04.10&searchView=cafeteria&searchCafeteria=OCL03.02&Language_gb=OCL04.10"

    if date == "error":
        return "오류 발생, 정확한 날짜를 기입해주세요"  # 필요하면 date/cafeteria/lang 인자도 함께 넘겨서 수정 가능

    else:
        menu = get_menu(url)  # 필요하면 date/cafeteria/lang 인자도 함께 넘겨서 수정 가능

        context_lines = [f"질문: {message}", f"대상: {target}", f"식당: {cafeteria}"]

        for meal_type in 식사명_리스트:
            try:
                meals = menu["data"][cafeteria][meal_type]
                if meals:
                    if  target =="전체":
                        context_lines.append(f"{meal_type} - {cafeteria}-{target} - {meals}")

                    else:
                        menu_info = meals[target]
                        menu_string =menu_info[0]
                        menu_string = menu_string.strip()
                        price, string_menu = (menu_string.split(None, 1) + [""])[:2]  # 공백이 없을 때도 안전
                        context_lines.append(f"{meal_type} - {cafeteria}- {target}: 메뉴: {string_menu} / 가격: {price}")

                else:
                    context_lines.append(f"{meal_type} - 정보 없음")
            except:
                context_lines.append(f"{meal_type} - 오류 발생")
    return "\n".join(context_lines)

def rag_answer_from_menu(message):
    if "1학" in message or "1학생회관" in message:
        # (1) 1학생회관 메뉴 로드
        menu_dict = fixed_menu_1

        # (2) Retrieval context 생성
        retrieval_context = make_rag_context_from_fixed_menu(menu_dict)

    else:
        retrieval_context = make_rag_context_from_menu(message)

    return retrieval_context
#--------------------------------------------식단 끝-------------------------------------------------------------



#--------------------------------------------통학/버스-------------------------------------------------------------
def Busstop_Info(busstop):

    busstop_dic={}

    for i in Daejon_BUS_all_stops:
        if i["BUSSTOP_NM"]==busstop and busstop not in busstop_dic:
                busstop_dic[i["BUS_NODE_ID"]]=i["INF"]
    # print(busstop_dic)

    # --------------------- 버스 정류장 id 검색-------------------

    url = 'http://openapitraffic.daejeon.go.kr/api/rest/arrive/getArrInfoByStopID'
    result=[]

    for item in busstop_dic:
        params ={'serviceKey' : BUS_KEY, 'BusStopID' : item }
        r = requests.get(url, params=params, timeout=20);
        r.raise_for_status();
        r.encoding = "utf-8"
        info=busstop_dic[item]
        root = ET.fromstring(r.text)
        result_dic={}
        result_dic[info]=[]
        for it in root.findall(".//itemList"):

            result_dic[info].append({
                "버스번호": it.findtext("ROUTE_NO"),
                "남은시간": it.findtext("EXTIME_MIN"),
                "마지막 정류장": it.findtext("DESTINATION"),
                "ROUTE_TP":it.findtext("ROUTE_TP"),
            })
        result.append(result_dic)
    return result

def get_bus_number(message):
    if re.search(r'(M1)', message):
        target_bus = 'M1'
    elif re.search(r'(특)', message):
        target_bus = '특구1'
    else:
        bus_number = re.search(r'(\d{1,4})\s*번?', message)
        target_bus = bus_number.group(1) if bus_number else None

    return target_bus

def rag_answer_from_bus(message):
    bus_route=bus_stops
    bus_number = get_bus_number(message)

    if re.search(r"(노선|경로|어디|어느|정보)",message):
        #버스의 노선을 물어볼 때
        # ex) 48번 버스 노선 어떻게 돼?, 어디로 가?, 경로 알려줘
        result = []
        if bus_number is None:
            if re.search(r"(셔틀|순환|캠퍼스)", message):
                if re.search(r"(보운|교외)",message):
                    bus_number = "캠퍼스순환"
                else:
                    bus_number = "교내순환"
        bus_id = str(bus_number)

        if bus_number=='캠퍼스순환' or  bus_number=="교내순환":
            for section, content in bus_route.items():
                if isinstance(content, dict) and bus_id == section:
                    result.append(f"버스번호: {bus_id} 노선: {content}")
        else:
            for section, content in bus_route.items():
                if isinstance(content, dict) and bus_id in content:
                    result.append(f"버스번호: {bus_id} 노선: {content[bus_id]}")


        return "\n".join(result)

    else:# 정류장의 정보(버스 종류, 버스 위치)를 물어볼 때
        # ex) 정문 48번 버스 언제와?, 정문 버스들 남은 도착시간 알려줘
        location = None
        #정류소 위치
        if re.search(r"(충남대학교|입구|정문|학교정문|앞)", message):
            location = ["충남대학교","충남대학교입구"]
        elif re.search(r"(서문|서쪽)", message):
            location = ["충대서문"]
        elif re.search(r"(동문|동쪽|농대)", message):
            location = ["충대농대종점"]
        elif re.search(r"(순환|교내|학교셔틀|교내셔틀)", message):
            location = ["교내순환"]
        elif re.search(r"(산학연)", message):
            location = ["충남대산학연"]
        elif re.search(r"(중도|도서관|중앙도서관)", message):
            location = ["충남대도서관"]
        elif re.search(r"(세무서|북대전|보훈|요양원)", message):
            location = ["북대전세무서/보훈요양원"]
        elif re.search(r"(수의)", message):
            location = ["충대수의대"]
        elif re.search(r"(궁동교)", message):
            location = ["궁동교"]
        elif re.search(r"(장대|네거리)", message):
            location = ["장대네거리"]
        elif re.search(r"(다솔)", message):
            location = ["다솔아파트"]
        elif re.search(r"(궁동)", message):
            location = ["궁동"]
        elif re.search(r"(한빛|아파트)", message):
            location = ["한빛아파트"]

        print("location",location,"\n","bus_number",bus_number)
        if location is None:
            return "오류가 발생했습니다.🚨"

        else:
            StopInfo = {}
            for loc in location:
                result = Busstop_Info(loc)
                for i in result:
                    for section, content in i.items():
                        if bus_number:
                            for info in content:
                                if info['버스번호'] == bus_number:
                                    StopInfo[section] = []
                                    StopInfo[section].append(info)
                        else:
                            StopInfo[section] = content
            for section, content in StopInfo.items():
                for c in StopInfo[section]:
                    if c["ROUTE_TP"] == '5  ':
                        c['버스번호'] = '마을' + c['버스번호']
                    c.pop("ROUTE_TP")

            return str(StopInfo)


# 정류장 운행 정보
# 버스 노선

#-------------------------------------------------버스 end------------------------------------------------



#-------------------------------------------------졸업 요건----------------------------------------------------
def get_from_graduate(message):

    # 1) 파일 경로(프로젝트 구조에 맞게 조정)
    data_file = Path(__file__).resolve().parents[1] / "rag_data" / "graduation_requirements" / "graduation_RAG.json"
    with open(data_file, encoding="utf-8") as f:
        rag = json.load(f)  # <- 최상위에 "경영학부", "경제학과" 등이 바로 옴

    # 2) 모든 학과/학부 키 수집
    departments = list(rag.keys())  # 예: ["경영학부", "경제학과", ...]

    # 3) 별칭 생성(“경제학과” → “경제”, “경제학”, “경제학과” 등)
    suffixes = ["학과", "학부", "대학", "교육과"]
    dept_alias = {}
    for name in departments:
        aliases = {name}
        for suf in suffixes:
            if name.endswith(suf) and len(name) > len(suf):
                base = name[:-len(suf)]
                # 흔히 쓰는 변형들 추가
                aliases.update({
                    base,            # "경제"
                    base + "학",     # "경제학"
                    base + "과",     # "경제과" (안 쓰일 수도 있지만 안전하게)
                    base + "학과",   # "경제학과"
                    base + "학부",   # "경제학부"
                })
        # 공백 제거 버전도 매칭에 쓰도록 추가
        no_space = {a.replace(" ", "") for a in aliases}
        dept_alias[name] = aliases | no_space

    # 4) 메시지 정규화(공백 제거)
    msg_norm = re.sub(r"\s+", "", message)

    # 5) 메시지에서 학과/학부 키 찾기(매칭되면 그 학과 전체 반환)
    dept_key = next(
        (k for k, als in dept_alias.items() if any(a in msg_norm for a in als)),
        None
    )
    if not dept_key:
        # 못 찾으면 힌트 주거나 전체 키 나열해서 선택하도록 할 수 있음
        return "질문에서 학과/학부명을 찾지 못했어요. 예: '경제학과 졸업요건 알려줘'처럼 학과명을 포함해 주세요."

    # 6) 해당 학과의 모든 데이터 그대로 반환(문자열로)
    dept_data = rag.get(dept_key, {})
    return json.dumps({dept_key: dept_data}, ensure_ascii=False, indent=2)
#---------------------------------------------졸업요건끝-------------------------------------------------

#---------------------------------------------학사일정---------------------------------------------------
def rag_answer_from_academic_calendar(message):
    data_file = Path(__file__).resolve().parents[1] / "rag_data" / "calendar" / "academic_calendar.json"
    with open(data_file, encoding="utf-8") as f:
        calendar_data = json.load(f)
    date=get_date(message)
    result=f"기준 날짜:{date} 실제 학사일정 데이터:{calendar_data}"
    return result
#----------------------------------------------학사일정끝--------------------------------------------------


#----------------------------------------------공지사항----------------------------------------------------
def rag_answer_for_notices(user_message, top_k=3):
    data_file = Path(__file__).resolve().parents[1] / "rag_data" / "notice" / "notices.json"
    with open(data_file, encoding="utf-8") as f:
        notice_data = json.load(f)["data"]

    user_message_lower = user_message.lower()

    # 간단한 keyword matching 기반 검색 (추후 BM25나 FAISS로 교체 가능)
    matches = []
    for item in notice_data:
        score = 0
        title = item["title"].lower()
        content = item["content"].lower()

        for word in user_message_lower.split():
            if word in title:
                score += 50
            if word in content:
                score += 5

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
#---------------------------------------------공지사항 끝-------------------------------------------


#---------------------------------------------답변하기--------------------------------------------
#분류기를 사용하여 topic을 우선 추출
def extract_topic_from_message(message):

    # 자동으로 저장된 토크나이저 타입을 불러옴
    inputs = tokenizer_classification(message, return_tensors="pt", truncation=False, padding=True)
    NOTICE_KEYS = [
        "공지", "공지사항", "안내", "알림", "모집", "신청", "접수", "선발", "채용", "공고",
        "변경 안내", "유의", "첨부", "붙임", "파일", "다운로드", "게시", "발표","홍보"
    ]
    # 모델 예측
    with torch.no_grad():
        outputs = model_classification(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    id2label = {0: '버스/통학', 1: '식단', 2: '졸업요건', 3: '학교공지사항', 4: '학사일정'}
    topic = id2label[predicted_class]
    if topic=='학사일정' and any(k in message for k in NOTICE_KEYS):
        topic="학교공지사항"
    return topic

def Chatmodel(message,rag,topic):
    # 1) 토큰/프로바이더 설정

    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN, provider=PROVIDER)
    m=(message or "").strip()
    t=(topic or "").strip()
    r=(rag or "").strip()
    system_promt=dedent(f"""
        너는 학교 information 데스크에 있는 안내원처럼 친절하고 상세히 설명해주는 AI야
        첫 문장에는 ""{m}(이)라고 물으셨다면 👊{t}👊에 대해 궁금하시군요 😁!"를 꼭 넣어줘
        !중요 {r}의 내용을 최우선으로 참고하고 해당 내용이 완전하지 않으면 근본적인 내용은 바뀌지 않는 선에서 조금 더 유추해 보강해줘, 이거에 대한 언급은 따로 출력하지마
        !중요 {t}가 '졸업요건'일 경우: 
            1.표 형식과 같이 가독성이 좋게 출력해줘
            2.{m}에서 특정한 부분에 대해 물었다면 {r}에서 해당 하는 부분만 짚어서 설명에 요약까지 해줘
        !중요 {t}가 '식단'일 경우 :아래와 같은 조건을 만족해줘
            1.오류 발생해서 정보를 가져오지 못했으면 오류가 발생했다는 문구만을 출력해줘 
            2.식단을 찾았는데 운영 안 함이라는 결과만 있다면 해당 요일은 휴일이거나 해당 데이터가 아직 업데이트 되지 않았을 수 있다고 언급해줘
            3.식단을 출력할 때 해당 식단의 날짜도 같이 출력해줘
        !중요 {t}가 '통학/버스'일  경우 :아래와 같은 조건을 만족해줘
            1.만약 에러 문구가 {r}에 있으면 정확한 정류장 이름 및 버스 번호를  입력해 달라는 느낌으로 너가 스스로출력해줘, 반드시 {t}가 '통학/버스'일때만 출력해줘
            2.{r}은 문자열이지만 dictionary형태야 이를 가독성 좋게 표 형식으로 출력하고 이에 대한 언급은 하지마, 이 때 모든  키 값들은 정류장의 위치이므로 표의 제목으로 사용하되 표 안에는 표시하지마
            3.{r}에 버스번호에 해당하는 값은 숫자 뒤에 '번', 남은시간 숫자 뒤에 '분'을 꼭 붙여서 출력하고 이에 대한 언급은 하지마
            4.{m}에서 마을버스에 대해 물으면 {r}에서 마을버스에 해당하는 부분만 출력해줘 
        !중요 {t}가 '학사일정'일 경우 :아래와 같은 조건을 만족해줘
            1.기준날짜를 바탕으로 실제 학사일정 데이터에서 {m}에서 궁금해하는 일정 알려줘
        !중요 {t}가 '학사공지일 경우: 아래와 같은 조건을 만족해줘
            1.현재 학사공지는 학교 종합 공지만 지원되므로 단과대 공지사항을 찾는 경우는 아래의 조건에 맞는 문의 관련 문구를 출력해줘 
            2.만약 {m}에서 예를 들어 최신 사항을 알려달라고 언급한다면 {r}에서 날짜를 보고 그에 맞는 데이터를 출력해줘
        !중요 {m}에서 사용자가 표현을 어떻게 해서 답을 해달라고 하면 그에 맞는 표현으로 답변을 표현하고 이에 맞게 처리했다고 언급해줘
        !중요 문의 관련 문구를 출력 할 때는 https://plus.cnu.ac.kr/html/kr/  충남대 홈페이지 또는 해당 과 사무실에 문의하라는 말을 너가 알이서 부드럽게 바꿔서 전달해줘.
        !중요 필요한 것 있으면 언제든지 물어봐달라라는 내용과 자세하게 입력해주시면 정확도가 더 올라가는다는 내용을 너가 스스로 표현 정해서 맨 마지막에 언급해주고 이모티콘도 적극 활용해줘
        """)

    messages = [
        {"role": "system", "content": system_promt},
        {"role": "user", "content": f"질문{m}+참고데이터{r}"}
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
        rag = rag_answer_from_bus(message)

    elif topic == "졸업요건":
        rag = get_from_graduate(message)

    elif topic == "학사일정":
        rag = rag_answer_from_academic_calendar(message)

    elif topic == "학교공지사항":
        rag = rag_answer_for_notices(message)              #
    print(topic)
    print(rag)

    response=Chatmodel(message,rag,topic)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history