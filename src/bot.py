import os, json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
load_dotenv()
import os, json, pprint

token=os.getenv("TOKEN")
# 1. 모델 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
roberta_path = "spidyun/roberta-kor-finetuned"
# kogpt2_path = os.path.join(base_dir, "..", "kogpt2-finetuned")

# 2. 라벨 ID → 토픽 문자열 매핑
label_map = {
    0: "졸업요건",
    1: "학교 공지사항",
    2: "학사일정",
    3: "식단 안내",
    4: "통학/셔틀 버스"
}

# 3. 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. 모델 로드
try:
    roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_path,token=token,use_fast=True)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_path,token=token).to(device).eval()
except TypeError:
# 구버전 호환 (token 인자 미지원)
    roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_path, use_auth_token=token, use_fast=True)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_path, use_auth_token=token).to(
        device).eval()
# kogpt2_tokenizer = AutoTokenizer.from_pretrained(kogpt2_path)
# kogpt2_model = AutoModelForCausalLM.from_pretrained(kogpt2_path).to(device)
# generator = pipeline("text-generation", model=kogpt2_model, tokenizer=kogpt2_tokenizer, device=0 if torch.cuda.is_available() else -1)

# # 5. 데이터 경로
# input_path = os.path.join(base_dir, "..", "data", "cls_test.json")
# output_path = os.path.join(base_dir, "..","outputs", "output.json")

# # 6. JSON 데이터 로드
# with open(input_path, "r", encoding="ut[entry.get("question", "").strip() for entry in data if entry.get("question", "").strip()]f-8") as f:
#     data = json.load(f)
#


questions = "인공지능학과 졸업요건 알려줘"
# if not questions:
#     print("❗ 유효한 질문이 없습니다.")
#     exit()

# 7. Roberta로 topic 예측 (batch 처리)
inputs = roberta_tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to(device)
with torch.no_grad():
    logits = roberta_model(**inputs).logits
    pred_ids = logits.argmax(dim=-1).cpu().tolist()
topics = [label_map.get(pred_id, "기타") for pred_id in pred_ids]

# 8. 프롬프트 생성 (KoGPT2용)
prompts = [f"[TOPIC] {topic} [USER] {question} [SEP] [BOT]" for topic, question in zip(topics, questions)]
print(prompts)

# # 9. KoGPT2로 답변 생성 (batch 처리)
# generated_outputs = generator(
#     prompts,
#     max_new_tokens=50,
#     num_return_sequences=1,
#     do_sample=True,
#     pad_token_id=kogpt2_tokenizer.eos_token_id
# )

# 10. 결과 정리
# results = []
# for question, prompt, topic, gen in zip(questions, prompts, topics, generated_outputs):
#     generated_text = gen[0]["generated_text"]  # ✅ 여기 수정됨
#     answer = generated_text.replace(prompt, "").strip()
#
#     if not answer or len(answer) < 5:
#         answer = "죄송합니다. 답변을 생성할 수 없습니다."
#
#     results.append({
#         "question": question,
#         "topic": topic,
#         "answer": answer
#     })
#
# # 11. 결과 저장
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(results, f, ensure_ascii=False, indent=2)
#
# print(f"✅ 총 {len(results)}건 질문 처리 완료 → {output_path}")