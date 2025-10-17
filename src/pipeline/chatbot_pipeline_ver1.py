import json, re, urllib.parse, requests
import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer,AutoModelForSequenceClassification, GPT2LMHeadModel,StoppingCriteria, StoppingCriteriaList
data_updated = False


load_dotenv()
token=os.getenv("TOKEN")

model_dir = "spidyun/chatbot-roberta"
chat_dir = "spidyun/kogpt2-finetuned"

#분류 모델 - RobertA-large
tokenizer_classification = AutoTokenizer.from_pretrained(model_dir, token=token, use_fast=True)
model_classification = AutoModelForSequenceClassification.from_pretrained(model_dir, token=token).eval()

#챗 봇 모델 - KoGPT2
tokenizer = AutoTokenizer.from_pretrained(chat_dir, token=token, use_fast=True)
model = GPT2LMHeadModel.from_pretrained(chat_dir, token=token).eval()

# 2문장만 답하도록
class TwoSentenceStopper(StoppingCriteria):
    def __init__(self, prompt_len, tokenizer, target=2):
        self.prompt_len = prompt_len
        self.tok = tokenizer
        self.target = target
    def __call__(self, input_ids, scores, **kwargs):
        gen = input_ids[0][self.prompt_len:]
        text = self.tok.decode(gen, skip_special_tokens=True)

        # ✅ 맨 앞 [BOT] (또는 다른 태그) 제거
        text = re.sub(r'^\s*\[(?:BOT|USER|TOPIC|SEP)\]\s*', '', text)

        # 문장 끝 패턴: ., !, ?, 그리고 한국어 흔한 어말 '다.', '요.'
        cnt = len(re.findall(r'(?:[.!?])|(?:다\.|요\.)', text))
        return cnt >= self.target

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

# 최종 응답
def respond(message, history=None):
    topic=extract_topic_from_message(message)

    final_prompt = (
        f"[TOPIC] {topic} [USER] {message} [SEP]"
    )

    print(final_prompt)

    input_ids = tokenizer(final_prompt, return_tensors="pt").input_ids
    stopper = StoppingCriteriaList([TwoSentenceStopper(input_ids.shape[-1], tokenizer, target=2)])

    outs = model.generate(
        input_ids,
        max_new_tokens=120,  # 2문장 나올 만큼 여유
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.5,
        stopping_criteria=stopper,
        pad_token_id=tokenizer.eos_token_id
    )


    raw = tokenizer.decode(outs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    raw = re.sub(r'^\s*\[(?:BOT|USER|TOPIC|SEP)\]\s*', '', raw)
    response = raw.strip()

    if history is None:
        history = []


    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    print("response:", response)

    return "", history




