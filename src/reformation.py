# pip install -U "huggingface_hub>=0.33" python-dotenv

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


def reform(query):
    # 1) 토큰/프로바이더 설정
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")  # .env에 HF_TOKEN=hf_***** 로 저장
    PROVIDER = os.getenv("HF_PROVIDER", None)  # 'fireworks-ai' | 'together' | 'hf-inference' | None

    # 2) 사용할 모델 (120B가 느리면 20B 권장: "openai/gpt-oss-20b")
    MODEL_ID = os.getenv("MODEL_ID", "openai/gpt-oss-120b")

    # 3) 클라이언트 생성
    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN, provider=PROVIDER)

    # 4) 대화 메시지(입력)
    messages = [
        {"role": "system", "content": """
        "너는 학교 infromation 데스크에 있는 안내원처럼 친절하고 상세히 설명해주는 AI야"
        "너는 글을 입력으로 받으면 해당 글이 완전하지 않더라도 그 글들을 기바능로 유추해서 상세하게 완성해야해"
    
        """},
        {"role": "user", "content": query},
    ]

    # 5) API 호출(출력)
    resp = client.chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.2,
    )

    # 6) 응답 반환
    return resp.choices[0].message.content
