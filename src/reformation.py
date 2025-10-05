# pip install -U "huggingface_hub>=0.33" python-dotenv

import os
from textwrap import dedent

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


def reform(message,query,topic):
    # 1) 토큰/프로바이더 설정
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")  # .env에 HF_TOKEN=hf_***** 로 저장
    PROVIDER = os.getenv("HF_PROVIDER", None)  # 'fireworks-ai' | 'together' | 'hf-inference' | None

    # 2) 사용할 모델 (120B가 느리면 20B 권장: "openai/gpt-oss-20b")
    MODEL_ID = os.getenv("MODEL_ID", "openai/gpt-oss-120b")

    # 3) 클라이언트 생성
    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN, provider=PROVIDER)
    m=(message or "").strip()
    t=(topic or "").strip()
    q=(query or "").strip()
    system_promt=dedent(f"""
        너는 학교 infromation 데스크에 있는 안내원처럼 친절하고 상세히 설명해주는 AI야
        너는 글을 입력으로 받으면 해당 글이 완전하지 않더라도 그 글들을 기반으로 유추해서 상세하게 완성해야해
        첫 문장에는 ""{m}(이)라고 물으셨다면 👊{t}👊에 대해 궁금하시군요 😁!"를 꼭 넣어줘
        만{q}에서 들어온 글자가 {t}와 맞지 않는 다른 주제의 얘기인 부분은 과감하게 제거해
        {m}에서 사용자가 표현을 어떻게 해서 답을 해다랄고 하면 그에 맞는 표현으로 최종 출력해주고 맨 마지막줄에 그에 맞는 표현도 언급해줘
        !중요 전달해야하는 부분이 끝나면 더이상 출력하지 말고 멈춰
        """)
    # 4) 대화 메시지(입력)
    messages = [
        {"role": "system", "content": system_promt},
        {"role": "user", "content": query},
    ]

    # 5) API 호출(출력)
    resp = client.chat_completion(
        messages=messages,
        max_tokens=2048,
        temperature=0.2,
    )

    # 6) 응답 반환
    return resp.choices[0].message.content
