# # Use a pipeline as a high-level helper
# from transformers import pipeline
# import os
# from openai import OpenAI
#
# pipe = pipeline("text-generation", model="openai/gpt-oss-120b")
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# pipe(messages)
#
# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
# model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-120b")
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# inputs = tokenizer.apply_chat_template(
# 	messages,
# 	add_generation_prompt=True,
# 	tokenize=True,
# 	return_dict=True,
# 	return_tensors="pt",
# ).to(model.device)
#
# outputs = model.generate(**inputs, max_new_tokens=40)
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
# os.environ['HF_TOKEN'] = 'YOUR_TOKEN_HERE'
#
#
#
# client = OpenAI(
#     base_url="https://router.huggingface.co/v1",
#     api_key=os.environ["HF_TOKEN"],
# )
#
# completion = client.chat.completions.create(
#     model="openai/gpt-oss-120b",
#     messages=[
#         {
#             "role": "user",
#             "content": "What is the capital of France?"
#         }
#     ],
# )
#
# print(completion.choices[0].message)
# api_only.py
import os
from openai import OpenAI

# 1) 토큰 먼저 설정 (환경변수 권장)
# macOS/Linux 터미널에서:  export HF_TOKEN=hf_********
# Windows PowerShell:      $env:HF_TOKEN="hf_********"
# 코드에서 할 때는 아래 줄 활성화:
# os.environ["HF_TOKEN"] = "hf_********"

# 2) HF Inference Router를 OpenAI 호환으로 사용
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["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"],
)

# 3) 120B는 서버 측 제공/자원 이슈가 잦으니, 우선 20B로 테스트 권장
resp = client.chat.completions.create(
    model="openai/gpt-oss-20b",  # 120b 대신 20b 권장
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=64,
)

print(resp.choices[0].message.content)