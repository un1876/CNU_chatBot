import gradio as gr
from chatbot_pipeline import respond  # 핵심 respond 함수만 import
from academic_crawler import crawl_academic_calendar
from notice_crawler import CNUNoticeCrawler

data_updated = False
def update_all_data_once():
    global data_updated
    if not data_updated:
        print("📌 데이터 최초 갱신 중...")
        crawl_academic_calendar()
        update_notices()
        data_updated = True
def update_notices():
    crawler = CNUNoticeCrawler()
    notices = crawler.crawl_notices(max_pages=10)
    crawler.save_to_json(notices, filename="notices.json")

# update_all_data_once()
css_code = """
html, body {
    margin: 0;
    padding: 0;
    height: 100%;
    background-image: url("https://images.unsplash.com/photo-1504674900247-0877df9cc836?auto=format&fit=crop&w=1600&q=80");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    font-family: 'Arial', sans-serif;
}
#chat-container {
    
    width: 100dvw;
    height: 90vh;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: rgba(255, 255, 255, 0.9);
    background-color: #f0f4f8;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    padding: 20px;
}

#chatbox {
    width: 100%;
    height: 70vh;  /* 전체 높이의 70% */
    max-height: 600px;
    overflow-y: auto;
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    margin: 0 auto;
}

#msgbox {
    width: 100%;
    margin-top: 12px;
}
/* submit 버튼 배경/테두리/글자색 */
#submit-btn > button {
  background-color: #B9EAF1 !important;
  border: 1px solid #B9EAF1 !important;
  color: #0F2E36 !important; /* 가독성 좋은 진한 색 */
}

/* 호버/포커스 상태 */
#submit-btn > button:hover,
#submit-btn > button:focus {
  background-color: #A7E2EA !important;
  border-color: #A7E2EA !important;
}
"""

def reset():
    return "", []

with gr.Blocks(css=css_code) as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=50): pass
        with gr.Column(scale=6, elem_id="chat-container"):
            gr.Markdown("## 💬 CNU_ChatBot")
            chatbot = gr.Chatbot(elem_id="chatbox", label="대화 내용", type='messages')
            msg = gr.Textbox(placeholder="질문을 입력해 주세요...", label="질문 입력", lines=1, elem_id="msgbox")
            with gr.Row():
                submit_btn = gr.Button("질문 보내기", variant="primary",elem_id="submit-btn")
                clear_btn = gr.Button("초기화",elem_id="clear-btn")

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
            clear_btn.click(reset, outputs=[msg, chatbot])
        with gr.Column(scale=1, min_width=50): pass

demo.launch()