import gradio as gr
from chatbot_pipeline import respond  # 그대로 사용
from academic_crawler import crawl_academic_calendar
from notice_crawler import CNUNoticeCrawler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project/ 기준
WEB_DIR = ROOT / "web"
CSS_PATH = WEB_DIR / "page.css"
HERO_PATH = WEB_DIR / "page.html"


# ===== 선택(그대로 두면 비활성) =====
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

css_code = CSS_PATH.read_text(encoding="utf-8")
hero_html = HERO_PATH.read_text(encoding="utf-8")

def reset():
    return "", []

with gr.Blocks(css=css_code, fill_height=False, title="CNU ChatBot (Gradient UI)") as demo:

    # 상단 Hero
    gr.HTML(hero_html)

    # 채팅 패널
    with gr.Column(elem_id="chat-panel"):
        gr.Markdown(
            '<div class="container" style="text-align:center; margin-bottom:.6rem;">'
            '<h3 class="gradient-text" style="font-size:1.75rem; font-weight:800; letter-spacing:-0.01em;">💬 대화창</h3>'
            '<p class="lead" style="margin-top:.25rem;">질문을 입력하고, <b>질문 보내기</b>를 눌러보세요.</p>'
            '</div>',
            elem_id="panel-title"
        )

        chatbot = gr.Chatbot(
            elem_id="chatbox",
            label=None,
            type="messages",
            height=520,
            show_copy_button=True
        )

        # ✅ 입력줄: [텍스트박스][버튼 세로 묶음]
        with gr.Row(elem_id="composer"):
            msg = gr.Textbox(
                placeholder="질문을 입력해 주세요...",
                lines=1,
                elem_id="msgbox",
                show_label=False
            )
            with gr.Column(elem_id="actions", scale=0, min_width=10):
                submit_btn = gr.Button("질문 보내기", elem_id="submit-btn")
                clear_btn = gr.Button("초기화", elem_id="clear-btn")

        # 동작 연결
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
        clear_btn.click(reset, outputs=[msg, chatbot])

    # 푸터
    gr.HTML(
        '<div style="text-align:center; padding:1rem 0 2rem; color:#6b7280;">'
        '<small>© 2025 CNU ChatBot • UI styled with gradient & glassmorphism</small>'
        '</div>'
    )

demo = demo
if __name__ == "__main__":
    demo.launch()