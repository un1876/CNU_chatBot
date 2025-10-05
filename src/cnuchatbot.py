import gradio as gr
from chatbot_pipeline import respond  # 그대로 사용
from academic_crawler import crawl_academic_calendar
from notice_crawler import CNUNoticeCrawler

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

# ===== 스타일: Next.js 랜딩 느낌 + 버튼 세로 스택 =====
css_code = """
:root{
  --grad-from: #7c3aed; /* purple-600 */
  --grad-via : #ec4899; /* pink-500 */
  --grad-to  : #3b82f6; /* blue-500 */
  --ink-900  : #0f172a;
  --panel-bg : rgba(255,255,255,0.55);
  --panel-stroke: rgba(255,255,255,0.45);
}

/* 전체 배경 */
html, body{
  height:100%;
  margin:0; padding:0;
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, Helvetica, Arial, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
  background:
    radial-gradient(60rem 60rem at 10% 0%, rgba(124,58,237,0.13), transparent 60%),
    radial-gradient(60rem 60rem at 100% 20%, rgba(236,72,153,0.12), transparent 60%),
    linear-gradient(135deg, #faf5ff 0%, #fff1f7 50%, #eff6ff 100%);
  color: var(--ink-900);
}

/* 영웅영역(헤더) */
#hero{
  position: relative;
  overflow: hidden;
  padding: 5rem 1rem 2rem 1rem;
  text-align:center;
}
#hero .halo{
  position:absolute; inset:0;
  background: radial-gradient(40rem 40rem at 50% -10%, rgba(124,58,237,0.15), transparent),
              radial-gradient(40rem 40rem at -10% 30%, rgba(236,72,153,0.15), transparent),
              radial-gradient(40rem 40rem at 110% 60%, rgba(59,130,246,0.15), transparent);
  filter: blur(20px);
  z-index:-1;
}
.container{ max-width: 1100px; margin: 0 auto; }

/* 뱃지 아이콘 원 */
.badge{
  width:3rem;height:3rem; display:inline-flex; align-items:center; justify-content:center;
  border-radius:9999px;
  background: linear-gradient(135deg, var(--grad-from), var(--grad-via));
  box-shadow: 0 8px 30px rgba(124,58,237,0.25);
  margin-right: .75rem;
}

/* 그라데이션 텍스트 */
/* ✅ 안정화된 그라데이션 텍스트 + 브라우저 폴백 */
.gradient-text{
  background: 
  linear-gradient(169deg, rgba(124, 58, 237, 0.04), rgba(235, 71, 152, 0.57) 50%, rgba(122, 109, 213, 0.52) 82.35%, rgba(59, 130, 246, 0.24)) !important
  -webkit-background-clip: rgb(171, 136, 191);
  background-clip: text;
  -webkit-text-fill-color: rgb(171, 136, 191);  /* Safari/웹킷 필수 */
  color: transparent;                     /* 기타 브라우저 */
  display: inline-block;                  /* 배경 계산 안정화 */
  position: relative; z-index: 1;         /* 드문 겹침 이슈 예방 */
}

/* ⛑️ 폴백: 배경클립 미지원 환경에서는 그냥 단색으로 보여주기 */
@supports not (-webkit-background-clip: text) {
  .gradient-text{
    background: none !important;
    color: #ffffff !important; /* 보라색 단색 텍스트 */
  }
}

/* 보조 설명 */
.lead{ font-size:1.125rem; line-height:1.7; color:#475569; max-width:46rem; margin: 0.75rem auto 1rem; }

/* 버튼 스타일 */
a.btn, .btn{
  display:inline-flex; align-items:center; gap:.5rem;
  padding:.9rem 1.1rem; border-radius:0.875rem; text-decoration:none; font-weight:600;
  border: none; cursor:pointer;
  transition: transform .15s ease, box-shadow .2s ease, opacity .2s ease;
}
a.btn.primary, .btn.primary{
  color:#fff; background: linear-gradient(90deg, var(--grad-from), var(--grad-via));
  box-shadow: 0 10px 30px rgba(236,72,153,0.25);
}
a.btn.primary:hover, .btn.primary:hover{ transform: translateY(-1px); }
a.btn.ghost{ color:#1f2937; background:transparent; border:1px solid rgba(2,6,23,0.08); }
a.btn.ghost:hover{ background:rgba(2,6,23,0.035); }

/* 채팅 패널 (글래스모피즘) */
#chat-panel{
  width: min(100%, 1100px);
  margin: 0 auto 2.5rem auto;
  padding: 1.25rem;
  background: var(--panel-bg);
  border: 1px solid var(--panel-stroke);
  border-radius: 1.25rem;
  backdrop-filter: blur(10px);
  box-shadow: 0 15px 40px rgba(2,6,23,0.08);
}

/* Chatbot 박스 */
#chatbox{
  height: clamp(420px, 64vh, 680px);
  overflow: auto;
  background: rgba(255,255,255,0.7);
  border: 1px solid rgba(148,163,184,0.25);
  border-radius: 1rem;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.25);
}

/* ✅ 입력줄을 2열 그리드: [텍스트박스][버튼세로묶음] */
#composer{
  display: grid;
  grid-template-columns: 1fr auto;  /* ⬅️ 2열로 변경 */
  gap: .5rem;
  margin-top: .9rem;
  align-items: start;
}

/* ✅ 버튼 묶음(세로 스택) */
#actions{
  display: flex;
  flex-direction: column;   /* 세로 정렬 */
  gap: .5rem;
  align-items: stretch;
}

/* ✅ 텍스트박스 폭/높이 */
#msgbox{ width: 100%; min-width: 0; height: 100%; }
#msgbox textarea, #msgbox input{
  height: 100%;
  width: 100%;
  min-height: 44px;
  padding: .75rem 1rem;
  border-radius: .75rem;
  border: 1px solid rgba(17,24,39,.12);
  background: #fff;
  box-sizing: border-box;
}

/* Gradio 버튼 커스텀 */
#submit-btn > button, #clear-btn > button{
  border-radius:.75rem; height: 44px; padding: 0 1rem; font-weight:600;
  border: none;
}
#submit-btn > button{
  color:#fff!important;
  background: linear-gradient(90deg, var(--grad-from), var(--grad-via))!important;
}
#submit-btn > button:hover{ opacity:.95; }
#clear-btn > button{
  background: transparent!important; color:#111827!important;
  border:1px solid rgba(17,24,39,.12)!important;
}
#clear-btn > button:hover{ background: rgba(17,24,39,.04)!important; }

/* 말풍선 스타일 */
#chatbox .message, #chatbox .wrap, #chatbox [class*="message"]{
  border-radius: .9rem !important;
}
#chatbox [data-testid="bot"] .message, #chatbox .bot{
  background: #ffffff !important;
  border: 1px solid rgba(148,163,184,0.25) !important;
}
#chatbox [data-testid="user"] .message, #chatbox .user{
  background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(236,72,153,0.12)) !important;
  border: 1px solid rgba(124,58,237,0.25) !important;
}

/* 스크롤바 */
#chatbox::-webkit-scrollbar{ width:10px; }
#chatbox::-webkit-scrollbar-thumb{
  background: linear-gradient(135deg, rgba(124,58,237,.35), rgba(236,72,153,.35));
  border-radius:999px;
}
#chatbox::-webkit-scrollbar-track{ background: rgba(0,0,0,0.04); border-radius:999px; }

/* 반응형: 모바일에서도 세로 스택 유지 */
@media (max-width: 640px){
  #hero{ padding:3.5rem 1rem 1.5rem; }
  .lead{ font-size:1rem; }
  #chat-panel{ padding:.9rem; }
  #composer{ grid-template-columns: 1fr; } /* 모바일은 버튼 묶음이 텍스트박스 아래로 */
}
"""

# ===== 상단 Hero =====
hero_html = """
<section id="hero">
  <div class="halo"></div>
  <div class="container">
    <div style="display:flex; align-items:center; justify-content:center; gap:.75rem; margin-bottom:.5rem;">

      <h1 class="gradient-text" style="font-weight:800; font-size: clamp(2rem, 3vw, 3.25rem); margin:0;">
        CNU ChatBot🤖
      </h1>

    </div>
    <p class="lead">교내 졸업요건, 학사일정, 학사공지, 운영 버스, 식당에 대해 대화해보세요!</p>
    <div style="display:flex; gap:.6rem; justify-content:center; flex-wrap:wrap; margin-top: .6rem;">
      <a class="btn primary" href="#chat-panel">지금 시작하기</a>
    </div>
  </div>
</section>
"""

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