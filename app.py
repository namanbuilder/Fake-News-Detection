import streamlit as st
from predict import predict

st.set_page_config(page_title="Fake News Detector", layout="centered")

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {font-family: 'Inter', sans-serif;}
.stApp {background: #f6f7fb;}
.block-container {max-width: 760px; padding-top: 3rem; padding-bottom: 4rem;}
#MainMenu, footer, header {visibility: hidden;}

/* Header */
.eyebrow {
    display: inline-block; font-size: .72rem; font-weight: 600; letter-spacing: .12em;
    text-transform: uppercase; color: #4f46e5; background: #eef2ff;
    padding: .35rem .8rem; border-radius: 999px; margin-bottom: 1rem;
}
.title {
    font-size: 2.8rem; font-weight: 800; line-height: 1.1; color: #0f172a;
    letter-spacing: -.03em; margin: 0 0 .6rem;
}
.title span {
    background: linear-gradient(90deg, #4f46e5, #9333ea);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.subtitle {font-size: 1.05rem; color: #64748b; margin-bottom: 2rem; max-width: 560px;}

/* Inputs */
.stTextInput label, .stTextArea label {
    font-weight: 600; font-size: .85rem; color: #334155;
}
.stTextInput input, .stTextArea textarea {
    background: #ffffff !important; border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important; color: #0f172a !important; font-size: .98rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #6366f1 !important; box-shadow: 0 0 0 4px rgba(99,102,241,.12) !important;
}

/* Buttons */
div.stButton > button {
    border-radius: 12px; font-weight: 600; font-family: 'Inter', sans-serif;
    transition: all .15s ease;
}
div.stButton > button[kind="primary"] {
    width: 100%; padding: .75rem; border: none; color: #fff; font-size: 1rem;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    box-shadow: 0 8px 20px rgba(79,70,229,.28);
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px); box-shadow: 0 12px 26px rgba(79,70,229,.36); color: #fff;
}
div.stButton > button[kind="secondary"] {
    background: #fff; color: #475569; border: 1.5px solid #e2e8f0; font-size: .85rem;
}
div.stButton > button[kind="secondary"]:hover {
    border-color: #6366f1; color: #4f46e5;
}

/* Result */
.result {
    background: #fff; border-radius: 18px; padding: 1.8rem; margin-top: 1.6rem;
    box-shadow: 0 10px 30px rgba(15,23,42,.06); border-left: 6px solid;
}
.result.real {border-color: #10b981;}
.result.fake {border-color: #ef4444;}
.result .tag {
    font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
}
.result.real .tag {color: #059669;}
.result.fake .tag {color: #dc2626;}
.result .verdict {
    font-size: 2rem; font-weight: 800; color: #0f172a; letter-spacing: -.02em; margin: .2rem 0 1rem;
}
.bar-wrap {background: #eef0f5; border-radius: 999px; height: 10px; overflow: hidden;}
.bar {height: 100%; border-radius: 999px;}
.real .bar {background: linear-gradient(90deg, #34d399, #10b981);}
.fake .bar {background: linear-gradient(90deg, #f87171, #ef4444);}
.bar-meta {
    display: flex; justify-content: space-between; margin-top: .5rem;
    font-size: .82rem; color: #64748b; font-weight: 500;
}
.note {
    margin-top: 1.2rem; font-size: .82rem; color: #64748b; line-height: 1.5;
    padding-top: 1rem; border-top: 1px solid #eef0f5;
}
.hint {font-size: .8rem; color: #94a3b8; margin: .4rem 0 1rem;}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="eyebrow">NLP Classifier</div>
<div class="title">Fake News <span>Detector</span></div>
<div class="subtitle">Paste a headline and article text. The model analyzes writing patterns
and estimates whether the story reads as real or fake news.</div>
""", unsafe_allow_html=True)

# ---------- State ----------
st.session_state.setdefault("title", "")
st.session_state.setdefault("body", "")

def load_example(t, b):
    st.session_state.title = t
    st.session_state.body = b

def clear_all():
    st.session_state.title = ""
    st.session_state.body = ""

# ---------- Input ----------
st.text_input("Headline (optional)", key="title", placeholder="Enter the news headline")
st.text_area("Article text", key="body", height=220,
             placeholder="Paste the full article text here")

c1, c2, c3 = st.columns(3)
c1.button("Sample: real-style", on_click=load_example, use_container_width=True, args=(
    "Government announces new infrastructure funding",
    "The finance ministry announced on Monday a new funding package for road and rail "
    "projects, with officials saying the plan will be reviewed by parliament next month."))
c2.button("Sample: fake-style", on_click=load_example, use_container_width=True, args=(
    "SHOCKING: Doctors HATE this one trick!!!",
    "You won't believe what they are hiding from you! Secret miracle cure leaked, "
    "share before it gets deleted by the elites!!!"))
c3.button("Clear", on_click=clear_all, use_container_width=True)

st.markdown('<div class="hint">Longer, complete articles give more reliable results.</div>',
            unsafe_allow_html=True)

analyze = st.button("Analyze article", type="primary")

# ---------- Result ----------
# ---------- Result ----------
if analyze:
    content = f"{st.session_state.title} {st.session_state.body}".strip()
    words = len(content.split())

    if words < 5:
        st.warning("Please enter a longer piece of text (at least 5 words).")
    else:
        with st.spinner("Analyzing..."):
            try:
                label, conf = predict(content)
            except Exception as e:
                st.exception(e)
                st.stop()

        is_real = str(label).strip().lower() == "real"
        css = "real" if is_real else "fake"
        verdict = "Likely Real" if is_real else "Likely Fake"

        bar = ""
        if conf is not None:
            conf = float(conf)
            if conf > 1:
                conf /= 100
            conf = max(0.0, min(conf, 1.0))
            bar = (
                f'<div class="bar-wrap"><div class="bar" style="width:{conf*100:.1f}%"></div></div>'
                f'<div class="bar-meta"><span>Confidence</span><span>{conf*100:.1f}%</span></div>'
            )

        short = " Short text gives less reliable results." if words < 30 else ""

        html = (
            f'<div class="result {css}">'
            f'<div class="tag">Result</div>'
            f'<div class="verdict">{verdict}</div>'
            f'{bar}'
            f'<div class="note">{words} words analyzed. This is a statistical model based on '
            f'writing patterns, not a fact-checker. Always verify with trusted sources.{short}</div>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
        