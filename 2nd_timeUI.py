import streamlit as st
import joblib
import pandas as pd
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GROWTH MATRIX — NLP Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* Title */
h1 {
    font-family: 'Space Mono', monospace !important;
    font-size: 2rem !important;
    letter-spacing: -1px;
    color: #ffffff !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #161920;
    padding: 6px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: #8890a0;
    background: transparent;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #252a35 !important;
    color: #ffffff !important;
}

/* Cards */
.nlp-card {
    background: #161920;
    border: 1px solid #252a35;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 18px;
}

/* Result boxes */
.result-spam {
    background: linear-gradient(135deg, #3a1a1a, #2a1010);
    border: 1px solid #cc3333;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    color: #ff6b6b;
    letter-spacing: 1px;
}
.result-ham {
    background: linear-gradient(135deg, #0f2a1a, #0a1f14);
    border: 1px solid #2a8f50;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    color: #4ade80;
    letter-spacing: 1px;
}
.result-neutral {
    background: linear-gradient(135deg, #1a1f2e, #141824);
    border: 1px solid #3a4a6a;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    color: #93c5fd;
    letter-spacing: 1px;
}
.result-positive {
    background: linear-gradient(135deg, #0f2a1a, #0a1f14);
    border: 1px solid #2a8f50;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    color: #4ade80;
    letter-spacing: 1px;
}
.result-negative {
    background: linear-gradient(135deg, #3a1a1a, #2a1010);
    border: 1px solid #cc3333;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    color: #ff6b6b;
    letter-spacing: 1px;
}

/* Inputs */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: #1e2330 !important;
    border: 1px solid #2e3545 !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #4a90d9 !important;
    box-shadow: 0 0 0 2px rgba(74,144,217,0.15) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.4) !important;
}

/* Dataframe */
.stDataFrame {
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1117 !important;
    border-right: 1px solid #1e2330 !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #8890a0;
    font-size: 0.9rem;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #161920;
    border: 2px dashed #2e3545;
    border-radius: 12px;
    padding: 10px;
}

/* Section labels */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4a90d9;
    margin-bottom: 8px;
}

/* Divider */
hr {
    border-color: #1e2330 !important;
    margin: 24px 0 !important;
}

/* Badge */
.badge {
    display: inline-block;
    background: #1e2330;
    border: 1px solid #2e3545;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    color: #8890a0;
    font-family: 'Space Mono', monospace;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    return {
        "spam":     joblib.load(os.path.join(base, "spam_classifier.pkl")),
        "language": joblib.load(os.path.join(base, "lang_det.pkl")),
        "news":     joblib.load(os.path.join(base, "news_cat.pkl")),
        "review":   joblib.load(os.path.join(base, "review.pkl")),
    }

models = load_models()

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_badges = st.columns([3, 2])
with col_title:
    st.markdown("# 🔬 GROWTH MATRIX")
    st.markdown("**NLP Suite** — Intelligent Text Analysis Platform")
with col_badges:
    st.markdown("""
    <div style='text-align:right; padding-top:14px;'>
        <span class='badge'>v1.0</span>
        <span class='badge'>4 models</span>
        <span class='badge'>ML-powered</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🛡️ Spam Classifier",
    "🌐 Language Detection",
    "🍽️ Food Review Sentiment",
    "📰 News Classification",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Spam Classifier
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Spam Classifier")
    st.markdown("Detect whether a message is spam or legitimate.")
    st.markdown("---")

    # ── Single message ──
    st.markdown("<div class='section-label'>Single Message</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        msg = st.text_input("Enter message", placeholder="Type or paste a message here…", key="spam_input", label_visibility="collapsed")
    with col2:
        predict_btn = st.button("Analyse", key="spam_btn", use_container_width=True)

    if predict_btn:
        if not msg.strip():
            st.warning("⚠️ Please enter a message before analysing.")
        else:
            pred = models["spam"].predict([msg])
            if pred[0] == 0:
                st.markdown("<div class='result-spam'>🚨 SPAM DETECTED</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-ham'>✅ NOT SPAM — Looks Legitimate</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Bulk upload ──
    st.markdown("<div class='section-label'>Bulk Classification — Upload CSV</div>", unsafe_allow_html=True)
    st.caption("CSV file should have one message per row (no header needed).")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv", "txt"], key="spam_upload", label_visibility="collapsed")

    if uploaded_file:
        try:
            df_spam = pd.read_csv(uploaded_file, header=None, names=["Message"])
            df_spam = df_spam.dropna(subset=["Message"])
            preds = models["spam"].predict(df_spam["Message"])
            df_spam.index = range(1, len(df_spam) + 1)
            df_spam["Prediction"] = preds
            df_spam["Prediction"] = df_spam["Prediction"].map({0: "🚨 Spam", 1: "✅ Not Spam"})

            total   = len(df_spam)
            n_spam  = (preds == 0).sum()
            n_ham   = (preds == 1).sum()

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Messages", total)
            m2.metric("Spam",     n_spam,  delta=f"{n_spam/total*100:.1f}%",  delta_color="inverse")
            m3.metric("Not Spam", n_ham,   delta=f"{n_ham/total*100:.1f}%",   delta_color="normal")

            st.dataframe(df_spam, use_container_width=True)

            csv_out = df_spam.to_csv().encode("utf-8")
            st.download_button("⬇️ Download Results", csv_out, "spam_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — Language Detection
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Language Detection")
    st.markdown("Identify the language of any text input.")
    st.markdown("---")

    st.markdown("<div class='section-label'>Single Text</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        lang_input = st.text_input("Enter text", placeholder="Enter text in any language…", key="lang_input", label_visibility="collapsed")
    with col2:
        lang_btn = st.button("Detect", key="lang_btn", use_container_width=True)

    if lang_btn:
        if not lang_input.strip():
            st.warning("⚠️ Please enter some text.")
        else:
            pred = models["language"].predict([lang_input])
            st.markdown(f"<div class='result-neutral'>🌐 Detected Language: <strong>{pred[0]}</strong></div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<div class='section-label'>Bulk Detection — Upload CSV</div>", unsafe_allow_html=True)
    st.caption("CSV file should have one text per row (no header needed).")
    lang_file = st.file_uploader("Upload CSV", type=["csv", "txt"], key="lang_upload", label_visibility="collapsed")

    if lang_file:
        try:
            df_lang = pd.read_csv(lang_file, header=None, names=["Text"])
            df_lang = df_lang.dropna(subset=["Text"])
            df_lang["Detected Language"] = models["language"].predict(df_lang["Text"])
            df_lang.index = range(1, len(df_lang) + 1)

            lang_counts = df_lang["Detected Language"].value_counts()
            st.markdown(f"**{len(df_lang)} texts** processed across **{len(lang_counts)} languages**")
            st.dataframe(df_lang, use_container_width=True)

            csv_out = df_lang.to_csv().encode("utf-8")
            st.download_button("⬇️ Download Results", csv_out, "language_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — Food Review Sentiment
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Food Review Sentiment")
    st.markdown("Analyse whether a food review is positive or negative.")
    st.markdown("---")

    st.markdown("<div class='section-label'>Single Review</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        review_input = st.text_area("Enter review", placeholder="Write a food review here…", key="review_input", label_visibility="collapsed", height=100)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        review_btn = st.button("Analyse", key="review_btn", use_container_width=True)

    if review_btn:
        if not review_input.strip():
            st.warning("⚠️ Please enter a review.")
        else:
            pred = models["review"].predict([review_input])
            label = str(pred[0]).lower()
            if label in ["1", "positive", "pos"]:
                st.markdown("<div class='result-positive'>😋 POSITIVE REVIEW</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-negative'>😞 NEGATIVE REVIEW</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<div class='section-label'>Bulk Sentiment — Upload CSV</div>", unsafe_allow_html=True)
    st.caption("CSV file should have one review per row (no header needed).")
    review_file = st.file_uploader("Upload CSV", type=["csv", "txt"], key="review_upload", label_visibility="collapsed")

    if review_file:
        try:
            df_review = pd.read_csv(review_file, header=None, names=["Review"])
            df_review = df_review.dropna(subset=["Review"])
            preds = models["review"].predict(df_review["Review"])
            df_review.index = range(1, len(df_review) + 1)
            df_review["Sentiment"] = [
                "😋 Positive" if str(p).lower() in ["1", "positive", "pos"] else "😞 Negative"
                for p in preds
            ]

            pos = sum(1 for p in preds if str(p).lower() in ["1", "positive", "pos"])
            neg = len(preds) - pos
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Reviews", len(preds))
            m2.metric("Positive", pos)
            m3.metric("Negative", neg)

            st.dataframe(df_review, use_container_width=True)

            csv_out = df_review.to_csv().encode("utf-8")
            st.download_button("⬇️ Download Results", csv_out, "sentiment_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — News Classification
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### News Classification")
    st.markdown("Categorise news articles or headlines into topics.")
    st.markdown("---")

    st.markdown("<div class='section-label'>Single Headline / Article</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        news_input = st.text_area("Enter news text", placeholder="Paste a headline or article snippet…", key="news_input", label_visibility="collapsed", height=100)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        news_btn = st.button("Classify", key="news_btn", use_container_width=True)

    if news_btn:
        if not news_input.strip():
            st.warning("⚠️ Please enter some news text.")
        else:
            pred = models["news"].predict([news_input])
            st.markdown(f"<div class='result-neutral'>📰 Category: <strong>{pred[0]}</strong></div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<div class='section-label'>Bulk Classification — Upload CSV</div>", unsafe_allow_html=True)
    st.caption("CSV file should have one article/headline per row (no header needed).")
    news_file = st.file_uploader("Upload CSV", type=["csv", "txt"], key="news_upload", label_visibility="collapsed")

    if news_file:
        try:
            df_news = pd.read_csv(news_file, header=None, names=["Article"])
            df_news = df_news.dropna(subset=["Article"])
            df_news["Category"] = models["news"].predict(df_news["Article"])
            df_news.index = range(1, len(df_news) + 1)

            cat_counts = df_news["Category"].value_counts()
            st.markdown(f"**{len(df_news)} articles** classified into **{len(cat_counts)} categories**")
            st.bar_chart(cat_counts)
            st.dataframe(df_news, use_container_width=True)

            csv_out = df_news.to_csv().encode("utf-8")
            st.download_button("⬇️ Download Results", csv_out, "news_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 GROWTH MATRIX")
    st.markdown("**NLP Suite v1.0**")
    st.markdown("---")

    st.markdown("#### 🧠 Loaded Models")
    for name, icon in [
        ("Spam Classifier",        "🛡️"),
        ("Language Detector",      "🌐"),
        ("Food Sentiment Analyser","🍽️"),
        ("News Categoriser",       "📰"),
    ]:
        st.markdown(f"{icon} {name} &nbsp; `✓ ready`", unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("🧑‍🤝‍🧑 About Us"):
        st.markdown("""
        We are a group of students exploring the exciting world of **Natural Language Processing**.
        
        This suite brings together four ML models for real-world text classification tasks.
        """)

    with st.expander("📞 Contact Us"):
        st.markdown("""
        📱 **Phone:** +91- 7266091264
        📧 **Email:** kaithwasyaman@gmail.com  
        🕘 Available 24*7
        """)

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit & scikit-learn")