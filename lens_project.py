import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px  # Using Plotly for better interactivity

# --- UI Setup & Custom CSS ---
st.set_page_config(page_title="LENSX | Advanced NLP Suite", layout="wide", page_icon="🔍")

# Injecting Custom CSS for a modern "Dark/Neon" aesthetic
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #00d4ff;
        color: black;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #008fb3;
        color: white;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00d4ff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load models (with caching for speed) ---
@st.cache_resource
def load_models():
    # Replace these with your actual paths
    try:
        s_m = joblib.load("spam_classifier.pkl")
        l_m = joblib.load("lang_det.pkl")
        n_m = joblib.load("news_cat.pkl")
        r_m = joblib.load("review.pkl")
        return s_m, l_m, n_m, r_m
    except:
        st.error("Model files not found. Please ensure .pkl files are in the directory.")
        return None, None, None, None

spam_model, language_model, news_model, review_model = load_models()

# --- Sidebar ---
with st.sidebar:
    st.title("🔍 LENSX")
    st.image("https://via.placeholder.com/150", caption="NLP Intelligence Unit") # Replace with flag.jpg
    st.markdown("---")
    with st.expander("🧑‍🤝‍🧑 Our Mission"):
        st.info("LENSX is a high-performance NLP suite designed by students to decode human language through machine learning.")
    with st.expander("📞 Tech Support"):
        st.write("📧 support@lensx.ai")
        st.write("📱 +1 (800) LENSX-NLP")
    st.markdown("---")
    st.caption("v2.0.1 - Powered by Scikit-Learn")

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Spam Shield", 
    "🌐 Polyglot AI", 
    "🍽️ Sentiment Pro", 
    "📰 News Radar"
])

# --- Tab 1: Spam Classifier ---
with tab1:
    st.header("🎯 Spam Shield")
    col_input, col_viz = st.columns([1, 1])

    with col_input:
        st.subheader("Manual Inspection")
        msg = st.text_area("Analyze a message", placeholder="Type here...", key="spam_input")
        if st.button("Run Diagnostic", key="btn_spam"):
            if msg:
                pred = spam_model.predict([msg])
                if pred[0] == 0:
                    st.error("🚨 SPAM DETECTED")
                else:
                    st.success("✅ MESSAGE CLEAR")
            else:
                st.warning("Input required.")

    with col_viz:
        st.subheader("Bulk Analytics")
        uploaded_file = st.file_uploader("Upload Batch (CSV/TXT)", type=["csv", "txt"])
        
    if uploaded_file:
        df_spam = pd.read_csv(uploaded_file, header=None, names=['Msg'])
        df_spam["Prediction"] = spam_model.predict(df_spam.Msg)
        df_spam["Prediction"] = df_spam["Prediction"].map({0: 'Spam', 1: 'Safe'})
        
        # Dashboard Metrics
        m1, m2 = st.columns(2)
        m1.metric("Total Analyzed", len(df_spam))
        m2.metric("Spam Found", len(df_spam[df_spam["Prediction"] == "Spam"]))

        # Plotly Interactivity
        fig = px.pie(df_spam, names='Prediction', hole=.4, 
                     color_discrete_sequence=['#ff4b4b', '#00d4ff'])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_spam.head(10), use_container_width=True)

# --- Tab 2: Language Detection ---
with tab2:
    st.header("🌐 Polyglot AI")
    st.markdown("Instantly identify over 20+ languages using vector analysis.")
    lang_text = st.text_area("Paste text here", height=200, placeholder="Bonjour, how can I help you?")
    
    if st.button("Identify Language"):
        if lang_text:
            lang_pred = language_model.predict([lang_text])
            st.balloons()
            st.markdown(f"""
            <div style="background-color:#1e2130; padding:20px; border-radius:10px; text-align:center;">
                <h2 style="color:#00d4ff;">Detected Language: {lang_pred[0]}</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text.")

# --- Tab 3: Food Review Sentiment ---
with tab3:
    st.header("🍽️ Sentiment Pro")
    review_text = st.text_area("Customer Review", placeholder="The pasta was divine but the service was slow...")
    
    if st.button("Analyze Sentiment"):
        if review_text:
            rev_pred = review_model.predict([review_text])
            score = "Positive" if rev_pred[0] == 1 else "Negative"
            
            if score == "Positive":
                st.write(f"### Score: 🟢 {score}")
                st.progress(90) # Decorative progress bar
            else:
                st.write(f"### Score: 🔴 {score}")
                st.progress(20)
        else:
            st.warning("Review text is missing.")

# --- Tab 4: News Classification ---
with tab4:
    st.header("📰 News Radar")
    news_text = st.text_input("Enter Headline")
    
    if st.button("Categorize"):
        if news_text:
            news_pred = news_model.predict([news_text])
            st.info(f"The content belongs to the **{news_pred[0].upper()}** category.")
            
            # Interactive visual: A simple probability bar (mockup)
            st.caption("Confidence Level")
            st.progress(85)
        else:
            st.warning("Please enter news text.")