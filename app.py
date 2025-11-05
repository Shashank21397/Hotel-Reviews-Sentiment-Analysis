import streamlit as st
import pandas as pd
import base64
import plotly.graph_objects as go
from transformers import pipeline
import time

# ===============================
# 🎥 Background Video (Professional Dark Theme)
# ===============================
def add_bg_video(video_file):
    """Adds a background video with a refined, professional dark overlay."""
    try:
        with open(video_file, "rb") as f:
            video_bytes = f.read()
            base64_video = base64.b64encode(video_bytes).decode("utf-8")
    except FileNotFoundError:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #121212;
                color: #E0E0E0;
            }
            </style>
            """, unsafe_allow_html=True)
        return

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: transparent !important;
            overflow: auto; 
        }}
        .video-bg {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            z-index: -2;
        }}
        .overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(2px);
            z-index: -1;
        }}
        h1.main-title {{
            text-align: center;
            color: #FFB300; 
            font-weight: 800;
            font-size: 3.4rem;
            margin-top: 35px;
            text-shadow: 0 0 12px rgba(255, 179, 0, 0.4);
        }}
        p.subtitle {{
            text-align: center;
            color: #A9A9A9;
            font-size: 1.25rem;
            margin-bottom: 45px;
        }}
        label, p, h2, h3, div[data-testid="stMarkdownContainer"] {{
             color: #E0E0E0 !important;
        }}
        .content-card-wrapper {{
            background: rgba(25, 25, 25, 0.85); 
            border: 1px solid rgba(255, 179, 0, 0.3); 
            border-radius: 15px; 
            padding: 30px 40px;
            backdrop-filter: blur(8px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            margin-bottom: 50px;
        }}
        div.stButton > button:first-child {{
            background: linear-gradient(90deg, #FFB300, #FF8F00); 
            color: #000;
            font-weight: 700;
            border-radius: 10px;
            border: none;
            padding: 0.8em 3em; 
            font-size: 1.15em;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        div.stButton > button:hover {{
            background: linear-gradient(90deg, #FFCA28, #FFB300);
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(255, 179, 0, 0.4);
        }}
        [data-testid="stMetricValue"] {{
            font-size: 2.8rem; 
            font-weight: 800;
            color: #1E88E5; 
            text-shadow: 0 0 5px rgba(30, 136, 229, 0.3);
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 1.05rem;
            font-weight: 600;
            color: #B0BEC5; 
            margin-bottom: 5px;
        }}
        [data-testid="stMetricDelta"] {{
            color: #FFB300 !important; 
        }}
        .sentiment-card {{
            background: rgba(40, 40, 40, 0.7); 
            border-radius: 10px;
            padding: 15px 20px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            transition: border 0.3s;
        }}
        .positive {{ border-left: 6px solid #4CAF50; }} 
        .neutral {{ border-left: 6px solid #FFC107; }} 
        .negative {{ border-left: 6px solid #F44336; }} 
        .sentiment-label {{ 
            font-weight: bold; 
            margin-bottom: 4px;
            font-size: 1.1rem;
            color: #FFFFFF;
        }}
        .review-text {{ 
            color: #C0C0C0; 
            font-size: 0.95rem;
        }}
        .results-header {{
            text-align:center; 
            color:#FFB300; 
            font-weight: 700;
            font-size: 2.2rem;
            margin-top: 10px;
            margin-bottom: 25px;
        }}
        </style>
        <video class="video-bg" autoplay muted loop playsinline>
            <source src="data:video/mp4;base64,{base64_video}" type="video/mp4">
        </video>
        <div class="overlay"></div>
        """, unsafe_allow_html=True
    )

# ===============================
# ⚙️ Load Model
# ===============================
@st.cache_resource
def load_model():
    """Loads the sentiment analysis model."""
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

model = load_model()

if 'df_result' not in st.session_state:
    st.session_state['df_result'] = pd.DataFrame()

# ===============================
# 🎬 Layout
# ===============================
st.set_page_config(page_title="Hotel Sentiment Dashboard", page_icon="🏨", layout="wide")
add_bg_video("galaxy.mp4")

st.markdown("<h1 class='main-title'> Hotel Reviews Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyze customer feedback for immediate operational insights and quality control.</p>", unsafe_allow_html=True)

# ===============================
# 📤 Input Section
# ===============================
st.markdown('<div class="content-card-wrapper">', unsafe_allow_html=True)
with st.container():
    st.subheader("Customer Feedback Data Input")

    tab_csv, tab_manual = st.tabs(["📂 Upload Review Data", "✍️ Manual Entry"])
    reviews = []

    with tab_csv:
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                review_col = next((col for col in df.columns if "review" in col.lower()), None)
                if review_col:
                    reviews = df[review_col].dropna().astype(str).tolist()
                    st.success(f"✅ Loaded {len(reviews)} reviews from {review_col}")
                else:
                    st.error("No 'review' column found.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    with tab_manual:
        text_input = st.text_area("Enter reviews (one per line):", height=180)
        if text_input.strip():
            reviews = [r.strip() for r in text_input.split("\n") if r.strip()]
            st.info(f"{len(reviews)} reviews ready for analysis.")

    st.markdown("---")

    if st.button("🚀 Analyze Reviews & Generate Report"):
        if not reviews:
            st.warning("Please upload or enter reviews.")
            st.stop()

        st.markdown("<h2 class='results-header'>Sentiment Analysis Report</h2>", unsafe_allow_html=True)
        st.info(f"⏳ Analyzing {len(reviews)} reviews...")

        progress = st.progress(0)
        results = []
        batch_size = 100
        start = time.time()
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i+batch_size]
            batch_results = model(batch)
            results.extend(batch_results)
            progress.progress(min((i+batch_size)/len(reviews), 1.0))
        progress.empty()

        sentiments = [r["label"].capitalize() for r in results]
        df_result = pd.DataFrame({"Review": reviews, "Sentiment": sentiments})
        st.session_state['df_result'] = df_result
        st.success(f"✅ Done! Time: {time.time()-start:.2f}s")

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# 📊 Display Section
# ===============================
if not st.session_state['df_result'].empty:
    df_result = st.session_state['df_result']
    total = len(df_result)

    st.markdown('<div class="content-card-wrapper">', unsafe_allow_html=True)
    with st.container():
        # 📈 Sentiment Performance Indicators (Fixed)
        st.markdown("### 📈 Sentiment Performance Indicators", unsafe_allow_html=True)
        st.markdown("---")

        counts = df_result["Sentiment"].value_counts().to_dict()
        pos_percent = counts.get("Positive", 0) / total * 100
        neu_percent = counts.get("Neutral", 0) / total * 100
        neg_percent = counts.get("Negative", 0) / total * 100

        col_total, col_pos, col_neu, col_neg = st.columns([1.2, 1, 1, 1])
        col_total.metric("TOTAL REVIEWS ANALYZED", f"{total:,}")
        col_pos.metric("POSITIVE SHARE", f"{pos_percent:.1f}%", f"{counts.get('Positive', 0)} Reviews")
        col_neu.metric("NEUTRAL/UNCERTAIN", f"{neu_percent:.1f}%", f"{counts.get('Neutral', 0)} Reviews", delta_color="off")
        col_neg.metric("CRITICAL NEGATIVE RISK", f"{neg_percent:.1f}%", f"{counts.get('Negative', 0)} Reviews", delta_color="inverse")

        st.markdown("---")

        # 📊 Pie Chart + Download Section
        col_chart, col_download = st.columns([3, 2])
        with col_chart:
            sentiment_counts = df_result["Sentiment"].value_counts()
            colors = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}

            fig = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.55,
                marker=dict(colors=[colors.get(s, "#888") for s in sentiment_counts.index]),
                textinfo="label+percent",
                textfont=dict(color="white", size=14),
                hoverlabel=dict(bgcolor="#303030", font_size=13, font_color="white")
            )])
            fig.update_layout(
                title=dict(text="Distribution of Sentiment Classes", font=dict(color="#E0E0E0", size=18), x=0.5),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, 
                margin=dict(t=50, b=10, l=40, r=40),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_download:
            @st.cache_data
            def convert_df(df): return df.to_csv(index=False).encode('utf-8')
            csv = convert_df(df_result)
            st.download_button("⬇️ Download Full Sentiment Data (CSV)", csv, "hotel_sentiment_report.csv", "text/csv")

        # 🔍 Filtered Sentiment Tabs
        st.markdown("---")
        st.markdown("### 🔍 Explore Reviews by Sentiment Category", unsafe_allow_html=True)
        tab_pos, tab_neu, tab_neg = st.tabs(["✅ Positive Reviews", "⚠️ Neutral Reviews", "❌ Negative Reviews"])

        with tab_pos:
            pos_df = df_result[df_result["Sentiment"] == "Positive"]
            if not pos_df.empty:
                for review in pos_df["Review"]:
                    st.markdown(f"<div class='sentiment-card positive'><div class='sentiment-label'>✅ Positive</div><div class='review-text'>{review}</div></div>", unsafe_allow_html=True)
            else:
                st.info("No positive reviews found.")

        with tab_neu:
            neu_df = df_result[df_result["Sentiment"] == "Neutral"]
            if not neu_df.empty:
                for review in neu_df["Review"]:
                    st.markdown(f"<div class='sentiment-card neutral'><div class='sentiment-label'>⚠️ Neutral</div><div class='review-text'>{review}</div></div>", unsafe_allow_html=True)
            else:
                st.info("No neutral reviews found.")

        with tab_neg:
            neg_df = df_result[df_result["Sentiment"] == "Negative"]
            if not neg_df.empty:
                for review in neg_df["Review"]:
                    st.markdown(f"<div class='sentiment-card negative'><div class='sentiment-label'>❌ Negative</div><div class='review-text'>{review}</div></div>", unsafe_allow_html=True)
            else:
                st.info("No negative reviews found.")
    st.markdown('</div>', unsafe_allow_html=True)
