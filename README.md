# 🏨 Hotel Reviews Sentiment Analysis  

An **AI-powered Hotel Reviews Sentiment Analysis Dashboard** built using **Streamlit** and **RoBERTa Transformer** (a deep learning model for Natural Language Processing).  
This project analyzes hotel customer reviews and classifies them as **Positive**, **Neutral**, or **Negative**, helping hotel managers quickly understand guest satisfaction trends.

---

## 🌟 Key Features  

- 🧠 **AI-Powered NLP Model** — Uses **RoBERTa**, a state-of-the-art Transformer model for contextual sentiment analysis.  
- 📊 **Interactive Dashboard** — Streamlit-based UI with real-time sentiment metrics and visualizations.  
- 📁 **Supports Text & CSV Uploads** — Analyze single or bulk reviews.  
- ⚙️ **Smart Neutral Detection** — Balanced sentiment for reviews like “Service was acceptable but not impressive.”  
- 💾 **Downloadable Results** — Export analyzed data as CSV.  
- 🎥 **Modern UI** — Gold-dark themed dashboard with an animated galaxy video background.  

---

## 🧠 How It Works  

This app uses **Natural Language Processing (NLP)** powered by a **Transformer-based AI model** from Hugging Face:  
[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

| Technology | Description |
|-------------|-------------|
| **AI / Deep Learning** | Transformer (RoBERTa) model trained on millions of human-written texts |
| **ML Concepts Used** | Text preprocessing, vectorization, contextual classification |
| **Frontend Framework** | Streamlit |
| **Visualization** | Plotly |

In simple terms — your model doesn’t just look for keywords; it **understands the meaning and emotion behind the words**, like a human.

---

## ⚙️ Tech Stack  

| Component | Technology |
|------------|-------------|
| Frontend | Streamlit (Python) |
| NLP Model | RoBERTa Transformer |
| Visualization | Plotly |
| Data Handling | Pandas |
| Model Management | Joblib |
| Environment | Jupyter Notebook + Python 3.9+ |

---

## 🧩 Folder Structure  

```

Hotel-Reviews-Sentiment-Analysis/
│
├── app.py                    # Streamlit app file
├── artifacts/                # Folder containing trained models
│   ├── sentiment_model.joblib
│   └── vectorizer.joblib
├── galaxy.mp4                # Background animation (optional)
├── HotelSentiment.ipynb      # Jupyter notebook (training + EDA)
├── requirements.txt          # All dependencies
└── README.md                 # Project documentation

````

---

## 🚀 Run Locally  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Shashank21397/Hotel-Reviews-Sentiment-Analysis.git
cd Hotel-Reviews-Sentiment-Analysis
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
streamlit run app.py
```

Then open the local URL in your browser (usually `http://localhost:8501`).

---

## 🧾 Example Reviews

| Review                                           | Sentiment   |
| ------------------------------------------------ | ----------- |
| The staff were extremely friendly and helpful.   | 👍 Positive |
| Service was acceptable but not impressive.       | 😐 Neutral  |
| The room was dirty and the AC was broken.        | 👎 Negative |
| Great location but the food was average.         | 😐 Neutral  |
| Absolutely loved the ambience and quick service! | 👍 Positive |

---

## 📈 Dashboard Preview

The app displays:

* Sentiment percentages (Positive, Neutral, Negative)
* Distribution chart (Pie or Bar)
* Individual review cards
* Downloadable sentiment report

**Theme:**
✨ Gold & dark UI with a galaxy background and smooth card animations

---

## 🔒 Model Details

Pretrained Transformer model:

* **Name:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
* **Framework:** Hugging Face Transformers
* **Type:** Contextual Sentiment Classifier (3-label: positive / neutral / negative)

Models are optionally stored locally under `/artifacts`:

```
artifacts/
├── sentiment_model.joblib
└── vectorizer.joblib
```

---

## 👨‍💻 Author

**👤 Shashank A U**
📍 Project: *AI-Powered Hotel Reviews Sentiment Analysis *
🔗 GitHub: [Shashank21397](https://github.com/Shashank21397)
Mail : shashank21396@gmail.com

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---


```

---

```

---





