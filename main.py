import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- PAGE ----------------
st.set_page_config(page_title="AI Spam Shield Pro", page_icon="🛡️", layout="centered")

st.title("🛡️ AI Spam Shield Pro")
st.caption("ML + NLP + Explainable AI + Visualization Dashboard")

# ---------------- DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

df = load_data()

# ---------------- MODEL ----------------
@st.cache_resource
def train_model(data):
    X = data['message']
    y = data['label']

    vectorizer = CountVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)

    return model, vectorizer

model, vectorizer = train_model(df)

# ---------------- KEYWORDS ----------------
spam_keywords = [
    "free", "win", "winner", "cash", "prize",
    "click", "urgent", "offer", "limited", "buy now",
    "congratulations", "selected", "claim", "password"
]

# ---------------- FEATURES ----------------
def extract_features(text):
    return {
        "length": len(text),
        "digits": len(re.findall(r"\d", text)),
        "links": len(re.findall(r"http|www", text.lower())),
        "caps": sum(1 for c in text if c.isupper()) / (len(text) + 1)
    }

def analyze_message(text):
    words = text.lower().split()
    return [w for w in words if w in spam_keywords]

# ---------------- INPUT ----------------
message = st.text_area("✉ Enter Message:", height=150)

if st.button("Analyze AI 🚀"):

    if message.strip() == "":
        st.warning("Please enter a message!")
    else:

        # ---------------- MODEL ----------------
        vec = vectorizer.transform([message])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        spam_prob = prob[1] * 100
        safe_prob = prob[0] * 100
        confidence = max(spam_prob, safe_prob)

        # ---------------- FEATURES ----------------
        features = extract_features(message)
        found_keywords = analyze_message(message)

        # ---------------- RESULT ----------------
        st.subheader("📊 AI Security Report")

        col1, col2 = st.columns(2)
        col1.metric("🟢 Safe", f"{safe_prob:.2f}%")
        col2.metric("🔴 Spam", f"{spam_prob:.2f}%")

        st.progress(int(confidence))

        # ---------------- RISK ----------------
        if spam_prob > 80:
            risk = "🔴 HIGH RISK SPAM"
        elif spam_prob > 50:
            risk = "🟠 MEDIUM RISK"
        else:
            risk = "🟢 LOW RISK"

        st.markdown(f"### {risk}")

        # ---------------- FINAL RESULT ----------------
        if pred == 1:
            st.error(f"🚨 SPAM DETECTED ({spam_prob:.2f}%)")
        else:
            st.success(f"✅ SAFE MESSAGE ({safe_prob:.2f}%)")

        # ---------------- EXPLANATION ----------------
        st.subheader("🧠 AI Explanation")

        if found_keywords:
            st.warning(f"⚠ Spam keywords found: {found_keywords}")
        else:
            st.info("No strong spam keywords detected.")

        st.write(f"📏 Length: {features['length']}")
        st.write(f"🔢 Digits: {features['digits']}")
        st.write(f"🔗 Links: {features['links']}")
        st.write(f"🔠 CAPS intensity: {features['caps']:.2f}")

        # ---------------- GRAPH 1: PROBABILITY ----------------
        st.subheader("📊 Probability Chart")

        fig1, ax1 = plt.subplots()
        ax1.bar(["Safe", "Spam"], [safe_prob, spam_prob])
        ax1.set_title("Spam vs Safe Probability")
        st.pyplot(fig1)

        # ---------------- GRAPH 2: FEATURES ----------------
        st.subheader("📊 Message Feature Analysis")

        fig2, ax2 = plt.subplots()
        ax2.bar(
            ["Length", "Digits", "Links", "Caps"],
            [
                features["length"],
                features["digits"],
                features["links"],
                features["caps"] * 100
            ]
        )
        ax2.set_title("Message Behavior Features")
        st.pyplot(fig2)

        # ---------------- GRAPH 3: RISK PIE ----------------
        st.subheader("📊 Risk Distribution")

        fig3, ax3 = plt.subplots()
        ax3.pie(
            [spam_prob, safe_prob],
            labels=["Spam Risk", "Safe"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax3.set_title("Risk Breakdown")
        st.pyplot(fig3)

        # ---------------- SAFETY GUIDE ----------------
        st.subheader("📌 Safety Guide")

        if pred == 1:
            st.error("""
🚨 Do NOT click links  
🚨 Do NOT share personal info  
🚨 Block sender  
🚨 Report spam  
            """)
        else:
            st.success("""
✔ Message is safe  
✔ Normal communication  
✔ No action required  
            """)

# ---------------- DATA PREVIEW ----------------
with st.expander("📂 Dataset Preview"):
    st.dataframe(df.head())

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("🔥 AI Spam Shield Pro | ML + NLP + Explainable AI + Visualization Dashboard")