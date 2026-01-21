import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import re

# -- page config -- 
st.set_page_config(
    page_title="Sentiment Analysis with FinBERT",
    layout="centered",
)

st.markdown("""
<style>
/* Center ONLY the main title (FinBERT) */
h1 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# -- page title --
st.title("Financial Sentiment Analyzer with FinBERT")
st.caption("FinBERT + LoRA fine-tuned on Indian financial news")

# -- load model and tokenizer --
@st.cache_resource
def load_model():
    model_name = "Hemanth7774/finbert-lora-financial-sentiment-merged"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -- pre-processing function ---
def preprocess_text(text):
    """
    Minimal and BERT-safe text preprocessing for sentiment analysis.
    """
    # 1. Handle missing values
    if text is None:
        return ""
    # 2. Convert to string (safety)
    text = str(text)   
    # 3. text to lower case
    text = text.lower()
    # 4. Remove leading/trailing spaces
    text = text.strip()
    # 5. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # 6. Remove unwanted special characters
    # Keep: letters, numbers, punctuation, currency, %
    text = re.sub(r"[^a-zA-Z0-9.,!?₹% ]", "", text)
    # 7. Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)
    return text

# -- sentiment prediction function --
def predict_sentiment(text):
    text = preprocess_text(text)
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    negative_prob = probs[0]
    positive_prob = probs[1]
    
    label = "Positive 📈" if positive_prob >= 0.60 else "Negative 📉"
    
    return label, positive_prob, negative_prob

# -- user input --
user_input = st.text_area(
    "📰 Enter Financial News / Statement",
    height=150,
    placeholder="Example: RBI raises interest rates amid inflation concerns..."
)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text for analysis.")
    else:
        label, pos_prob, neg_prob = predict_sentiment(user_input)
        
        st.subheader("📌 Prediction")
        st.markdown(f"### **{label}**")
        
        st.subheader("📊 Probabilities")
        st.write(f"**Positive:** {pos_prob:.3f}")
        st.write(f"**Negative:** {neg_prob:.3f}")
        
        # Convert probabilities safely
        pos_prob = float(pos_prob)
        neg_prob = float(neg_prob)

        st.progress(pos_prob if label.startswith("POSITIVE") else neg_prob)
        
    
    # -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Model: FinBERT (ProsusAI) + LoRA fine-tuning | Built with Streamlit")