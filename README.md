# 📈 Financial Sentiment Analyzer with FinBERT + LoRA

🔗 Live App:  
👉 https://finbert-lora-financial-sentiment-mhz4zxdyk75ljpytuxnnmm.streamlit.app/

---

## 🚀 Project Overview

This project is a **production-ready web application** that performs **financial sentiment analysis** using a **BERT-based model fine-tuned with LoRA**.

It takes financial news text as input and classifies it as either:
- **POSITIVE** 📈
- **NEGATIVE** 📉

The application is built with **Streamlit** and the model is hosted via **Hugging Face Model Hub**.

---

## 🧠 Why This Project Matters

Sentiment analysis for financial text is challenging due to:
- Domain-specific language
- Highly nuanced sentiment signals
- Risk-sensitive decisioning

Standard sentiment models often fail on financial text. By fine-tuning **FinBERT** specifically on financial headlines using **LoRA (Low-Rank Adaptation)**, we achieve:
- Efficient training
- Fast inference
- High-quality sentiment predictions

---

## 🛠 Key Components

### 📌 Model
- **Base Model**: `ProsusAI/finbert`
- **Fine-tuning Method**: LoRA
- **Task**: Binary Sentiment Classification  
  - 0 → NEGATIVE  
  - 1 → POSITIVE

### 💻 Frontend
- **Streamlit** web app UI
- Easy text input
- Finance-safe threshold slider
- Probability/confidence display

### 📦 Deployment
- Model hosted on **Hugging Face**
- Web app deployed on **Streamlit Cloud**

---

## 🧪 Model Training Summary

1. **Dataset**: Indian financial news headlines with labels  
   - Positive  
   - Negative

2. **Fine-Tuning with LoRA**  
   - LoRA adapters added to FinBERT attention layers  
   - Only ~few % of parameters trained
   - Efficient GPU use

3. **Evaluation Metrics**
   - Accuracy ~86%
   - F1-score ~85%
   - ROC-AUC ~94%

4. **Threshold Tuning**
   - Default threshold: 0.5
   - Finance-safe threshold: ~0.60  
   (to reduce false-positive bullish signals)

---

## 💡 How It Works (Inference Logic)

1. User enters financial text in the web app
2. Text is tokenized using Hugging Face tokenizer
3. FinBERT + LoRA model returns logits
4. Softmax applied to convert to probabilities
5. Custom threshold is applied:
   - If `positive_prob >= threshold`, label is **POSITIVE**
   - Else label is **NEGATIVE**

---

## 🧩 Example Usage (Python)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

model_name = "Hemanth7774/finbert-lora-financial-sentiment-merged"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

inputs = tokenizer("Markets rallied after the policy announcement", return_tensors="pt")
outputs = model(**inputs)
probs = F.softmax(outputs.logits, dim=1)

print(f"Negative: {probs[0][0]:.3f}, Positive: {probs[0][1]:.3f}")
