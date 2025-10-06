import streamlit as st
from transformers import pipeline

st.title("🧠 Sentiment-Based Text Generator")

# Load models safely and cache them
@st.cache_resource
def load_models():
    try:
        sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        generator_model = pipeline("text-generation", model="gpt2")
        return sentiment_model, generator_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

sentiment_pipe, generator_pipe = load_models()

# User input
text = st.text_area("✍️ Enter your text or topic:", height=150, placeholder="e.g., Climate change and its effects")

# Buttons
col1, col2 = st.columns(2)

with col1:
    detect_btn = st.button("🔍 Detect Sentiment")
with col2:
    generate_btn = st.button("✨ Generate Text")

# Detect sentiment
if detect_btn:
    if not text.strip():
        st.warning("Please enter some text first!")
    else:
        result = sentiment_pipe(text)[0]
        st.success(f"**Sentiment:** {result['label']} (confidence: {result['score']:.2f})")

# Choose sentiment for generation
st.markdown("### 🎯 Choose Sentiment for Text Generation")
sentiment_choice = st.radio("Select one:", ["positive", "negative", "neutral"])

# Generate text
if generate_btn:
    if not text.strip():
        st.warning("Please enter some text first!")
    else:
        prompt_prefix = {
            "positive": "Write something positive about: ",
            "negative": "Write something negative about: ",
            "neutral": "Write a neutral paragraph about: "
        }[sentiment_choice]

        prompt = prompt_prefix + text
        with st.spinner("Generating text... please wait ⏳"):
            result = generator_pipe(prompt, max_new_tokens=80, do_sample=True, temperature=0.8)
            output = result[0]["generated_text"]

        st.subheader("📝 Generated Text:")
        st.write(output)
