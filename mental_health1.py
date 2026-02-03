import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from PIL import Image
import os

# Set base path to images
base_path = "C:/Users/PRATHIKSHA/Downloads/Social Media Posts"

# Page setup
st.set_page_config(page_title="MindScope", layout="wide")

# Load fine-tuned BERT model and tokenizer
model_path = "MentalBERT"  # Replace with actual path to your saved model folder
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Label mapping
label_map = {
    0: "ADHD", 1: "Anxiety", 2: "Autism", 3: "Bipolar", 4: "Depression",
    5: "Eating Disorder", 6: "Mental Health (General)", 7: "OCD", 8: "PTSD", 9: "Schizophrenia"
}

# Load logo and main image
logo = Image.open(os.path.join(base_path, "logo.png"))
main_img = Image.open(os.path.join(base_path, "5.png"))

# Custom styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Great+Vibes&family=Playfair+Display&display=swap');

    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        animation: bg 20s ease infinite;
    }

    @keyframes bg {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .main-title {
        font-family: 'Great Vibes', cursive;
        font-size: 48px;
        font-weight: 400;
        margin-bottom: 10px;
    }

    .intro-text {
        font-size: 16px;
        line-height: 1.5;
        margin-top: 10px;
        color: #333;
    }

    .section-title {
        font-size: 24px;
        font-weight: bold;
        margin-top: 30px;
        color: #222;
    }

    .quote-text {
        font-family: 'Playfair Display', serif;
        font-size: 24px;
        line-height: 1.5;
        text-align: center;
        margin-top: 40px;
    }

    .quote-text em {
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Layout: 3 columns
left, center, right = st.columns([2, 2.5, 3])

# LEFT COLUMN
with left:
    st.image(logo, width=90)
    st.markdown("<div class='main-title'>MindScope</div>", unsafe_allow_html=True)
    st.markdown("<div class='intro-text'>Welcome to your mental health assistant.</div>", unsafe_allow_html=True)
    st.markdown("<div class='intro-text'>Let us know about your thoughts and feelings, and we will help you assess your emotional well-being.</div>", unsafe_allow_html=True)

# CENTER COLUMN
with center:
    st.markdown("<div class='section-title'>Talk to us</div>", unsafe_allow_html=True)
    st.write("We're here to listen. Tell us how you're feeling today...")
    user_input = st.text_area("", placeholder="I feel like everything is too much lately...")

    if st.button("💬 Analyze"):
        if user_input:
            with st.spinner("Analyzing your message..."):
                # Tokenize and predict
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
                prediction = label_map[predicted_class_id]
                st.success(f"Based on your input, you may be experiencing: **{prediction}**")
        else:
            st.warning("Please share a little about how you're feeling.")

# RIGHT COLUMN
with right:
    st.markdown(
        """
        <div class='quote-text'>
            Your <em>happiness</em> is essential. Your <em>self-care</em> is a necessity.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image(main_img, use_container_width=True)
