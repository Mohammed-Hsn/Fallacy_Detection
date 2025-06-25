# app.py
import streamlit as st
from Llama import ask_llama
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import time
import pandas as pd


bert_model = TFBertForSequenceClassification.from_pretrained("./my_bert_model")
bert_tokenizer = BertTokenizer.from_pretrained("./my_bert_model")

def predict_with_bert(text):
    encodings = bert_tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="tf"
    )
    outputs = bert_model(encodings)
    logits = outputs.logits
    prediction = tf.argmax(logits, axis=1).numpy()[0]
    return "Fallacy" if prediction == 1 else "Not a Fallacy"


if "history" not in st.session_state:
    st.session_state.history = []


st.set_page_config(page_title="Fallacy Detector", layout="centered")
st.title("Fallacy Detector")



text = st.text_area("Enter Argument:")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter a valid argument.")
    else:
        
        progress_bar = st.progress(0)
        progress_bar.progress(10)
        time.sleep(0.1)

        
        type_prompt = f"Is this argument related to education or climate? Just answer Yes or No:\n{text}"
        type_reply = ask_llama(type_prompt).strip().lower()
        progress_bar.progress(40)
        time.sleep(0.1)


        if "yes" in type_reply:
            predicted_label = predict_with_bert(text)
            progress_bar.progress(70)
            time.sleep(0.1)

            explain_prompt = f"Give a short and concise reason (1-2 sentences only), no extra words, why this argument is classified as '{predicted_label}':\n{text}"
            explanation = ask_llama(explain_prompt).strip()

        else:
            classify_prompt = f"""
You are a fallacy classifier.

If the following sentence is an argumentative statement and contains a fallacy, reply with exactly "Fallacy".

If it is an argumentative statement and contains no fallacy, reply with exactly "Not a Fallacy".

If it is a factual statement or observation (not an argument), reply with exactly "Not a Fallacy".

ONLY reply with exactly one word: "Fallacy" or "Not a Fallacy".

Argument:  
{text}
            """.strip()
            predicted_label = ask_llama(classify_prompt).strip()
            progress_bar.progress(70)
            time.sleep(0.1)

            explain_prompt = f"Give a short and concise reason (1-2 sentences only), no extra words, why this argument is classified as {predicted_label}:\n{text}"
            explanation = ask_llama(explain_prompt).strip()

        
        if predicted_label == "Fallacy":
            highlight_prompt = f"""
Highlight only the exact part of this sentence that contains the fallacy. 

Reply only with the exact words or phrase. No commentary.

Sentence: {text}
            """.strip()
            highlighted_part = ask_llama(highlight_prompt).strip()
        else:
            highlighted_part = "None"

        progress_bar.progress(100)
        time.sleep(0.1)

     
        st.success(f"Predicted: {predicted_label}")
        st.info(f"Reason: {explanation}")
        st.info(f"Highlighted Part: {highlighted_part}")

   
        st.session_state.history.append({
            "Argument": text,
            "Predicted": predicted_label,
            "Reason": explanation,
            "Highlight": highlighted_part
        })

        progress_bar.empty()

# Static sidebar header - moved outside the loop
st.sidebar.header("📜 History")

# History items loop
for idx, item in enumerate(st.session_state.history[::-1]):
    st.sidebar.write(f"**Argument:** {item['Argument']}")
    st.sidebar.write(f"**Predicted:** {item['Predicted']}")
    st.sidebar.write(f"**Reason:** {item['Reason']}")
    st.sidebar.write(f"**Highlight:** {item['Highlight']}")

    if item['Predicted'] == "Fallacy":
        rewrite_button = st.sidebar.button(f"✏️ Rewrite (#{idx + 1})", key=f"rewrite_{idx}")
        if rewrite_button:
            rewrite_prompt = f"""
Rewrite this sentence to make it logically sound and remove the fallacy.

Reply ONLY with the rewritten sentence. Do not add any explanation.

Original: {item['Argument']}
            """.strip()

            rewritten_sentence = ask_llama(rewrite_prompt).strip()
            st.sidebar.success(f"{rewritten_sentence}")

    st.sidebar.write("---")


if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name='fallacy_detection_results.csv',
        mime='text/csv'
    )