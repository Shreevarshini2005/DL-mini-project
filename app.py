import streamlit as st
from predict import predict_personality

st.title("DeepAura MBTI Predictor")

user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        prediction = predict_personality(user_input)
        st.success(f"Predicted MBTI: {prediction}")
