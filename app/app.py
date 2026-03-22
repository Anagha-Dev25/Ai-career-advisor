import streamlit as st
import pickle
import shap
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- LOAD MODEL ----------------------
model = pickle.load(open("../model/career_model.pkl", "rb"))
le = pickle.load(open("../model/label_encoder.pkl", "rb"))

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="AI Career Advisor", page_icon="🚀", layout="centered")

st.title("🚀 AI Career Advisor")
st.markdown("### Find your path. Backed by data. Explained by AI.")
st.divider()

# ---------------------- INPUT SECTION ----------------------
st.subheader("🧾 Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    python_skill = st.slider("Python Skill", 0, 10)
    math_skill = st.slider("Math Skill", 0, 10)
    communication = st.slider("Communication", 0, 10)
    creativity = st.slider("Creativity", 0, 10)

with col2:
    business_knowledge = st.slider("Business Knowledge", 0, 10)
    interest_ai = st.slider("Interest in AI", 0, 10)
    interest_design = st.slider("Interest in Design", 0, 10)
    cgpa = st.slider("CGPA", 0.0, 10.0)

st.divider()

# ---------------------- PREDICTION ----------------------
if st.button("✨ Predict My Career"):

    # Create input dataframe
    input_data = pd.DataFrame([[python_skill, math_skill, communication, creativity,
                                business_knowledge, interest_ai, interest_design, cgpa]],
                              columns=["python_skill", "math_skill", "communication", "creativity",
                                       "business_knowledge", "interest_ai", "interest_design", "cgpa"])

    # Prediction
    prediction = model.predict(input_data)
    career = le.inverse_transform(prediction)

    st.success(f"🎯 Recommended Career: **{career[0]}**")

    # ---------------------- CONFIDENCE ----------------------
    probs = model.predict_proba(input_data)[0]
    confidence = max(probs)

    st.info(f"🧠 Confidence Score: **{confidence:.2f}**")

    # ---------------------- TOP MATCHES ----------------------
    top_2 = probs.argsort()[-2:][::-1]

    st.subheader("📊 Top Career Matches")
    for i in top_2:
        st.markdown(f"**{le.classes_[i]}** → `{probs[i]:.2f}`")

    st.divider()
    st.subheader("🧠 Interpretation")
    
    top1 = probs[top_2[0]]
    top2_prob = probs[top_2[1]]
    
    # Strong confidence case
    if top1 > 0.75:
        st.success("🔥 Strong match! Your profile clearly aligns with this career path.")
    
    # Moderate confidence
    elif top1 > 0.5:
        st.info("👍 Good match. Your skills are well suited, but there are other possible paths too.")
    
    # Close competition case
    if abs(top1 - top2_prob) < 0.15:
        st.warning(f"⚖️ Your profile also closely matches **{le.classes_[top_2[1]]}**. You may explore both options.")
    
    # Low confidence case
    if top1 < 0.5:
        st.error("🤔 Your profile is quite diverse. Consider exploring multiple career paths or refining your focus.")

    # ---------------------- SHAP EXPLANATION ----------------------
    explainer = shap.TreeExplainer(model)

    # Use new API
    shap_values = explainer(input_data)
    
    st.subheader("🔍 Why this recommendation?")
    st.caption("This shows how each feature influenced the prediction")
    
    # Always use class index 0 (safe fallback)
    try:
        pred_class = prediction[0]
        shap.plots.waterfall(shap_values[0, :, pred_class])
    except:
        # fallback if SHAP gives single output
        shap.plots.waterfall(shap_values[0])
    
    fig = plt.gcf()
    st.pyplot(fig)
    plt.clf()
    st.caption("Built with ❤️ using Machine Learning & Explainable AI")
