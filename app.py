import streamlit as st
import numpy as np
import pickle


with open("disease_model.sav", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.sav", "rb") as f:
    le = pickle.load(f)

with open("symptom_columns.sav", "rb") as f:
    symptoms = pickle.load(f)


st.set_page_config(page_title="Disease Predictor", layout="wide")
st.title("Symptom-Based Disease Predictor")
st.markdown("Select symptoms to predict the most likely disease.(Having more symptoms will increase the accuracy of my Model)")


selected_symptoms = st.multiselect("Pick your symptoms", symptoms)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        
        input_vector = np.zeros((1, len(symptoms)))
        for symptom in selected_symptoms:
            if symptom in symptoms:
                idx = symptoms.index(symptom)
                input_vector[0, idx] = 1

        
        y_proba = model.predict_proba(input_vector)
        class_index = np.argmax(y_proba)
        predicted_disease = le.inverse_transform([class_index])[0]

    
        top3 = np.argsort(y_proba[0])[::-1][:3]
        st.success(f"✅ Predicted Disease: **{predicted_disease}**")

        st.subheader("🔍 Top 3 Likely Diseases")
        for i in top3:
            name = le.inverse_transform([i])[0]
            prob = y_proba[0][i]
            st.write(f"• {name} — {prob:.2%}")
