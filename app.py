import streamlit as st
import pandas as pd
import joblib

model = joblib.load('football/injury.pkl')
feature_names = joblib.load('football/columns.pkl')

if 'Injury_Next_Season' in feature_names:
    feature_names.remove('Injury_Next_Season')

position_map = {'Goalkeeper': 0, 'Defender': 1, 'Midfielder': 2, 'Forward': 3}

st.title("⚽Football Player Next Season Injury Risk Predictor")
st.write("Enter the player's details to predict injury risk for the upcoming season.")

inputs = {}
for feature in feature_names:
    if feature == 'Position':
        inputs[feature] = st.selectbox("Position", list(position_map.keys()))
    elif feature in ['Warmup_Routine_Adherence']:
        inputs[feature] = st.selectbox("Warmup Routine Adherence", [0, 1])
    elif feature in ['Age']: 
        inputs[feature] = st.slider("Age", min_value=10,max_value=50)
            
    else:
        inputs[feature] = st.number_input(f"{feature}", min_value=0.0)

inputs['Position'] = position_map[inputs['Position']]

input_df = pd.DataFrame([inputs])

if st.button("Predict Injury Risk"):
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"Player is at risk of injury next season")
    else:
        st.success(f"Player is not at risk of injury")
