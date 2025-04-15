import streamlit as st
import numpy as np
import pickle

# Load the trained logistic regression model
model = pickle.load(open('titanic_log_model.pkl', 'rb'))

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival probability:")

Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Sex_male = st.selectbox("Sex", ['Male', 'Female'])
Age = st.slider("Age", 0, 80, 25)
SibSp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
Fare = st.number_input("Fare", 0.0, 600.0, 30.0)
Embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

Sex_male = 1 if Sex_male == 'Male' else 0
Embarked_S = 1 if Embarked == 'S' else 0
Embarked_Q = 1 if Embarked == 'Q' else 0

features = np.array([[Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q, Embarked_S]])
prediction = model.predict(features)[0]
proba = model.predict_proba(features)[0][1]

if st.button("Predict"):
    st.write(f"🧾 **Prediction:** {'Survived' if prediction == 1 else 'Did Not Survive'}")
    st.write(f"📊 **Survival Probability:** {proba * 100:.2f}%")
