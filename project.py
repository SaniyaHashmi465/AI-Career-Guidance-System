import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset
df = pd.read_csv("students_skills.csv")

X = df.drop("role", axis=1)
y = df["role"]

# Training the model
model = RandomForestClassifier()
model.fit(X, y)

# Skill gap function
def skill_gap(role, skills):
    role_data = df[df["role"] == role].iloc[0]
    missing = []

    for skill in skills:
        if skills[skill] == 0 and role_data[skill] == 1:
            missing.append(skill)

    return missing

#  Streamlit UI  #

st.title("AI Career Guidance System")

st.write("Select your skills:")

python = st.checkbox("Python")
java = st.checkbox("Java")
sql = st.checkbox("SQL")
ml = st.checkbox("Machine Learning")
html_css = st.checkbox("HTML/CSS")

if st.button("Predict Career Role"):

    skills = {
        "python": int(python),
        "java": int(java),
        "sql": int(sql),
        "ml": int(ml),
        "html_css": int(html_css)
    }

    input_data = np.array(list(skills.values())).reshape(1, -1)
    predicted_role = model.predict(input_data)[0]

    missing_skills = skill_gap(predicted_role, skills)

    st.success(f"Recommended Role: {predicted_role}")
    st.warning(f" Missing Skills: {missing_skills}")