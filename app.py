import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

st.set_page_config(page_title="Chef AI 🍳", layout="wide")
st.title("🍳 Chef AI - Find What to Cook!")


data = pd.read_csv("food_recipes.csv")
data = data[['recipe_title','ingredients','instructions','url']]  # include url
data = data.dropna()
data["ingredients"] = data["ingredients"].str.lower()

X = data["ingredients"]
y = data["recipe_title"]

vectorizer = TfidfVectorizer(stop_words="english")
X_vector = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vector, y)


ingredients_input = st.text_input("Enter ingredients you have (comma separated):")

if ingredients_input:
    ingredients_input = ingredients_input.lower()
    input_vector = vectorizer.transform([ingredients_input])
    probs = model.predict_proba(input_vector)
    top3 = np.argsort(probs[0])[-3:]  # top 3 recipes
    top_recipes = [model.classes_[i] for i in reversed(top3)]

    st.subheader("🍽 Top 3 dishes you can cook:")
    for idx, recipe_name in enumerate(top_recipes, 1):
        st.write(f"{idx}. {recipe_name}")


    selected_recipe = st.selectbox("Select recipe to see steps:", top_recipes)
    if selected_recipe:
        recipe_row = data[data["recipe_title"] == selected_recipe]
        steps = recipe_row["instructions"].values[0]
        step_list = steps.split("|")

        st.subheader(f"📖 Steps to cook {selected_recipe}:")
        for i, step in enumerate(step_list, 1):
            step = step.strip()
            if step != "":
                st.write(f"**Step {i}:** {step}")

      
        recipe_url = recipe_row["url"].values[0]
        if recipe_url:
            st.markdown(f"[🔗 View full recipe online]({recipe_url})")

   
