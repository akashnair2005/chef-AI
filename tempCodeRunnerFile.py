import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("food_recipes.csv")

data = data[['recipe_title','ingredients','instructions']]

data = data.dropna()

data["ingredients"] = data["ingredients"].str.lower()

X = data["ingredients"]
y = data["recipe_title"]

vectorizer = TfidfVectorizer(stop_words="english")
X_vector = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vector, y)

print("🍳 Chef AI Model Ready!")

ingredients = input("\nEnter ingredients you have: ").lower()

input_vector = vectorizer.transform([ingredients])

probs = model.predict_proba(input_vector)

top3 = np.argsort(probs[0])[-3:]

print("\n🍽 Top 3 dishes you can cook:\n")

recipes = []

for idx, i in enumerate(reversed(top3), 1):
    recipe_name = model.classes_[i]
    recipes.append(recipe_name)
    print(f"{idx}. {recipe_name}")

choice = int(input("\nSelect recipe number to see steps: "))

selected_recipe = recipes[choice-1]

recipe_row = data[data["recipe_title"] == selected_recipe]

steps = recipe_row["instructions"].values[0]

print(f"\n📖 Steps to cook {selected_recipe}:\n")

step_list = steps.split("|")

for i, step in enumerate(step_list, 1):
    step = step.strip()
    if step != "":
        print(f"Step {i}: {step}\n")