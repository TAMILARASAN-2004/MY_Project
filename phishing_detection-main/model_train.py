import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("phishing_data.csv")

vec = TfidfVectorizer(ngram_range=(1,2))
X = vec.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vec, open("vectorizer.pkl","wb"))

print("Model trained and saved")