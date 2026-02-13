from flask import Flask, render_template, request
import pickle
from preprocess import clean_text

app = Flask(__name__)

model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        news = request.form["news"]
        clean_news = clean_text(news)
        vect = vectorizer.transform([clean_news])
        result = model.predict(vect)[0]
        prediction = "REAL NEWS" if result == 1 else "FAKE NEWS"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
