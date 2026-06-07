from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))

def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

model = load_pickle(os.path.join(base_dir, 'static', 'model', 'model.pkl'))
vectorizer = load_pickle(os.path.join(base_dir, 'artifacts', 'vectorizer.pkl'))
selector = load_pickle(os.path.join(base_dir, 'artifacts', 'selector.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review', '').strip()

    if not review:
        return render_template('index.html', error='Please enter a movie review for sentiment analysis.')

    review_vectorized = vectorizer.transform([review])
    review_selected = selector.transform(review_vectorized)
    prediction = model.predict(review_selected)

    sentiment = 'Positive 😊' if int(prediction[0]) == 1 else 'Negative 😞'
    label = 'Positive' if int(prediction[0]) == 1 else 'Negative'

    return render_template('index.html', prediction=sentiment, label=label, review=review)

# ALWAYS KEEP THIS AT THE END
if __name__ == "__main__":
    app.run(debug=True)