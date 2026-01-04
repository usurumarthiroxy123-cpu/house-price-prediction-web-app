from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('house_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])

    prediction = model.predict(np.array([[area, bedrooms]]))
    output = round(prediction[0], 2)

    return render_template(
        'index.html',
        prediction_text=f'Estimated House Price: ₹ {output} Lakhs'
    )

if __name__ == '__main__':
    app.run(debug=True)
