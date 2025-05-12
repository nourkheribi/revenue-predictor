from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and optionally scaler
model = joblib.load('rf2.pkl')

category_map = {'Protein': 0, 'Vitamin': 1, 'Omega': 2, 'Performance': 3, 'Amino Acid': 4,
                'Mineral': 5, 'Herbal': 6, 'Sleep Aid': 7, 'Fat Burner': 8, 'Hydration': 9}
location_map = {'Canada': 0, 'UK': 1, 'USA': 2}
platform_map = {'Amazon': 0, 'Walmart': 1, 'iHerb': 2}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    category = request.form['category']
    units_sold = float(request.form['units_sold'])
    price = float(request.form['price'])
    discount = float(request.form['discount'])
    returned = float(request.form['returned'])
    location = request.form['location']
    platform = request.form['platform']

    data = pd.DataFrame([{
        'Category': category_map[category],
        'Units Sold': units_sold,
        'Price': price,
        'Discount': discount,
        'Units Returned': returned,
        'Location': location_map[location],
        'Platform': platform_map[platform]
    }])

    # If you have a saved scaler, use it here
    scaler = StandardScaler()
    data[['Units Sold', 'Price', 'Discount']] = scaler.fit_transform(data[['Units Sold', 'Price', 'Discount']])

    prediction = model.predict(data)[0]

    return render_template('result.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
