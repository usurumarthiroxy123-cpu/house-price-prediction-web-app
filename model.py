import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample dataset
data = {
    'area': [800, 1000, 1200, 1500, 1800, 2000],
    'bedrooms': [1, 2, 2, 3, 3, 4],
    'price': [40, 50, 65, 80, 95, 110]
}

df = pd.DataFrame(data)

X = df[['area', 'bedrooms']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

# Save the model
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully")
