from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# MongoDB connection details
MONGO_URI = "mongodb+srv://vinith:vinith143@cluster0.t4ah2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
ADMIN_DB_NAME = "admin_database"
ADMIN_COLLECTION_NAME = "users"
REGISTRATION_DB_NAME = "registration_database"
REGISTRATION_COLLECTION_NAME = "users"

# Initialize MongoDB clients and databases
client = MongoClient(MONGO_URI)

# Admin Database
admin_db = client[ADMIN_DB_NAME]
admin_users_collection = admin_db[ADMIN_COLLECTION_NAME]

# Registration Database
registration_db = client[REGISTRATION_DB_NAME]
registration_users_collection = registration_db[REGISTRATION_COLLECTION_NAME]

# Temporary storage for OTP
otp_storage = {}

# Create a test admin user
admin_email = "ckvinith786@gmail.com"
admin_password = "vinith143@V"
if not admin_users_collection.find_one({"email": admin_email}):
    admin_users_collection.insert_one({"email": admin_email, "password": admin_password})
    print("Test admin user created successfully in the admin database!")
else:
    print("Test admin user already exists in the admin database!")

#volume route
@app.route('/volme')
def volume():
    return render_template('volume.html')
# Admin Login Page
@app.route('/')
def index():
    return render_template('admin_login.html')

# Admin Login
@app.route('/admin_login', methods=['POST'])
def admin_login():
    email = request.form.get('email')
    password = request.form.get('password')

    if not email or not password:
        return jsonify({"error": "Email and Password are required"}), 400

    user = admin_users_collection.find_one({"email": email})

    if user and user['password'] == password:
        return render_template('admin_dashboard.html')  # Admin dashboard page
    else:
        return jsonify({"error": "Invalid email or password"}), 401

# User Registration Page
@app.route('/registration_page')
def registration_page():
    return render_template('user_registration.html')

# User Registration
@app.route('/register', methods=['POST'])
def register():
    data = request.form
    firstname = data.get('firstname')
    lastname = data.get('lastname')
    email = data.get('email')
    mobile = data.get('mobile')
    password = data.get('password')
    confirm_password = data.get('confirm_password')

    if not all([firstname, lastname, email, mobile, password, confirm_password]):
        return jsonify({"error": "All fields are required"}), 400

    if password != confirm_password:
        return jsonify({"error": "Passwords do not match"}), 400

    if registration_users_collection.find_one({"mobile": mobile}):
        return jsonify({"error": "Mobile number already registered"}), 400

    registration_users_collection.insert_one({
        "firstname": firstname,
        "lastname": lastname,
        "email": email,
        "mobile": mobile,
        "password": password
    })
    return render_template('user_login.html')

# User Login
@app.route('/login', methods=['GET'])
def login():
    return render_template('user_login.html')

@app.route('/validate', methods=['POST'])
def validate():
    data = request.form
    email = data.get('email')
    password = data.get('password')
    if registration_users_collection.find_one({"email": email, "password": password}):
        return render_template('home.html')
    return "Enter a Valid Email Or Password"

# Fetch all registered users for Admin Dashboard
@app.route('/api/users', methods=['GET'])
def get_users():
    try:
        users = list(registration_users_collection.find({}, {"_id": 0}))  # Exclude `_id` field
        return jsonify(users)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Stock Prediction Route
    # Updated list of stock symbols (Companies)
companies = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'WMT', 'NFLX', 'ORCL', 'META', 'MCD', 'BAC']

# Function to fetch stock data
def fetch_stock_data(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period="1y")  # Get 1 year of stock data
    return data

# Feature Engineering for the data
def prepare_data(data):
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date']).map(lambda x: x.timestamp())
    features = data[['Date']]
    target = data['Close']
    return features, target

# Train the model and save it to disk
def train_and_save_model(stock_symbol, features, target):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the model and scaler with the stock symbol in the filename
    pickle.dump(model, open(f'model_{stock_symbol}.pkl', 'wb'))
    pickle.dump(scaler, open(f'scaler_{stock_symbol}.pkl', 'wb'))

    return model, scaler

# Predict the next day's stock price
def predict_next_day(model, scaler, last_date):
    last_date_scaled = scaler.transform([[last_date]])
    prediction = model.predict(last_date_scaled)
    return prediction[0]

# Load model and scaler for a specific stock symbol if they exist, otherwise train a new model
def load_model(stock_symbol):
    model_path = f'model_{stock_symbol}.pkl'
    scaler_path = f'scaler_{stock_symbol}.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = pickle.load(open(model_path, 'rb'))
        scaler = pickle.load(open(scaler_path, 'rb'))
        return model, scaler
    else:
        return None, None

# Function to calculate stock growth (percentage change over 1 year)
def calculate_growth(data):
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    growth = ((end_price - start_price) / start_price) * 100
    return growth, start_price, end_price


#Us Stock Prediction 
# Flask route for the main page (UI)
@app.route('/us_stocks', methods=['POST','GET'])
def us_stocks():
    return render_template('index.html', companies=companies)

# Flask route for prediction logic
@app.route('/predict', methods=['GET'])
def predict():
    stock_symbol = request.args.get('symbol')

    if not stock_symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400

    # Fetch stock data for the selected company          
    data = fetch_stock_data(stock_symbol)
    features, target = prepare_data(data)

    # Try to load the model specific to this stock
    model, scaler = load_model(stock_symbol)

    # If no model exists, train a new one
    if model is None or scaler is None:
        model, scaler = train_and_save_model(stock_symbol, features, target)

    # Get the last date for prediction
    last_date = features['Date'].iloc[-1]

    # Predict the next day's stock price
    next_day_prediction = predict_next_day(model, scaler, last_date)

    # Calculate stock growth over the past year
    growth, start_price, end_price = calculate_growth(data)

    # Suggest Buy/Sell/Hold
    suggestion = "Hold"
    if growth > 10:
        suggestion = "Buy"
    elif growth < 0:
        suggestion = "Sell"

    return jsonify({
        'prediction': next_day_prediction,
        'growth': growth,
        'start_price': start_price,
        'end_price': end_price,
        'suggestion': suggestion
    })


#Indian Stock Prediction 

# List of companies and their stock symbols
companies_ns = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Wipro": "WIPRO.NS",
    "Zomato": "ZOMATO.NS",
    "NCL": "NCLIND.NS",
    "Airtel": "BHARTIARTL.NS",
    "Indian Bank": "INDIANB.NS",
    "Indian Overseas Bank": "IOB.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
}

@app.route("/indian_stocks", methods=['POST','GET'])
def indian_stocks():
    return render_template("index1.html", companies=companies_ns.keys())

@app.route("/predict_us", methods=["GET"])
def predict_us():
    symbol = request.args.get("symbol")
    if not symbol or symbol not in companies_ns:
        return jsonify({"error": "Invalid company name."}), 400

    # Get the stock symbol
    stock_symbol = companies_ns[symbol]

    try:
        # Fetch historical stock data for the past year
        stock_data = yf.Ticker(stock_symbol)
        historical = stock_data.history(period="1y")

        if historical.empty:
            return jsonify({"error": "No historical data available."}), 404

        # Prepare data for prediction
        historical.reset_index(inplace=True)
        historical["Days"] = range(1, len(historical) + 1)

        # Features and target
        X = historical[["Days"]]
        y = historical["Close"]

        # Train-Test split and model training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict next day's price
        next_day = [[historical["Days"].max() + 1]]
        predicted_price = model.predict(next_day)[0]

        # Calculate growth and other metrics
        start_price = historical["Close"].iloc[0]
        end_price = historical["Close"].iloc[-1]
        growth = ((end_price - start_price) / start_price) * 100

        # Generate suggestion based on predicted price and actual end price
        suggestion = "Buy" if predicted_price > end_price else "Sell"

        # Respond with calculated data
        return jsonify({
            "prediction": round(predicted_price, 2),
            "growth": round(growth, 2),
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "suggestion": suggestion,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
