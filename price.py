from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import sqlite3
import bcrypt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a strong secret key

# ---------------------------
# Initialize SQLite database
# ---------------------------
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ---------------------------
# Load and prepare ML model
# ---------------------------
df = pd.read_csv('bitcoin_price_data.csv')

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce')

df.dropna(subset=numeric_cols + ['Date'], inplace=True)
df.sort_values('Date', inplace=True)

df['Open_Close_diff'] = df['Open'] - df['Close']
df['High_Low_diff'] = df['High'] - df['Low']
df['Pct_change_1d'] = df['Close'].pct_change()
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

features = ['Open', 'High', 'Low', 'Volume', 'Open_Close_diff', 'High_Low_diff', 'Pct_change_1d']
X = df[features].values
y = df['Target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
model.fit(X_train_scaled, y_train)

# ---------------------------
# Helper for future prediction
# ---------------------------
def predict_future_price(target_date_str):
    last_date = df['Date'].max()
    last_row_df = df[df['Date'] == last_date]

    if last_row_df.empty:
        return None, "No data available for last date in dataset."

    last_row = last_row_df.iloc[0].copy()

    try:
        target_date = pd.to_datetime(target_date_str)
    except Exception:
        return None, "Invalid date format. Use YYYY-MM-DD."

    if target_date <= last_date:
        return None, f"Target date must be AFTER the last dataset date ({last_date.strftime('%Y-%m-%d')})."

    days_diff = (target_date - last_date).days
    current_row = last_row
    current_date = last_date

    for _ in range(days_diff):
        input_features = np.array([
            current_row['Open'],
            current_row['High'],
            current_row['Low'],
            current_row['Volume'],
            current_row['Open_Close_diff'],
            current_row['High_Low_diff'],
            current_row['Pct_change_1d']
        ]).reshape(1, -1)

        input_scaled = scaler.transform(input_features)
        predicted_close = model.predict(input_scaled)[0]

        next_date = current_date + timedelta(days=1)
        new_open = current_row['Close']
        high_change = np.random.uniform(0, 0.02)
        low_change = np.random.uniform(0, 0.02)

        new_high = max(new_open, predicted_close) * (1 + high_change)
        new_low = min(new_open, predicted_close) * (1 - low_change)
        new_volume = current_row['Volume'] * np.random.uniform(0.95, 1.05)

        open_close_diff = new_open - predicted_close
        high_low_diff = new_high - new_low
        pct_change_1d = (predicted_close - current_row['Close']) / current_row['Close']

        new_row = {
            'Date': next_date,
            'Close': predicted_close,
            'Open': new_open,
            'High': new_high,
            'Low': new_low,
            'Volume': new_volume,
            'Open_Close_diff': open_close_diff,
            'High_Low_diff': high_low_diff,
            'Pct_change_1d': pct_change_1d
        }

        current_row = pd.Series(new_row)
        current_date = next_date

    return predicted_close, None

# ---------------------------
# Flask Routes
# ---------------------------

@app.route('/')
def home():
    return redirect(url_for('login'))

# --- Register Route ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username and password:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            try:
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
                conn.commit()
                conn.close()
                session['user'] = username
                return redirect(url_for('predict'))
            except sqlite3.IntegrityError:
                error = "Username already exists."
        else:
            error = "Please enter both username and password."
    return render_template('register.html', error=error)

# --- Login Route ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password_input = request.form.get('password')

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if user and bcrypt.checkpw(password_input.encode(), user[0]):
            session['user'] = username
            return redirect(url_for('predict'))
        else:
            error = "Invalid username or password."

    return render_template('login.html', error=error)

# --- Logout Route ---
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# --- Prediction Page ---
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    predicted_price = None
    error = None

    if request.method == 'POST':
        date_input = request.form.get('date')
        option = request.form.get('option')

        if option == '1':
            try:
                input_date = pd.to_datetime(date_input)
            except Exception:
                error = "Invalid date format. Use YYYY-MM-DD."
                return render_template('predict.html', predicted_price=predicted_price, error=error)

            row = df[df['Date'] == input_date]
            if row.empty:
                error = "Date not found in dataset."
            else:
                X_input = row[features].values
                X_input_scaled = scaler.transform(X_input)
                pred = model.predict(X_input_scaled)[0]
                predicted_price = f"Predicted closing price for the NEXT day after {date_input}: ${pred:,.2f}"

        elif option == '2':
            predicted_close, err = predict_future_price(date_input)
            if err:
                error = err
            else:
                predicted_price = f"Predicted closing price on {date_input}: ${predicted_close:,.2f}"
        else:
            error = "Invalid option selected."

    return render_template('predict.html', predicted_price=predicted_price, error=error)

# ---------------------------
# Run App
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)
