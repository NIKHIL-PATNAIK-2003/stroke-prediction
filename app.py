import numpy as np
import pandas as pd
import os
import sqlite3
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, redirect, session
import pymysql

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import chi2, SelectKBest

import catboost as cb
import shap
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'welcome')

# ---------------- DATABASE ----------------
class SQLiteCursor:
    def __init__(self, cursor):
        self.cursor = cursor

    def execute(self, query, params=()):
        return self.cursor.execute(query.replace("%s", "?"), params)

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchall(self):
        return self.cursor.fetchall()


class SQLiteConnection:
    def __init__(self, connection):
        self.connection = connection

    def cursor(self):
        return SQLiteCursor(self.connection.cursor())

    def commit(self):
        self.connection.commit()

    def close(self):
        self.connection.close()


def init_sqlite(connection):
    cur = connection.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            result TEXT NOT NULL,
            probability REAL NOT NULL,
            risk TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute(
        "INSERT OR IGNORE INTO users(name,email,password) VALUES(?,?,?)",
        ("Admin", "admin", "admin")
    )
    connection.commit()


def init_mysql(connection):
    cur = connection.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_email VARCHAR(255) NOT NULL,
            result VARCHAR(50) NOT NULL,
            probability FLOAT NOT NULL,
            risk VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute(
        "INSERT IGNORE INTO users(name,email,password) VALUES(%s,%s,%s)",
        (
            os.environ.get("ADMIN_NAME", "Admin"),
            os.environ.get("ADMIN_EMAIL", "admin"),
            os.environ.get("ADMIN_PASSWORD", "admin"),
        )
    )
    connection.commit()


def getConnection():
    try:
        connection = pymysql.connect(
            host=os.environ.get('MYSQLHOST', 'localhost'),
            port=int(os.environ.get('MYSQLPORT', 3306)),
            user=os.environ.get('MYSQLUSER', 'root'),
            password=os.environ.get('MYSQLPASSWORD', ''),
            database=os.environ.get('MYSQLDATABASE', 'stroke_db')
        )
        init_mysql(connection)
        return connection
    except pymysql.err.OperationalError:
        connection = sqlite3.connect("local_stroke.db")
        init_sqlite(connection)
        return SQLiteConnection(connection)


# ---------------- LOAD + TRAIN MODEL ----------------
dataset = pd.read_csv("Dataset/healthcare-dataset-stroke-data.csv")
dataset.fillna(0, inplace=True)

labels = ['Normal', 'Stroke']

# Label Encoding
enc1, enc2, enc3, enc4, enc5 = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()

dataset['gender'] = enc1.fit_transform(dataset['gender'].astype(str))
dataset['ever_married'] = enc2.fit_transform(dataset['ever_married'].astype(str))
dataset['work_type'] = enc3.fit_transform(dataset['work_type'].astype(str))
dataset['Residence_type'] = enc4.fit_transform(dataset['Residence_type'].astype(str))
dataset['smoking_status'] = enc5.fit_transform(dataset['smoking_status'].astype(str))

Y = dataset['stroke']
dataset.drop(['id','stroke'], axis=1, inplace=True)

# -------- CORRECT PIPELINE --------
scaler = MinMaxScaler()
X = scaler.fit_transform(dataset.values)

X, Y = SMOTE().fit_resample(X, Y)

selector = SelectKBest(chi2, k=9)
X = selector.fit_transform(X, Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Better model (handling imbalance)
model = cb.CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    scale_pos_weight=5,
    verbose=0
)

model.fit(X_train, y_train)

# SHAP Explainer (create once)
explainer = shap.TreeExplainer(model)

# Ensure folder exists
os.makedirs("static/images", exist_ok=True)


# ---------------- ROUTES ----------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        con = getConnection()
        cur = con.cursor()
        cur.execute("INSERT INTO users(name,email,password) VALUES(%s,%s,%s)",
                    (request.form['name'], request.form['email'], request.form['password']))
        con.commit()
        con.close()
        return redirect('/login')

    return render_template('register.html')


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        con = getConnection()
        cur = con.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s AND password=%s",
                    (request.form['email'], request.form['password']))
        user = cur.fetchone()
        con.close()

        if user:
            session['user'] = request.form['email']
            return redirect('/dashboard')
        else:
            return render_template('login.html', msg="Invalid login")

    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/login')

    con = getConnection()
    cur = con.cursor()

    # Total predictions
    cur.execute("SELECT COUNT(*) FROM predictions WHERE user_email=%s", (session['user'],))
    total = cur.fetchone()[0]

    # Stroke count
    cur.execute("SELECT COUNT(*) FROM predictions WHERE result='Stroke' AND user_email=%s", (session['user'],))
    stroke = cur.fetchone()[0]

    # Normal count
    cur.execute("SELECT COUNT(*) FROM predictions WHERE result='Normal' AND user_email=%s", (session['user'],))
    normal = cur.fetchone()[0]

    con.close()

    return render_template(
        'dashboard.html',
        total=total,
        stroke=stroke,
        normal=normal
    )

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')


# ---------------- PREDICTION ----------------

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'user' not in session:
        return redirect('/login')

    if request.method == 'POST':

        # Input dataframe
        df = pd.DataFrame([[
            request.form['gender'],
            float(request.form['age']),
            int(request.form['hypertension']),
            int(request.form['heart_disease']),
            request.form['ever_married'],
            request.form['work_type'],
            request.form['Residence_type'],
            float(request.form['avg_glucose_level']),
            float(request.form['bmi']),
            request.form['smoking_status']
        ]], columns=[
            'gender','age','hypertension','heart_disease',
            'ever_married','work_type','Residence_type',
            'avg_glucose_level','bmi','smoking_status'
        ])

        # Encode (same as training)
        df['gender'] = enc1.transform(df['gender'])
        df['ever_married'] = enc2.transform(df['ever_married'])
        df['work_type'] = enc3.transform(df['work_type'])
        df['Residence_type'] = enc4.transform(df['Residence_type'])
        df['smoking_status'] = enc5.transform(df['smoking_status'])

        # Apply SAME scaler + selector
        df_scaled = scaler.transform(df)
        df_selected = selector.transform(df_scaled)

        # Prediction
        pred = model.predict(df_selected)[0]
        prob = model.predict_proba(df_selected)[0][1]

        risk = "Low" if prob < 0.3 else "Medium" if prob < 0.7 else "High"

        # -------- SHAP (single prediction waterfall) --------
        shap_values = explainer.shap_values(df_selected)

        plt.figure()
        shap.summary_plot(shap_values, df_selected, show=False)

        img_path = "static/images/shap.png"
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()

        # Save to DB
        con = getConnection()
        cur = con.cursor()
        cur.execute(
            "INSERT INTO predictions(user_email,result,probability,risk) VALUES(%s,%s,%s,%s)",
            (session['user'], labels[pred], float(prob), risk)
        )
        con.commit()
        con.close()

        return render_template(
            'result.html',
            result=labels[pred],
            prob=round(prob*100, 2),
            risk=risk
        )

    return render_template('predict.html')


# ---------------- HISTORY ----------------

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect('/login')

    con = getConnection()
    cur = con.cursor()

    cur.execute("SELECT result,probability,risk,created_at FROM predictions WHERE user_email=%s",
                (session['user'],))
    data = cur.fetchall()

    cur.execute("SELECT COUNT(*) FROM predictions WHERE risk='High' AND user_email=%s",(session['user'],))
    high = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions WHERE risk='Medium' AND user_email=%s",(session['user'],))
    medium = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions WHERE risk='Low' AND user_email=%s",(session['user'],))
    low = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions WHERE result='Stroke' AND user_email=%s",(session['user'],))
    stroke = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions WHERE result='Normal' AND user_email=%s",(session['user'],))
    normal = cur.fetchone()[0]

    con.close()

    return render_template(
        'history.html',
        data=data,
        high=high,
        medium=medium,
        low=low,
        stroke=stroke,
        normal=normal
    )


# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_DEBUG') == '1'
    )
