import os
import pickle
import io
import base64

import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

# ================= FLASK APP =================
app = Flask(__name__)
print("🚀 Flask app is starting...")

# ================= BASE PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "dataset", "train.csv")

# ================= LOAD MODEL =================
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Error loading model:", e)
else:
    print("⚠️ Model file not found (prediction will still open)")

# ================= LOAD DATA =================
if os.path.exists(DATA_PATH):
    transaction = pd.read_csv(DATA_PATH)
    print("✅ Dataset loaded successfully")
else:
    print("⚠️ Dataset not found (analysis page will still open)")
    transaction = pd.DataFrame()

# ================= DATE CLEANING =================
if "date" in transaction.columns:
    transaction["date"] = pd.to_datetime(transaction["date"], errors="coerce")

# ================= SALES ANALYSIS =================
def get_sales_analysis():
    plots = {}

    # ---- Total Sales ----
    total_sales = (
        transaction["avg_transaction_value"].sum()
        if "avg_transaction_value" in transaction.columns
        else 0
    )

    # ---- Top Products ----
    if {"product", "avg_transaction_value"}.issubset(transaction.columns):
        top_products = (
            transaction
            .groupby("product")["avg_transaction_value"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
    else:
        top_products = pd.Series(dtype=float)

    # ---- Churn vs Revenue ----
    if {"churn", "avg_transaction_value"}.issubset(transaction.columns):
        churn_revenue = transaction.groupby("churn")["avg_transaction_value"].sum()
    else:
        churn_revenue = pd.Series(dtype=float)

    # ---- Plot: Top Products ----
    if not top_products.empty:
        plt.figure(figsize=(9, 4))
        top_products.plot(kind="bar")
        plt.title("Top 10 Products by Revenue")
        plt.xlabel("Product")
        plt.ylabel("Revenue")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plots["top_products"] = base64.b64encode(buf.read()).decode()
        plt.close()

    # ---- Plot: Churn vs Revenue ----
    if not churn_revenue.empty:
        plt.figure(figsize=(5, 4))
        churn_revenue.plot(kind="bar")
        plt.title("Revenue by Churn Status")
        plt.xlabel("Churn (0 = Active, 1 = Churned)")
        plt.ylabel("Revenue")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plots["churn_revenue"] = base64.b64encode(buf.read()).decode()
        plt.close()

    # ---- Monthly Sales Trend ----
    monthly_plot = None
    if {"date", "avg_transaction_value"}.issubset(transaction.columns):
        transaction["month"] = transaction["date"].dt.to_period("M")
        monthly_sales = transaction.groupby("month")["avg_transaction_value"].sum()

        plt.figure(figsize=(8, 4))
        monthly_sales.plot(kind="line", marker="o")
        plt.title("Monthly Sales Trend")
        plt.xlabel("Month")
        plt.ylabel("Sales")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        monthly_plot = base64.b64encode(buf.read()).decode()
        plt.close()

    return {
        "total_sales": round(total_sales, 2),
        "top_products": top_products.to_dict(),
        "churn_revenue": churn_revenue.to_dict(),
        "plots": plots,
        "monthly_plot": monthly_plot,
    }

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analysis")
def analysis():
    data = get_sales_analysis()
    return render_template(
        "churn.html",
        total_sales=data["total_sales"],
        top_products=data["top_products"],
        churn_revenue=data["churn_revenue"],
        plots=data["plots"],
        monthly_plot=data["monthly_plot"],
    )

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    prediction_text = None

    if request.method == "POST":
        input_df = pd.DataFrame({
            "age": [int(request.form.get("age", 0))],
            "days_since_last_login": [int(request.form.get("last_login", 0))],
            "avg_time_spent": [float(request.form.get("avg_time_spent", 0))],
            "avg_transaction_value": [float(request.form.get("avg_transaction_value", 0))],
            "points_in_wallet": [float(request.form.get("points_in_wallet", 0))]
        })

        if model:
            result = model.predict(input_df)[0]
            prediction_text = f"Churn Prediction: {result}"
        else:
            prediction_text = "⚠️ Model not loaded"

    return render_template("prediction.html", prediction_text=prediction_text)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)




