from flask import Flask, render_template, request, jsonify, send_file
import pickle
import numpy as np
import sqlite3
from reportlab.pdfgen import canvas
from datetime import datetime

app = Flask(__name__)


def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS patients(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gender TEXT,
            age INTEGER,
            hypertension INTEGER,
            heart_disease INTEGER,
            ever_married TEXT,
            work_type TEXT,
            residence_type TEXT,
            glucose REAL,
            bmi REAL,
            smoking_status TEXT,
            prediction TEXT,
            probability REAL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


with open("stroke_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    gender = encoders["gender"].transform([data["gender"]])[0]
    ever_married = encoders["ever_married"].transform([data["ever_married"]])[0]
    work_type = encoders["work_type"].transform([data["work_type"]])[0]
    residence = encoders["Residence_type"].transform([data["Residence_type"]])[0]
    smoking = encoders["smoking_status"].transform([data["smoking_status"]])[0]

    features = np.array([[
        gender,
        float(data["age"]),
        int(data["hypertension"]),
        int(data["heart_disease"]),
        ever_married,
        work_type,
        residence,
        float(data["avg_glucose_level"]),
        float(data["bmi"]),
        smoking
    ]])

    features = scaler.transform(features)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    risk = "High Stroke Risk" if prediction == 1 else "Low Stroke Risk"

    if prediction == 1:
        recommendation = [
            "Consult a neurologist immediately.",
            "Monitor blood pressure regularly.",
            "Exercise at least 30 minutes daily.",
            "Reduce salt intake.",
            "Quit smoking and alcohol.",
            "Maintain a healthy weight."
        ]
    else:
        recommendation = [
            "Maintain a healthy lifestyle.",
            "Exercise regularly.",
            "Eat a balanced diet.",
            "Drink enough water.",
            "Have regular health checkups."
        ]

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO patients(
            gender,
            age,
            hypertension,
            heart_disease,
            ever_married,
            work_type,
            residence_type,
            glucose,
            bmi,
            smoking_status,
            prediction,
            probability
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            data["gender"],
            int(data["age"]),
            int(data["hypertension"]),
            int(data["heart_disease"]),
            data["ever_married"],
            data["work_type"],
            data["Residence_type"],
            float(data["avg_glucose_level"]),
            float(data["bmi"]),
            data["smoking_status"],
            risk,
            round(probability * 100, 2)
        )
    )
    conn.commit()
    conn.close()

    return jsonify({
        "risk": risk,
        "probability": round(probability * 100, 2),
        "recommendation": recommendation
    })


@app.route("/history")
def history():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients")
    patients = cursor.fetchall()
    conn.close()
    return render_template("history.html", patients=patients)


@app.route("/download_report")
def download_report():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM patients
        ORDER BY id DESC
        LIMIT 1
    """)
    patient = cursor.fetchone()
    conn.close()

    if patient is None:
        return "No patient records found."

    current_datetime = datetime.now()
    current_date = current_datetime.strftime("%d-%m-%Y")
    current_time = current_datetime.strftime("%I:%M:%S %p")

    pdf = canvas.Canvas("reports/stroke_report.pdf")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(120, 800, "Brain Stroke Prediction Report")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(350, 800, f"Date : {current_date}")
    pdf.drawString(350, 780, f"Time : {current_time}")

    pdf.drawString(50, 760, f"Gender : {patient[1]}")
    pdf.drawString(50, 740, f"Age : {patient[2]}")
    pdf.drawString(50, 720, f"Hypertension : {patient[3]}")
    pdf.drawString(50, 700, f"Heart Disease : {patient[4]}")
    pdf.drawString(50, 680, f"Ever Married : {patient[5]}")
    pdf.drawString(50, 660, f"Work Type : {patient[6]}")
    pdf.drawString(50, 640, f"Residence : {patient[7]}")
    pdf.drawString(50, 620, f"Glucose : {patient[8]}")
    pdf.drawString(50, 600, f"BMI : {patient[9]}")
    pdf.drawString(50, 580, f"Smoking Status : {patient[10]}")

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, 540, f"Prediction : {patient[11]}")
    pdf.drawString(50, 520, f"Probability : {patient[12]} %")
    pdf.drawString(50, 480, "Recommendation")

    pdf.setFont("Helvetica", 12)
    if patient[11] == "High Stroke Risk":
        pdf.drawString(70, 460, "• Visit a Neurologist")
        pdf.drawString(70, 440, "• Exercise Daily")
        pdf.drawString(70, 420, "• Reduce Salt Intake")
        pdf.drawString(70, 400, "• Monitor Blood Pressure")
        pdf.drawString(70, 380, "• Quit Smoking")
    else:
        pdf.drawString(70, 460, "• Maintain Healthy Lifestyle")
        pdf.drawString(70, 440, "• Exercise Regularly")
        pdf.drawString(70, 420, "• Eat Healthy Food")
        pdf.drawString(70, 400, "• Drink Enough Water")
        pdf.drawString(70, 380, "• Regular Health Checkups")

    pdf.save()
    return send_file("reports/stroke_report.pdf", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)