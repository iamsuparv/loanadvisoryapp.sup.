from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load ML model
model = joblib.load('loan_model.pkl')

# EMI calculation
def calculate_emi(loan_amount, annual_rate, tenure_months):
    r = annual_rate / (12*100)
    emi = (loan_amount * r * (1 + r)**tenure_months) / ((1 + r)**tenure_months - 1)
    return emi

# Rule-based advice
def get_advice(income, savings, loan_amount, tenure_months):
    emi = calculate_emi(loan_amount, 10, tenure_months)
    dti = (emi / income) * 100
    if dti < 30 and savings >= emi*6:
        advice = "Safe to take loan ✅"
    elif 30 <= dti <= 40 and savings >= emi*3:
        advice = "Moderate risk ⚠️"
    else:
        advice = "Risky, not recommended ❌"
    return advice, emi, dti

@app.route('/', methods=['GET','POST'])
def index():
    advice = emi = dti = ml_status = None
    if request.method == 'POST':
        income = float(request.form['income'])
        savings = float(request.form['savings'])
        loan_amount = float(request.form['loan_amount'])
        tenure = int(request.form['tenure'])
        purpose = request.form['purpose']

        # Rule-based
        advice, emi, dti = get_advice(income, savings, loan_amount, tenure)

        # ML prediction
        input_data = np.array([[income, loan_amount, tenure]])
        prediction = model.predict(input_data)[0]
        ml_status = "Approved ✅" if prediction==1 else "Not Approved ❌"

    return render_template('index.html', advice=advice, emi=emi, dti=dti, ml_status=ml_status)

if __name__ == "__main__":
    app.run(debug=True)
