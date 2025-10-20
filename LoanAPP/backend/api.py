import requests

sample = {
    "Age": 35,
    "Income": 90000,
    "LoanAmount": 25000,
    "CreditScore": 710,
    "MonthsEmployed": 60,
    "NumCreditLines": 4,
    "InterestRate": 7.2,
    "LoanTerm": 48,
    "DTIRatio": 0.25,
    "Education": "Master's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": 1,
    "HasDependents": 0,
    "LoanPurpose": "Car",
    "HasCoSigner": 0
}

resp = requests.post("http://localhost:8000/predict", json=sample)
print(resp.status_code)
print(resp.json())
