Steps : 

1. Create python env - python -m venv venv
2. Choose interpreter - Press Ctrl + Shift + P (or F1) to     open the Command Palette.
Type: Python: Select Interpreter
Choose the interpreter from .venv (e.g., ./.venv/Scripts/python.exe).

3. Bypass command if env activation does not work - Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

4. Env activation - .\\venv\Scripts\Activate

5. Install requirements.txt - pip install -r requirements.txt

6. Run churn.py - python churn.py

7. Run app (main.py) - uvicorn main:app --reload

8. Open postman - choose POST - enter this api in url http://127.0.0.1:8000/predict - go in body and choose raw and add the below json inside body and hit Send button

Sample JSON for /predict api 

{
  "tenure": 12,
  "MonthlyCharges": 70.5,
  "TotalCharges": 845.0,
  "gender": "Female",
  "SeniorCitizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"
}
