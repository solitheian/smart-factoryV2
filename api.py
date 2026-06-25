from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# Inisialisasi Aplikasi API
app = FastAPI(title="Smart Factory API", description="API untuk Predictive Maintenance")

# Load Model dan Scaler (Kita pake Random Forest yang udah di-tune)
model_rf = joblib.load('model_mesin_rf.pkl')
scaler = joblib.load('scaler_mesin.pkl')

# Format Input dari User (JSON)
class SensorData(BaseModel):
    air_temp: float
    process_temp: float
    rot_speed: int
    torque: float
    tool_wear: int

# Endpoint utama untuk klasifikasi
@app.post("/predict")
def predict_machine_failure(data: SensorData):
    # Ubah input JSON jadi DataFrame buat scaler
    input_df = pd.DataFrame([{
        'Air temperature [K]': data.air_temp,
        'Process temperature [K]': data.process_temp,
        'Rotational speed [rpm]': data.rot_speed,
        'Torque [Nm]': data.torque,
        'Tool wear [min]': data.tool_wear
    }])
    
    # Preprocessing & Prediksi
    input_scaled = scaler.transform(input_df)
    prediction = model_rf.predict(input_scaled)[0]
    prob = model_rf.predict_proba(input_scaled)[0]
    
    # Rapihin output buat respon API
    status = "Risiko Tinggi / Gagal" if prediction == 1 else "Normal"
    confidence = prob[1] if prediction == 1 else prob[0]
    
    return {
        "status_kode": 200,
        "hasil_prediksi": int(prediction),
        "status_mesin": status,
        "confidence": f"{round(float(confidence) * 100, 2)}%"
    }

# Script untuk jalanin server lokal
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)