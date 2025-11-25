from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)

@app.after_request
def set_cors(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Access-Control-Allow-Headers, X-Requested-With")
    return response

MODEL_INPUT_STEPS = 30  
MAX_HORIZON = 7

ort_session = ort.InferenceSession("model_forecast.onnx")
input_names = [i.name for i in ort_session.get_inputs()]

artifacts = joblib.load('training_artifacts.pkl')
scaler = artifacts['scaler']
le_prov = artifacts['label_encoder']

df = pd.read_csv('merge_dataset.csv')
df = df.sort_values(['provinsi','tanggal'])
df['harga'] = df.groupby('provinsi')['harga'].transform(lambda x: x.interpolate(method='linear'))
df['harga'] = df.groupby('provinsi')['harga'].ffill()
df['harga'] = df.groupby('provinsi')['harga'].bfill()

try:
    df['tanggal'] = pd.to_datetime(df['tanggal'])
except:
    pass 

df = df.sort_values(by=['provinsi', 'tanggal'])

def get_last_30_days_data(provinsi):
    prov_data = df[df['provinsi'] == provinsi]
    
    if prov_data.empty or len(prov_data) < MODEL_INPUT_STEPS:
        return None, None
        
    last_data = prov_data.tail(MODEL_INPUT_STEPS)
    last_prices = last_data['harga'].values.reshape(-1, 1)
    last_date = last_data['tanggal'].iloc[-1]
    
    last_prices_scaled = scaler.transform(last_prices)
    return last_prices_scaled, last_date

@app.route("/province/agregate")
def getAgregateProvince():
    agregate = df.groupby("provinsi")["harga"].describe().round(0)
    return jsonify(agregate.transpose().to_dict())

@app.route("/province/<name>")
def getDataProvince(name):
    sub = df[df['provinsi'] == name][-30:]
    sub['tanggal'] = sub['tanggal'].dt.strftime("%Y-%m-%d")
    result = {
        "provinsi": name,
        "data": sub[["tanggal", "harga"]].to_dict(orient="records")
    }
    return jsonify(result)

@app.route("/forecast", methods=["POST"])
def forecast_api():
    try:
        data = request.json
        prov_name = data.get("provinsi")
        
        req_horizon = int(data.get("horizon", 7))
        horizon = min(req_horizon, MAX_HORIZON) 

        if not prov_name:
            return jsonify({"error": "Parameter 'provinsi' wajib diisi"}), 400

        try:
            prov_id = le_prov.transform([prov_name])[0]
        except:
            return jsonify({"error": f"Provinsi '{prov_name}' tidak ditemukan"}), 400

        input_prices, last_date = get_last_30_days_data(prov_name)
        if input_prices is None:
            return jsonify({"error": "Data history kurang"}), 404

        # Input 1: Harga (1, 30, 1)
        input_seq = input_prices.reshape(1, MODEL_INPUT_STEPS, 1).astype(np.float32)
        # Input 2: Provinsi (1, 1)
        input_prov = np.array([[prov_id]]).astype(np.float32)

        onnx_inputs = {
            input_names[0]: input_seq,
            input_names[1]: input_prov
        }
        
        pred_raw = ort_session.run(None, onnx_inputs)[0]
        pred_needed = pred_raw[0][:horizon] 
        prices_rupiah = scaler.inverse_transform(pred_needed.reshape(-1, 1)).flatten()

        response_data = []
        current_iter_date = last_date
        
        for price in prices_rupiah:
            current_iter_date += timedelta(days=1)
            response_data.append({
                "tanggal": current_iter_date.strftime("%Y-%m-%d"),
                "harga": round(float(price), 0),
            })

        return jsonify({
            "provinsi": prov_name,
            "status": "success",
            "data": response_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
