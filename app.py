# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
import queue
import threading
from datetime import datetime, timezone, timedelta
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

# Optional: lightweight auto-refresh helper
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# ---------------------------
# Config
# ---------------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_SENSOR = "iot/sensor/data/altf4"
TOPIC_OUTPUT = "iot/output/altf4"
MODEL_PATH = "iot_temp_model.pkl"

TZ = timezone(timedelta(hours=7))
def now_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

GLOBAL_MQ = queue.Queue()

# ---------------------------
# Streamlit setup
# ---------------------------
st.set_page_config(page_title="IoT ML Realtime Dashboard — Stable", layout="wide")
st.title("🔥 IoT ML Realtime Dashboard — Stable")

# session state
if "msg_queue" not in st.session_state:
    st.session_state.msg_queue = GLOBAL_MQ

if "logs" not in st.session_state:
    st.session_state.logs = []

if "last" not in st.session_state:
    st.session_state.last = None

if "mqtt_thread_started" not in st.session_state:
    st.session_state.mqtt_thread_started = False

if "ml_model" not in st.session_state:
    st.session_state.ml_model = None

# ---------------------------
# Load ML model
# ---------------------------
@st.cache_resource
def load_ml_model(path):
    try:
        m = joblib.load(path)
        return m
    except Exception as e:
        st.warning(f"Could not load ML model from {path}: {e}")
        return None

if st.session_state.ml_model is None:
    st.session_state.ml_model = load_ml_model(MODEL_PATH)

if st.session_state.ml_model:
    st.success(f"Model loaded: {MODEL_PATH}")
else:
    st.info("No ML model loaded. Upload iot_temp_model.pkl.")

# ---------------------------
# MQTT callbacks
# ---------------------------
def _on_connect(client, userdata, flags, rc):
    try:
        client.subscribe(TOPIC_SENSOR)
    except Exception:
        pass
    GLOBAL_MQ.put({"_type": "status", "connected": (rc == 0), "ts": time.time()})

def _on_message(client, userdata, msg):
    payload = msg.payload.decode(errors="ignore")
    try:
        data = json.loads(payload)
    except Exception:
        GLOBAL_MQ.put({"_type": "raw", "payload": payload, "ts": time.time()})
        return

    GLOBAL_MQ.put({"_type": "sensor", "data": data, "ts": time.time(), "topic": msg.topic})

# ---------------------------
# Start MQTT thread
# ---------------------------
def start_mqtt_thread_once():
    def worker():
        client = mqtt.Client()
        client.on_connect = _on_connect
        client.on_message = _on_message

        while True:
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                client.loop_forever()
            except Exception as e:
                GLOBAL_MQ.put({"_type": "error", "msg": f"MQTT worker error: {e}", "ts": time.time()})
                time.sleep(5)

    if not st.session_state.mqtt_thread_started:
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        st.session_state.mqtt_thread_started = True
        time.sleep(0.05)

start_mqtt_thread_once()

# ---------------------------
# Prediction helper
# ---------------------------
def model_predict_label_and_conf(temp, hum):
    model = st.session_state.ml_model
    if model is None:
        return ("N/A", None)

    X = [[float(temp), float(hum)]]
    try:
        label = model.predict(X)[0]
    except Exception:
        label = "ERR"

    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = float(np.max(model.predict_proba(X)))
        except Exception:
            prob = None
    return (label, prob)

# ---------------------------
# Process queue
# ---------------------------
def process_queue():
    updated = False
    q = st.session_state.msg_queue

    while not q.empty():
        item = q.get()
        ttype = item.get("_type")

        if ttype == "status":
            st.session_state.last_status = item.get("connected", False)
            updated = True

        elif ttype == "error":
            st.error(item.get("msg"))
            updated = True

        elif ttype == "raw":
            row = {"ts": now_str(), "raw": item.get("payload")}
            st.session_state.logs.append(row)
            st.session_state.last = row
            updated = True

        elif ttype == "sensor":
            d = item.get("data", {})
            temp = float(d.get("temp", 0))
            hum = float(d.get("hum", 0))

            row = {
                "ts": datetime.fromtimestamp(item.get("ts", time.time()), TZ).strftime("%Y-%m-%d %H:%M:%S"),
                "temp": temp,
                "hum": hum,
            }

            label, conf = model_predict_label_and_conf(temp, hum)
            row["pred"] = label
            row["conf"] = conf

            # anomaly
            anomaly = False
            if conf is not None and conf < 0.6:
                anomaly = True

            temps = [r["temp"] for r in st.session_state.logs if r.get("temp") is not None]
            window = temps[-30:] if len(temps) > 0 else []

            if len(window) >= 5:
                mean = float(np.mean(window))
                std = float(np.std(window))
                if std > 0:
                    z = abs((temp - mean) / std)
                    if z >= 3.0:
                        anomaly = True

            row["anomaly"] = anomaly

            st.session_state.last = row
            st.session_state.logs.append(row)
            if len(st.session_state.logs) > 5000:
                st.session_state.logs = st.session_state.logs[-5000:]

            # publish output
            try:
                pub = mqtt.Client()
                pub.connect(MQTT_BROKER, MQTT_PORT, 60)

                if label == "Panas":
                    pub.publish(TOPIC_OUTPUT, "ALERT_ON")
                else:
                    pub.publish(TOPIC_OUTPUT, "ALERT_OFF")

                pub.disconnect()
            except:
                pass

            updated = True

    return updated

# initialize
process_queue()

# ---------------------------
# NOTIFICATION ALERT BESAR
# ---------------------------
if st.session_state.last:
    pred = st.session_state.last.get("pred")

    if pred == "Panas":
        st.error("🔥 ALERT: Suhu PANAS terdeteksi!")
    elif pred == "Normal":
        st.success("🟢 Suhu NORMAL")
    elif pred == "Dingin":
        st.info("🔵 Suhu DINGIN")
    else:
        st.warning("Menunggu prediksi...")

# ---------------------------
# UI Layout
# ---------------------------
if HAS_AUTOREFRESH:
    st_autorefresh(interval=2000, key="refresh")

left, right = st.columns([1, 2])

with left:

    st.header("Connection")
    st.metric("MQTT Connected", "Yes" if getattr(st.session_state, "last_status", False) else "No")

    st.header("Last Reading")
    if st.session_state.last:
        last = st.session_state.last

        if last["pred"] == "Panas":
            st.error(f"🔥 PANAS — Temp: {last['temp']}°C, Hum: {last['hum']}%")
        elif last["pred"] == "Normal":
            st.success(f"🟢 NORMAL — Temp: {last['temp']}°C, Hum: {last['hum']}%")
        elif last["pred"] == "Dingin":
            st.info(f"🔵 DINGIN — Temp: {last['temp']}°C, Hum: {last['hum']}%")
        else:
            st.write(last)

    st.header("Manual Output")
    col1, col2 = st.columns(2)
    if col1.button("ALERT_ON"):
        pub = mqtt.Client()
        pub.connect(MQTT_BROKER, MQTT_PORT, 60)
        pub.publish(TOPIC_OUTPUT, "ALERT_ON")
        pub.disconnect()

    if col2.button("ALERT_OFF"):
        pub = mqtt.Client()
        pub.connect(MQTT_BROKER, MQTT_PORT, 60)
        pub.publish(TOPIC_OUTPUT, "ALERT_OFF")
        pub.disconnect()

    st.header("Download Logs")
    if st.button("Export CSV"):
        if st.session_state.logs:
            df_dl = pd.DataFrame(st.session_state.logs)
            st.download_button("Download CSV", df_dl.to_csv(index=False), "logs.csv")
        else:
            st.info("No logs yet.")

with right:
    st.header("Live Chart (200 points)")
    df = pd.DataFrame(st.session_state.logs[-200:])
    if not df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df["temp"], name="Temp"))
        fig.add_trace(go.Scatter(x=df["ts"], y=df["hum"], name="Humidity", yaxis="y2"))

        fig.update_layout(
            yaxis=dict(title="Temp"),
            yaxis2=dict(title="Hum", overlaying="y", side="right"),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

process_queue()

# ---------------------------
# TABEL DATA
# ---------------------------
st.header("Tabel Data (Realtime)")
df_table = pd.DataFrame(st.session_state.logs[-300:])
if not df_table.empty:
    st.dataframe(df_table, use_container_width=True, height=400)
else:
    st.info("Belum ada data masuk.")
