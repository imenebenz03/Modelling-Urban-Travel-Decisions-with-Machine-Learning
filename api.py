import os
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# CONFIG

TRIPS_FILE = "C:/Users/Elitebook/Desktop/dashboard/trips_with_coordinates.xlsx"

BASE_URL_WEATHER_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
BASE_URL_WEATHER_CURRENT = "https://api.open-meteo.com/v1/forecast"

VALID_MODES = ["Car driver", "Car passenger", "Walking", "Public bus", "Bicycle"]


#  FEASIBILITY & WEATHER CONSTRAINTS
=
MAX_WALK_DIST_KM = 3.0
MAX_WALK_TIME_MIN = 45.0
MAX_BIKE_DIST_KM = 20.0

HEAVY_RAIN_CODES = {55, 63, 65, 82}
LIGHT_RAIN_CODES = {51, 53, 61, 80}


#  PERSONALISATION PARAMETERS

ALPHA = 0.4
BETA = 0.3
GAMMA = 1.0

MODE_ATTRIBUTES = {
    "Walking":        {"em": 1.0, "tm": 0.2, "car": 0, "bike": 0},
    "Bicycle":        {"em": 0.9, "tm": 0.5, "car": 0, "bike": 1},
    "Public bus":     {"em": 0.7, "tm": 0.6, "car": 0, "bike": 0},
    "Car passenger":  {"em": 0.3, "tm": 0.8, "car": 0, "bike": 0},
    "Car driver":     {"em": 0.1, "tm": 1.0, "car": 1, "bike": 0},
}


#    CO₂ EMISSIONS

EMISSION_FACTORS_G_PER_KM = {
    "Walking": 0.0,
    "Cycling": 0.0,
    "Public Bus": 80.0,
    "Car Passenger": 120.0,
    "Car Driver": 240.0
}


#    FLASK APP INIT

app = Flask(__name__)
CORS(app)

le_mode = None
le_purpose = None
imputer = None
model = None
model_features = None


#   HELPERS

def get_flag(data: dict, keys, default="no") -> str:
    """Return the first found key's string value lowercased, else default."""
    for k in keys:
        if k in data and data[k] is not None:
            return str(data[k]).strip().lower()
    return str(default).strip().lower()


#   UTILITIES

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * (2 * np.arcsin(np.sqrt(a)))

def get_weather_archive(lat, lon, date, hour):
    try:
        r = requests.get(
            BASE_URL_WEATHER_ARCHIVE,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": date.strftime("%Y-%m-%d"),
                "end_date": date.strftime("%Y-%m-%d"),
                "hourly": "temperature_2m,precipitation,wind_speed_10m,weathercode",
                "timezone": "auto",
            },
            timeout=15
        )
        hourly = pd.DataFrame(r.json().get("hourly", {}))
        if hourly.empty:
            return np.nan, np.nan, np.nan, np.nan

        hourly["time"] = pd.to_datetime(hourly["time"])
        target = datetime(date.year, date.month, date.day, hour)
        row = hourly.iloc[(hourly["time"] - target).abs().argmin()]

        return (
            float(row["temperature_2m"]),
            float(row["precipitation"]),
            float(row["wind_speed_10m"]),
            int(row["weathercode"])
        )
    except:
        return np.nan, np.nan, np.nan, np.nan

def get_current_weather(lat, lon):
    try:
        r = requests.get(
            BASE_URL_WEATHER_CURRENT,
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=5
        )
        cw = r.json().get("current_weather")
        if cw:
            return cw["temperature"], cw["windspeed"], cw["weathercode"]
    except:
        pass
    return 20.0, 5.0, 0


#   TRAINING PIPELINE

def startup_pipeline():
    global le_mode, le_purpose, imputer, model, model_features

  
    df = pd.read_excel(TRIPS_FILE)
    df = df[df["MAINMODE"].isin(VALID_MODES)].reset_index(drop=True)

    df["date"] = pd.to_datetime(
        df[["TRAVYEAR", "TRAVMONTH", "TRAVDATE"]]
        .rename(columns={"TRAVYEAR": "year", "TRAVMONTH": "month", "TRAVDATE": "day"})
    )

    def fetch_weather(row):
        hour = int(row.STARTIME // 60) if not pd.isna(row.STARTIME) else 12
        return get_weather_archive(row.orig_latitude, row.orig_longitude, row.date.date(), hour)

 
    rows = list(df.itertuples(index=False))
    df["temp"], df["precip"], df["wind"], df["weather_code"] = zip(
        *ThreadPoolExecutor(max_workers=10).map(fetch_weather, rows)
    )

    le_purpose = LabelEncoder()
    df["OVERALL_PURPOSE_encoded"] = le_purpose.fit_transform(df["OVERALL_PURPOSE"].astype(str))

    le_mode = LabelEncoder()
    df["MAINMODE_encoded"] = le_mode.fit_transform(df["MAINMODE"])

    model_features = [
        "DURATION", "CUMDIST", "STARTIME", "TRAVDOW",
        "OVERALL_PURPOSE_encoded",
        "TRAVYEAR", "TRAVMONTH", "TRAVDATE",
        "orig_latitude", "orig_longitude",
        "dest_latitude", "dest_longitude",
        "temp", "precip", "wind"
    ]

    X = df[model_features]
    y = df["MAINMODE_encoded"]

    imputer = SimpleImputer(strategy="mean")
    X_imp = imputer.fit_transform(X)

    print("🌲 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_imp, y)

    print("✅ Training completed")

startup_pipeline()


#   PREDICT

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json or {}

        origin_lat = float(data["origin_lat"])
        origin_lng = float(data["origin_lng"])
        dest_lat = float(data["dest_lat"])
        dest_lng = float(data["dest_lng"])

   
        eco = get_flag(data, ["eco_friendly", "eco-friendly", "ecoFriendly", "eco"], "no")
        time_pref = get_flag(data, ["timesensitive", "time_sensitive", "time-sensitive", "timeSensitive"], "no")
        car = get_flag(data, ["car"], "no")
        bike = get_flag(data, ["bike"], "no")

        dt = datetime.strptime(f"{data['date']} {data['time']}", "%Y-%m-%d %H:%M")

        cumdist = haversine_km(origin_lat, origin_lng, dest_lat, dest_lng)
        duration = (cumdist / 30) * 60

        
        if dt.date() < datetime.now().date():
            temp, precip, wind, weather_code = get_weather_archive(
                origin_lat, origin_lng, dt.date(), dt.hour
            )
        else:
            temp, wind, weather_code = get_current_weather(origin_lat, origin_lng)
            precip = 0.0

        stm = dt.hour * 60 + dt.minute
        dow = dt.weekday()
        p_encoded = le_purpose.transform([data["purpose"]])[0]

        row = [
            duration, cumdist, stm, dow, p_encoded,
            dt.year, dt.month, dt.day,
            origin_lat, origin_lng,
            dest_lat, dest_lng,
            temp, precip, wind
        ]

        X_row = imputer.transform([row])
        proba = model.predict_proba(X_row)[0]
        modes = le_mode.inverse_transform(model.classes_)

        Weco = 1 if eco == "yes" else 0
        Wtime = 1 if time_pref == "yes" else 0
        Wcar = 1 if car == "yes" else 0
        Wbike = 1 if bike == "yes" else 0

        utilities = []

        for i, m in enumerate(modes):
            a = MODE_ATTRIBUTES[m]

          
            if Wcar == 0 and m in ["Car driver", "Car passenger"]:
                utilities.append(0.0)
                continue
            if Wbike == 0 and m == "Bicycle":
                utilities.append(0.0)
                continue

          
            if m == "Walking" and (cumdist > MAX_WALK_DIST_KM or duration > MAX_WALK_TIME_MIN):
                utilities.append(0.0)
                continue
            if m == "Bicycle" and cumdist > MAX_BIKE_DIST_KM:
                utilities.append(0.0)
                continue

            penalty = (1 - Wcar) * a["car"] + (1 - Wbike) * a["bike"]

            Um = proba[i] * (
                1
                + ALPHA * Weco * a["em"]
                + BETA * Wtime * a["tm"]
                - GAMMA * penalty
            )

          
            if weather_code in HEAVY_RAIN_CODES and m in ["Walking", "Bicycle"]:
                Um *= 0.3
            elif weather_code in LIGHT_RAIN_CODES and m in ["Walking", "Bicycle"]:
                Um *= 0.6

            utilities.append(max(Um, 0.0))

        utilities = np.array(utilities, dtype=float)
        if utilities.sum() == 0:
            utilities = np.ones_like(utilities, dtype=float)
        utilities /= utilities.sum()

        
        if Wtime == 1:
            idx_by_mode = {m: i for i, m in enumerate(modes)}
            walking_idx = idx_by_mode.get("Walking")

            faster_modes = ["Public bus", "Car driver", "Car passenger"]
            faster_available = any(
                (idx_by_mode.get(m) is not None and utilities[idx_by_mode[m]] > 0)
                for m in faster_modes
            )

            if walking_idx is not None and faster_available and cumdist > 3.0:
                utilities[walking_idx] = 0.0
                if utilities.sum() == 0:
                    utilities = np.ones_like(utilities, dtype=float)
                else:
                    utilities /= utilities.sum()

     
      
      
        if Weco == 1:
            idx_cd = idx_cp = idx_bus = idx_bike = idx_walk = None
            eco_idxs = []

            for i, m in enumerate(modes):
                if m == "Car driver":
                    idx_cd = i
                elif m == "Car passenger":
                    idx_cp = i
                elif m == "Public bus":
                    idx_bus = i
                elif m == "Bicycle":
                    idx_bike = i
                elif m == "Walking":
                    idx_walk = i

                if m in ["Walking", "Bicycle", "Public bus"]:
                    eco_idxs.append(i)

          
            if eco_idxs and float(np.sum(utilities[eco_idxs])) <= 0 and idx_bus is not None:
                utilities[idx_bus] = max(utilities[idx_bus], 0.05)

        
            CAP_CAR_DRIVER = 0.10
            CAP_CAR_PASSENGER = 0.25

            remove = 0.0
            if idx_cd is not None and utilities[idx_cd] > CAP_CAR_DRIVER:
                remove += (utilities[idx_cd] - CAP_CAR_DRIVER)
                utilities[idx_cd] = CAP_CAR_DRIVER

            if idx_cp is not None and utilities[idx_cp] > CAP_CAR_PASSENGER:
                remove += (utilities[idx_cp] - CAP_CAR_PASSENGER)
                utilities[idx_cp] = CAP_CAR_PASSENGER

      
            if remove > 0 and eco_idxs:
                eco_sum = float(np.sum(utilities[eco_idxs]))
                if eco_sum <= 0:
                    if idx_bus is not None:
                        utilities[idx_bus] += remove
                else:
                    for i in eco_idxs:
                        utilities[i] += remove * (utilities[i] / eco_sum)

      
            if utilities.sum() == 0:
                utilities = np.ones_like(utilities, dtype=float)
            utilities /= utilities.sum()

         
            def ensure_geq(a_idx, b_idx):
                if a_idx is None or b_idx is None:
                    return
                if utilities[a_idx] <= 0 or utilities[b_idx] <= 0:
                    return
                if utilities[a_idx] < utilities[b_idx]:
                    utilities[a_idx], utilities[b_idx] = utilities[b_idx], utilities[a_idx]

            ensure_geq(idx_bike, idx_bus)
            ensure_geq(idx_bus, idx_cp)
            ensure_geq(idx_cp, idx_cd)

          
            if utilities.sum() == 0:
                utilities = np.ones_like(utilities, dtype=float)
            utilities /= utilities.sum()

     
        ranked_indices = list(np.argsort(utilities)[::-1])
        order = ranked_indices[:3]

        prediction = modes[order[0]]


        if Weco == 1 and prediction == "Car driver":
            for idx in order[1:]:
                if modes[idx] != "Car driver":
                    prediction = modes[idx]
                    break

        
        base_max = proba.max()
        local_factors = []
        n_features = X_row.shape[1]

        for i in range(n_features):
            Xp = X_row.copy()
            Xp[0, i] += 0.05 * (abs(Xp[0, i]) + 1e-3)
            new_max = model.predict_proba(Xp)[0].max()
            impact = abs(new_max - base_max)
            if impact > 0:
                local_factors.append({
                    "feature": model_features[i],
                    "importance": round(impact * 100, 2)
                })

        local_factors = sorted(local_factors, key=lambda x: x["importance"], reverse=True)[:6]

        co2 = {
            "Walking": round(EMISSION_FACTORS_G_PER_KM["Walking"] * cumdist, 1),
            "Cycling": round(EMISSION_FACTORS_G_PER_KM["Cycling"] * cumdist, 1),
            "Public Bus": round(EMISSION_FACTORS_G_PER_KM["Public Bus"] * cumdist, 1),
            "Car Passenger": round(EMISSION_FACTORS_G_PER_KM["Car Passenger"] * cumdist, 1),
            "Car Driver": round(EMISSION_FACTORS_G_PER_KM["Car Driver"] * cumdist, 1),
        }

        return jsonify({
            "prediction": prediction,
            "confidence": {
                "labels": [modes[i] for i in order],
                "probabilities": (utilities[order] * 100).round(1).tolist()
            },
            "distance_km": round(cumdist, 2),
            "duration_min": round(duration, 1),
            "factors": local_factors,
            "co2": co2,
            "debug_flags": {"eco": eco, "timesensitive": time_pref, "car": car, "bike": bike}
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


#  RUN

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
