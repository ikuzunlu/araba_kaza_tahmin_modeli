import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="UK Accident Severity Prediction", layout="wide")

st.title("🚗 Accident Severity Prediction")
st.caption("Model: Seçilebilir (.pkl) + Threshold karar kuralı")

BASE_DIR = Path(__file__).resolve().parent

# ---- Helpers ----
LABELS = {0: "Slight", 1: "Serious", 2: "Fatal"}

def apply_threshold(probs, fatal_thr=0.60, serious_thr=0.50):
    if probs[2] > fatal_thr:
        return 2
    elif probs[1] > serious_thr:
        return 1
    else:
        return 0

def list_pkl_files(folder: Path):
    return sorted([p.name for p in folder.glob("*.pkl")])

@st.cache_resource
def load_artifacts(model_file: str, feature_file: str):
    model_path = BASE_DIR / model_file
    feat_path = BASE_DIR / feature_file
    model = joblib.load(model_path)
    feature_cols = joblib.load(feat_path)
    return model, feature_cols

def set_onehot(values: dict, cols: list, prefix: str, chosen_suffix: str):
    """prefix'e uyan tüm one-hot kolonları 0 yap, seçileni 1 yap."""
    for c in cols:
        if c.startswith(prefix):
            values[c] = 1 if c == f"{prefix}{chosen_suffix}" else 0

def safe_has(cols, name):
    return name in cols

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Ayarlar")

    pkl_files = list_pkl_files(BASE_DIR)
    if len(pkl_files) == 0:
        st.error("Klasörde hiç .pkl dosyası yok. Model dosyanı bu klasöre koy.")
        st.stop()

    default_model = "final_xgb.pkl" if "final_xgb.pkl" in pkl_files else pkl_files[0]
    model_file = st.selectbox("Model dosyası (.pkl)", pkl_files, index=pkl_files.index(default_model))

    feature_file = st.selectbox(
        "Feature listesi (.pkl)",
        pkl_files,
        index=pkl_files.index("feature_columns.pkl") if "feature_columns.pkl" in pkl_files else 0
    )

    st.divider()
    st.subheader("🎚️ Thresholds")
    fatal_thr = st.slider("Fatal threshold", 0.00, 1.00, 0.60, 0.01)
    serious_thr = st.slider("Serious threshold", 0.00, 1.00, 0.50, 0.01)

    st.divider()
    st.caption("İpucu: Yeni yöntemde ürettiğin modelin .pkl dosyasını bu klasöre koy, listede görünsün.")

# ---- Load ----
try:
    model, feature_cols = load_artifacts(model_file, feature_file)
except FileNotFoundError as e:
    st.error(f"Dosya bulunamadı: {e}")
    st.stop()

cols = list(feature_cols)
values = {}

st.subheader("🧾 Girdi Özellikleri")

# -------------------------
# 1) ÜSTTE TOPLU/İLİŞKİLİ SEÇİMLER (tek seçim -> one-hot otomatik)
# -------------------------
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_ONEHOT_MAP = {
    "Mon": "Monday",
    "Tue": "Tuesday",
    "Wed": "Wednesday",
    "Thu": "Thursday",
    "Fri": None,        # Datasetinizde Day_Friday yoksa None kalsın
    "Sat": "Saturday",
    "Sun": "Sunday",
}

# Konum bilmeyenler için
with st.expander("📍 Konum bilgisi (Latitude/Longitude)"):
    unknown_latlon = st.checkbox("Latitude/Longitude bilmiyorum (0 yaz)", value=True)

st.markdown("### 🧩 Temel Zaman Bilgileri")
t1, t2, t3, t4 = st.columns(4)

with t1:
    if safe_has(cols, "Year"):
        values["Year"] = int(st.number_input("Year", min_value=2000, max_value=2030, value=2020, step=1))

with t2:
    if safe_has(cols, "Month"):
        mname = st.selectbox("Month", MONTH_NAMES, index=0)
        values["Month"] = int(MONTH_NAMES.index(mname) + 1)

with t3:
    if safe_has(cols, "Day"):
        values["Day"] = int(st.number_input("Day", min_value=1, max_value=31, value=15, step=1))

with t4:
    if safe_has(cols, "Hour"):
        values["Hour"] = int(st.slider("Hour", 0, 23, 12, 1))

# Weekday seçimi (ve Day_* one-hot otomatik)
if safe_has(cols, "Weekday"):
    st.markdown("### 🗓️ Gün Seçimi")
    wcol1, wcol2 = st.columns([1, 2])
    with wcol1:
        wname = st.selectbox("Weekday", WEEKDAY_NAMES, index=0)
        values["Weekday"] = int(WEEKDAY_NAMES.index(wname) + 1)  # istersen 0-6 yaparız
    with wcol2:
        st.caption("Not: Aşağıdaki Day_* checkbox'ları artık tekrar seçtirmiyor, otomatik dolduruluyor.")

    # Day_* one-hot kolonları varsa otomatik doldur
    chosen = DAY_ONEHOT_MAP.get(wname)
    day_prefix = "Day_"
    for c in cols:
        if c.startswith(day_prefix):
            suffix = c[len(day_prefix):]
            values[c] = 1 if (chosen is not None and suffix == chosen) else 0

# Season tek seçim -> Season_* otomatik
season_cols = [c for c in cols if c.startswith("Season_")]
if len(season_cols) > 0:
    st.markdown("### 🌦️ Mevsim")
    season_options = [c.replace("Season_", "") for c in season_cols]
    chosen_season = st.selectbox("Season", season_options, index=0)
    set_onehot(values, cols, "Season_", chosen_season)

# Road Type tek seçim -> Road_Type_* otomatik
road_cols = [c for c in cols if c.startswith("Road_Type_")]
veh_cols = [c for c in cols if c.startswith("Vehicle_Group_")]

if len(road_cols) > 0 or len(veh_cols) > 0:
    st.markdown("### 🛣️ Yol ve Araç Grubu")
    r1, r2 = st.columns(2)

    with r1:
        if len(road_cols) > 0:
            road_options = [c.replace("Road_Type_", "") for c in road_cols]
            chosen_road = st.selectbox("Road type", road_options, index=0)
            set_onehot(values, cols, "Road_Type_", chosen_road)

    with r2:
        if len(veh_cols) > 0:
            veh_options = [c.replace("Vehicle_Group_", "") for c in veh_cols]
            chosen_veh = st.selectbox("Main vehicle group", veh_options, index=0)
            set_onehot(values, cols, "Vehicle_Group_", chosen_veh)

# Junction Control tek seçim -> one-hot otomatik
jc_cols = [c for c in cols if c.startswith("Junction_Control_Grouped_")]
if len(jc_cols) > 0:
    st.markdown("### 🚦 Junction Control")
    jc_options = [c.replace("Junction_Control_Grouped_", "") for c in jc_cols]
    chosen_jc = st.selectbox("Junction control", jc_options, index=0)
    set_onehot(values, cols, "Junction_Control_Grouped_", chosen_jc)

# Junction Detail tek seçim -> one-hot otomatik
jd_cols = [c for c in cols if c.startswith("Junction_Detail_Grouped_")]
if len(jd_cols) > 0:
    st.markdown("### 🧭 Junction Detail")
    jd_options = [c.replace("Junction_Detail_Grouped_", "") for c in jd_cols]
    chosen_jd = st.selectbox("Junction detail", jd_options, index=0)
    set_onehot(values, cols, "Junction_Detail_Grouped_", chosen_jd)

# Speed Category tek seçim -> one-hot otomatik
sc_cols = [c for c in cols if c.startswith("Speed_Category_")]
if len(sc_cols) > 0:
    st.markdown("### 🏎️ Speed Category")
    sc_options = [c.replace("Speed_Category_", "") for c in sc_cols]
    chosen_sc = st.selectbox("Speed category", sc_options, index=0)
    set_onehot(values, cols, "Speed_Category_", chosen_sc)

# -------------------------
# 2) KALAN DİĞER FEATURE'LAR (tek tek)
# -------------------------
st.markdown("### 🧪 Diğer Özellikler")

skip_prefixes = (
    "Day_", "Season_", "Road_Type_", "Vehicle_Group_",
    "Junction_Control_Grouped_", "Junction_Detail_Grouped_", "Speed_Category_"
)

BOOL_PREFIXES = ("Is_",)

c1, c2, c3 = st.columns(3)

for i, c in enumerate(cols):
    if c.startswith(skip_prefixes):
        continue
    if c in values:
        continue

    target_col = [c1, c2, c3][i % 3]

    with target_col:
        # Latitude/Longitude özel
        if c in ["Latitude", "Longitude"] and unknown_latlon:
            values[c] = 0.0
            st.number_input(c, value=0.0, disabled=True)
            continue

        # Speed_limit varsa daha kolay seçim
        if c == "Speed_limit":
            values[c] = int(st.selectbox("Speed_limit", [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], index=3))
            continue

        # Urban_or_Rural_Area genelde 1/2 olur
        if c == "Urban_or_Rural_Area":
            opt = st.selectbox("Urban_or_Rural_Area", ["Urban (1)", "Rural (2)"])
            values[c] = 1 if opt.startswith("Urban") else 2
            continue

        # ✅ EKLENDİ: Bad_* değişkenleri 0/1 bayrak -> checkbox
        if c in ["Bad_Weather", "Bad_Road_Condition"]:
            values[c] = int(st.checkbox(c, value=False))
            continue

        # Is_* checkbox
        if c.startswith(BOOL_PREFIXES):
            values[c] = int(st.checkbox(c, value=False))
        else:
            values[c] = float(st.number_input(c, value=0.0, step=1.0))

# ---- Input DF ----
X_input = pd.DataFrame([values], columns=cols)

for c in X_input.columns:
    if X_input[c].dtype == bool:
        X_input[c] = X_input[c].astype(int)

st.divider()
predict_btn = st.button("🔮 Predict Severity", type="primary")

if predict_btn:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0]
        pred = apply_threshold(proba, fatal_thr=fatal_thr, serious_thr=serious_thr)

        left, right = st.columns([1, 1])
        with left:
            st.success(f"**Prediction:** {LABELS[pred]} (class {pred})")
            st.write("**Probabilities**")
            st.write({
                "Slight (0)": float(proba[0]),
                "Serious (1)": float(proba[1]),
                "Fatal (2)": float(proba[2]),
            })
            st.write("**Decision rule used**")
            st.code(
                f"if p_fatal > {fatal_thr:.2f}: Fatal\n"
                f"elif p_serious > {serious_thr:.2f}: Serious\n"
                f"else: Slight"
            )
        with right:
            prob_df = pd.DataFrame({"Class": ["Slight", "Serious", "Fatal"], "Probability": proba})
            st.bar_chart(prob_df.set_index("Class"))
    else:
        pred = int(model.predict(X_input)[0])
        st.success(f"**Prediction:** {LABELS.get(pred, pred)}")

    st.info("Not: Modelin doğru çalışması için feature sırası ve preprocessing birebir aynı olmalı.")
