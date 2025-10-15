import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load model and data ---
cb_model = pickle.load(open('CatBoost_Model.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# --- Screen resolution helper ---
def get_screen_res(text):
    if 'Full HD' in text:
        return 'Full HD'             # 1920x1080
    elif 'Quad HD+' in text:
        return 'Quad HD+'            # 2560x1440
    elif '4K Ultra HD' in text:
        return '4K Ultra HD'         # 3840x2160
    else:
        return 'HD'                  # default 1366x768

res_to_pixels = {
    'HD': (1366, 768),
    'Full HD': (1920, 1080),
    'Quad HD+': (2560, 1440),
    '4K Ultra HD': (3840, 2160)
}

# --- Streamlit title ---
st.title("💻 Laptop Price Predictor")

# --- Company ---
company = st.selectbox("Company", sorted(df['Company'].unique()))

# --- TypeName ---
typename = st.selectbox("Laptop Type", sorted(df['TypeName'].unique()))

# --- RAM ---
ram = st.selectbox("RAM (GB)", sorted(df['Ram'].unique()))

# --- Weight ---
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)

# --- IPS ---
has_ips = st.selectbox("IPS Display", ["Yes", "No"])
has_ips = 1 if has_ips == "Yes" else 0

# --- Touchscreen ---
has_touch = st.selectbox("Touchscreen", ["Yes", "No"])
has_touch = 1 if has_touch == "Yes" else 0

# --- Screen Resolution ---
screen_res_raw = st.selectbox(
    "Screen Resolution",
    sorted(df['Screen_Res'].apply(get_screen_res).unique())
)
width, height = res_to_pixels.get(screen_res_raw, (1920, 1080))

# --- Screen Size and PPI ---
screen_size = st.number_input("Screen Size (inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
ppi = ((width**2 + height**2)**0.5) / screen_size

# --- Operating System ---
os = st.selectbox("OS", sorted(df['OS'].unique()))

# --- CPU Brand and Clock ---
cpu_brand = st.selectbox("CPU Brand", sorted(df['CPU_Brand'].unique()))
cpu_clock = st.selectbox("CPU Clock Speed (GHz)", sorted(df['CPU_clock_speed'].unique()))

# --- GPU Type ---
gpu_type = st.selectbox("GPU Type", sorted(df['GPU_Type'].unique()))

# --- Storage ---
ssd = st.number_input("SSD (GB)", min_value=0, max_value=2000, value=256, step=128)
hdd = st.number_input("HDD (GB)", min_value=0, max_value=2000, value=512, step=128)
flash = st.number_input("Flash Storage (GB)", min_value=0, max_value=512, value=0, step=64)

categorical_cols = ['Company', 'TypeName', 'Screen_Res', 'OS', 'CPU_Brand', 'GPU_Type']

# --- Predict button ---
if st.button("Predict Price 💰"):
    # Prepare input as dataframe
    input_df = pd.DataFrame({
        'Company':[company],
        'TypeName':[typename],
        'Ram':[ram],
        'Weight':[weight],
        'Has_IPS':[has_ips],
        'Has_Touchscreen':[has_touch],
        'Screen_Res':[screen_res_raw],
        'OS':[os],
        'CPU_Brand':[cpu_brand],
        'CPU_clock_speed':[cpu_clock],
        'GPU_Type':[gpu_type],
        'PPI':[ppi],
        'SSD':[ssd],
        'HDD':[hdd],
        'Flash':[flash]
    })

    # Convert categorical columns to string
    for col in categorical_cols:
        input_df[col] = input_df[col].astype(str)

    # Ensure column order matches training
    model_columns = ['Company', 'TypeName', 'Ram', 'Weight', 'Has_IPS', 
                     'Has_Touchscreen', 'Screen_Res', 'OS', 'CPU_Brand', 
                     'CPU_clock_speed', 'GPU_Type', 'PPI', 'SSD', 'HDD', 'Flash']
    input_df = input_df[model_columns]

    # Predict (log-scale)
    y_pred_log = cb_model.predict(input_df)

    # Convert back from log1p
    price_in_inr = np.expm1(y_pred_log)[0]

    st.write(f"💵 Predicted Laptop Price: ₹{price_in_inr:,.0f}")
