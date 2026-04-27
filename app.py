import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- Page Config ---
st.set_page_config(page_title="Travel Price Predictor", page_icon="🌍", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- Step 1: Load and Clean Data ---
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('Travel details dataset.csv')
    
    # Cleaning Currency Columns
    for col in ['Accommodation cost', 'Transportation cost']:
        if col in df.columns:
            # Combined cleaning: force to string, remove non-numeric, convert to float
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
    
    # Basic Data Integrity
    df.dropna(subset=['Accommodation cost'], inplace=True)
    df[['Destination', 'Accommodation type', 'Transportation type']] = df[['Destination', 'Accommodation type', 'Transportation type']].fillna('Unknown')
    
    # Fill numeric NAs efficiently
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # --- Feature Engineering ---
    # Super_Group for granular tracking
    df['Super_Group'] = df['Destination'].astype(str) + "_" + df['Accommodation type'].astype(str) + "_" + df['Duration (days)'].astype(str)
    
    # Mean Target Encoding (The "Secret Sauce")
    df['Historical_Cost_Map'] = df.groupby('Super_Group')['Accommodation cost'].transform('mean')
    df['Transport_Impact'] = df.groupby('Transportation type')['Accommodation cost'].transform('mean')
    
    # Robust Age Binning
    df['Age_Bin'] = pd.cut(df['Traveler age'], bins=[0, 20, 40, 60, 100], labels=[1, 2, 3, 4]).astype(float)
    
    return df

df = load_and_clean_data()

# --- Step 2: Model Training ---
@st.cache_resource
def train_model(data):
    features = ['Historical_Cost_Map', 'Transportation cost', 'Transport_Impact', 'Duration (days)', 'Age_Bin']
    X = data[features].fillna(data[features].median())
    y = data['Accommodation cost']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Tuned Gradient Boosting
    model = GradientBoostingRegressor(n_estimators=1500, learning_rate=0.03, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    score = r2_score(y_test, model.predict(X_test))
    return model, score

model, model_score = train_model(df)

# --- Step 3: Sidebar & Inputs ---
st.title("🌍 AI Travel Cost Predictor")
st.info(f"Model precision is currently at **{model_score:.2%}** based on historical trends.")

with st.form("input_form"):
    st.subheader("Trip Configuration")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        input_dest = st.selectbox("Where to?", sorted(df['Destination'].unique()))
        input_acc = st.selectbox("Stay Type", sorted(df['Accommodation type'].unique()))
    with c2:
        input_dur = st.number_input("Duration (Days)", min_value=1, max_value=60, value=7)
        input_age = st.number_input("Your Age", min_value=1, max_value=100, value=25)
    with c3:
        input_trans_type = st.selectbox("Transport Method", sorted(df['Transportation type'].unique()))
        input_trans_cost = st.number_input("Transport Budget ($)", min_value=0, value=200)
    
    submit = st.form_submit_button("Calculate Estimated Cost", use_container_width=True)

# --- Step 4: Prediction ---
if submit:
    # 1. Multi-tier Fallback Logic
    user_group = f"{input_dest}_{input_acc}_{input_dur}"
    if user_group in df['Super_Group'].values:
        hist_val = df[df['Super_Group'] == user_group]['Accommodation cost'].mean()
    else:
        # Check by Destination and Style
        dest_style_match = df[(df['Destination'] == input_dest) & (df['Accommodation type'] == input_acc)]
        if not dest_style_match.empty:
            hist_val = dest_style_match['Accommodation cost'].mean()
        else:
            # Global Destination Mean
            hist_val = df[df['Destination'] == input_dest]['Accommodation cost'].mean() if input_dest in df['Destination'].values else df['Accommodation cost'].mean()

    # 2. Support Features
    trans_impact = df[df['Transportation type'] == input_trans_type]['Accommodation cost'].mean()
    u_age_bin = float(np.digitize(input_age, [20, 40, 60]) + 1)
    
    # 3. Model Prediction
    input_data = pd.DataFrame([[hist_val, input_trans_cost, trans_impact, input_dur, u_age_bin]], 
                            columns=['Historical_Cost_Map', 'Transportation cost', 'Transport_Impact', 'Duration (days)', 'Age_Bin'])
    
    prediction = max(0, model.predict(input_data)[0])
    
    # --- Results UI ---
    st.divider()
    res1, res2 = st.columns(2)
    
    res1.metric("Predicted Stay Cost (USD)", f"${prediction:,.2f}")
    res2.metric("Predicted Stay Cost (INR)", f"₹{prediction * 83.50:,.2f}")
    
    with st.expander("🔍 See Trip Breakdown"):
        st.write(f"**Destination:** {input_dest}")
        st.write(f"**Stay Level:** {input_acc}")
        st.write(f"**Daily Average:** ${prediction/input_dur:,.2f} per day")
        st.caption("Note: Predictions are based on historical averages and may vary by season.")
