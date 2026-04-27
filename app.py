import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- Page Config ---
st.set_page_config(page_title="Travel Price Predictor", page_icon="🌍", layout="centered")

st.title("🌍 Travel Cost Predictor")
st.markdown("Get a high-precision accommodation price estimate based on historical travel data.")

# --- Step 1: Load and Clean Data ---
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('Travel details dataset.csv')
    
    # 1. CLEAN CURRENCY (Prevents the 'dtype' TypeError on Streamlit Cloud)
    for col in ['Accommodation cost', 'Transportation cost']:
        if col in df.columns:
            # Force to string, strip non-numeric, then force to float
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where target is missing
    df.dropna(subset=['Accommodation cost'], inplace=True)
    
    # Fill categorical NAs
    df['Destination'] = df['Destination'].fillna('Unknown')
    df['Accommodation type'] = df['Accommodation type'].fillna('Unknown')
    df['Transportation type'] = df['Transportation type'].fillna('Unknown')
    
    # Fill numeric NAs with median
    df = df.fillna(df.median(numeric_only=True))

    # 2. FEATURE ENGINEERING
    # Super_Group for deep correlation
    df['Super_Group'] = (df['Destination'].astype(str) + "_" + 
                         df['Accommodation type'].astype(str) + "_" + 
                         df['Duration (days)'].astype(str))
    
    df['Historical_Cost_Map'] = df.groupby('Super_Group')['Accommodation cost'].transform('mean')
    df['Transport_Impact'] = df.groupby('Transportation type')['Accommodation cost'].transform('mean')
    
    # Efficient Age Binning
    df['Age_Bin'] = pd.cut(df['Traveler age'], bins=[0, 20, 40, 60, 100], labels=[1.0, 2.0, 3.0, 4.0]).astype(float)
    
    return df

df = load_and_clean_data()

# --- Step 2: Model Training ---
@st.cache_resource
def train_model(data):
    features = ['Historical_Cost_Map', 'Transportation cost', 'Transport_Impact', 'Duration (days)', 'Age_Bin']
    X = data[features].fillna(data[features].median())
    y = data['Accommodation cost']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Hyper-parameters tuned for high accuracy
    model = GradientBoostingRegressor(n_estimators=1200, learning_rate=0.04, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    score = r2_score(y_test, model.predict(X_test))
    return model, score

model, model_score = train_model(df)

# --- Step 3: User Input Form ---
with st.form("input_form"):
    st.subheader("Trip Configuration")
    col1, col2 = st.columns(2)
    with col1:
        input_dest = st.selectbox("Destination", sorted(df['Destination'].unique()))
        input_acc = st.selectbox("Accommodation Type", sorted(df['Accommodation type'].unique()))
        input_dur = st.number_input("Duration (Days)", min_value=1, max_value=365, value=7)
    with col2:
        input_age = st.number_input("Traveler Age", min_value=1, max_value=110, value=30)
        input_trans_type = st.selectbox("Transportation Type", sorted(df['Transportation type'].unique()))
        input_trans_cost = st.number_input("Transportation Cost ($)", min_value=0, value=500)
    
    submit = st.form_submit_button("Generate Prediction", use_container_width=True)

# --- Step 4: Prediction Execution ---
EXCHANGE_RATE = 83.50 

if submit:
    # 1. Determine Historical Value with Multi-Level Fallback
    user_group = f"{input_dest}_{input_acc}_{input_dur}"
    
    if user_group in df['Super_Group'].values:
        hist_val = df[df['Super_Group'] == user_group]['Accommodation cost'].mean()
    else:
        # Fallback 1: By Destination and Style
        fallback_mask = (df['Destination'] == input_dest) & (df['Accommodation type'] == input_acc)
        if fallback_mask.any():
            hist_val = df[fallback_mask]['Accommodation cost'].mean()
        else:
            # Fallback 2: General Destination Average
            dest_mask = df['Destination'] == input_dest
            if dest_mask.any():
                hist_val = df[dest_mask]['Accommodation cost'].mean()
            else:
                # Fallback 3: Global Dataset Average (Safety net)
                hist_val = df['Accommodation cost'].mean()
        
    # 2. Determine Transport Impact
    trans_impact = df[df['Transportation type'] == input_trans_type]['Accommodation cost'].mean()
    
    # 3. Numeric Age Binning Logic
    u_age_bin = float(np.digitize(input_age, [20, 40, 60]) + 1)
    
    # 4. Create Input for Model
    input_df = pd.DataFrame([[hist_val, input_trans_cost, trans_impact, input_dur, u_age_bin]], 
                            columns=['Historical_Cost_Map', 'Transportation cost', 'Transport_Impact', 'Duration (days)', 'Age_Bin'])
    
    # 5. Execute Prediction
    pred_usd = max(0, model.predict(input_df)[0]) # Ensure no negative prices
    pred_inr = pred_usd * EXCHANGE_RATE
    
    # --- UI Output ---
    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Predicted Stay (USD)", f"${pred_usd:,.2f}")
    m2.metric("Predicted Stay (INR)", f"₹{pred_inr:,.2f}")
    
    st.success(f"Trip Summary: {input_dur} days in {input_dest} ({input_acc})")
    
