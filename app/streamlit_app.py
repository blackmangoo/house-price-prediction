import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# Premium UI Configuration (Real Estate Theme)
# ==========================================
st.set_page_config(
    page_title="Luxe Estate | AI Pricing",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Real Estate Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant Serif Headers */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #f8fafc;
    }

    /* Premium Header */
    .premium-header {
        text-align: center;
        padding: 40px 0 20px 0;
    }
    .premium-header h1 {
        font-size: 3.5rem;
        margin-bottom: 0px;
        color: #ffffff;
    }
    .premium-header p {
        font-size: 1.2rem;
        color: #94a3b8;
        font-weight: 300;
        margin-top: -10px;
    }

    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 16px;
        padding: 40px 30px;
        text-align: center;
        color: white;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.15);
        margin: 20px 0 40px 0;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(45deg, transparent 40%, rgba(255,215,0,0.1) 50%, transparent 60%);
        background-size: 200% 200%;
        animation: shimmer 3s infinite linear;
    }

    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    .price-value {
        font-family: 'Playfair Display', serif;
        font-size: 4.5rem;
        font-weight: 600;
        background: linear-gradient(to right, #d4af37, #fdf4cd, #d4af37);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }

    /* Button */
    .stButton>button {
        background: linear-gradient(to right, #d4af37, #b8860b);
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #fdf4cd, #d4af37);
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.4);
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Main Layout
# ==========================================
st.markdown("""
<div class="premium-header">
    <h1>Luxe Estate</h1>
    <p>AI-Powered Valuation Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Path to the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_house_model.pkl')

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.error("⚠️ **Model Not Found!**")
    st.info("Please run the `notebooks/Colab_BigData_Training.ipynb` in Google Colab first to train the Big Data model. Once it downloads `xgboost_house_model.pkl`, place it in the `models/` directory!")
    st.stop()

# ==========================================
# Input Form
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📈 Demographics")
    med_inc = st.number_input("Median Income (in $10,000s)", min_value=0.5, max_value=15.0, value=3.5, step=0.1)
    population = st.number_input("Block Population", min_value=10, max_value=35000, value=1400, step=100)

with col2:
    st.markdown("### 🏡 Property Specs")
    house_age = st.slider("House Age (Years)", min_value=1, max_value=60, value=25)
    ave_rooms = st.number_input("Avg Rooms per Household", min_value=1.0, max_value=20.0, value=5.5, step=0.5)
    ave_bedrms = st.number_input("Avg Bedrooms", min_value=0.5, max_value=10.0, value=1.1, step=0.1)
    ave_occup = st.number_input("Avg Occupancy", min_value=1.0, max_value=10.0, value=3.0, step=0.1)

with col3:
    st.markdown("### 📍 Location")
    # California bounds
    latitude = st.slider("Latitude", min_value=32.0, max_value=42.0, value=35.6)
    longitude = st.slider("Longitude", min_value=-124.3, max_value=-114.3, value=-119.5)


# ==========================================
# Prediction Logic
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
if st.button("CALCULATE VALUATION"):
    
    # Construct DataFrame (must match the order of fetch_california_housing)
    # Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
    input_data = pd.DataFrame([[
        med_inc, house_age, ave_rooms, ave_bedrms, 
        population, ave_occup, latitude, longitude
    ]], columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])
    
    with st.spinner("Analyzing market data..."):
        # The model returns the median house value in hundreds of thousands of dollars ($100,000)
        prediction = model.predict(input_data)[0]
        actual_price = prediction * 100000
        
        # Format the price
        formatted_price = f"${actual_price:,.0f}"
        
        st.markdown(f"""
        <div class="prediction-card">
            <h3 style="color: #94a3b8; font-family: 'Inter', sans-serif; font-size: 1.2rem; font-weight: 400; margin: 0;">Estimated Property Value</h3>
            <div class="price-value">{formatted_price}</div>
            <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0;">Based on {house_age}-year-old properties in selected geo-coordinates</p>
        </div>
        """, unsafe_allow_html=True)
