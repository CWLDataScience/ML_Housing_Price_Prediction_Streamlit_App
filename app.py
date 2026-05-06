import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
from custom_transformers import *

# ---- Load model ----
model = joblib.load("housing_model.pkl")

# ---- Page config ----
st.set_page_config(page_title="Housing Predictor", layout="centered")
#load data
@st.cache_data
def load_data():
    return pd.read_csv("housing.csv")
df = load_data()

# ---- Header ----




# ---house vs price scatter plot ---
st.title("California Housing Price Predictor")
st.markdown("Explore housing prices and make predictions")
st.subheader(" Price vs Income")
st.scatter_chart(
    df,
    x="median_income",
    y="median_house_value",
    size=20,
)
# ---price distribution histogram ---
import altair as alt

st.subheader("📈 Price Distribution")

chart = alt.Chart(df).mark_bar().encode(
    alt.X("median_house_value:Q", bin=alt.Bin(maxbins=50), title="House Price"),
    alt.Y("count()", title="Frequency")
)

st.altair_chart(chart, use_container_width=True)
# ---price by ocean proximity bar chart ---
st.subheader("🌊 Price by Ocean Proximity")

st.bar_chart(
    df.groupby("ocean_proximity")["median_house_value"].mean()
)










# ---- MAP SECTION (PyDeck improved) ----
st.subheader("🗺️ Housing Price Map")




# sample for performance
df_sample = df.sample(n=3000, random_state=42).copy()

# ---- Normalize price for color ----
price_min = df_sample["median_house_value"].min()
price_max = df_sample["median_house_value"].max()

df_sample["price_scaled"] = (
    (df_sample["median_house_value"] - price_min) /
    (price_max - price_min)
)

# ---- Color mapping (blue → red gradient) ----
df_sample["color_r"] = (df_sample["price_scaled"] * 255).astype(int)
df_sample["color_b"] = (255 - df_sample["color_r"]).astype(int)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_sample,
    get_position='[longitude, latitude]',
    get_color='[color_r, 50, color_b, 160]',  # dynamic color
    get_radius=200,
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=33.91,   # auto center
    longitude=-118,
    zoom=8.5,
    pitch=0,
)
st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
tooltip={
    "html": """
    <b>Price:</b> ${median_house_value}<br/>
    <b>Latitude:</b> {latitude}<br/>
    <b>Longitude:</b> {longitude}<br/>
    <b>Income:</b> {median_income}<br/>
    <b>Age:</b> {housing_median_age}<br/>
    <b>Rooms:</b> {total_rooms}<br/>
    <b>Bedrooms:</b> {total_bedrooms}<br/>
    <b>Population:</b> {population}<br/>
    <b>Households:</b> {households}<br/>
    <b>Ocean:</b> {ocean_proximity}
    """,
    "style": {"backgroundColor": "black", "color": "white"}
}
))
st.markdown("### Price Legend:")

st.markdown("""
<div style="display: flex; align-items: center;">
    <span style="margin-right: 10px;">Low</span>
    <div style="
        width: 200px;
        height: 20px;
        background: linear-gradient(to right, blue, red);
        border-radius: 5px;
    "></div>
    <span style="margin-left: 10px;">High</span>
</div>
""", unsafe_allow_html=True)

# ---- INPUT SECTION ----
st.subheader("📥 Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", value=-122.23)
    latitude = st.number_input("Latitude", value=37.88)
    median_house_age = st.slider("House Age", 1, 52, 20)
    total_rooms = st.number_input("Total Rooms", value=2000)
    total_bedrooms = st.number_input("Total Bedrooms", value=400)

with col2:
    population = st.number_input("Population", value=1000)
    households = st.number_input("Households", value=300)
    median_income = st.number_input("Median Income (10k)", value=3.5)

    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"]
    )

st.divider()

# ---- PREDICTION ----
if st.button("📈 Predict Price"):

    input_data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "median_house_age": median_house_age,
        "total_rooms_block": total_rooms,
        "total_bedrooms_block": total_bedrooms,
        "population_block": population,
        "num_households": households,
        "median_income_10k": median_income,
        "ocean_proximity": ocean_proximity
    }])

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Price: ${prediction:,.0f}")

    except Exception as e:
        st.error("⚠️ Something broke. Likely input mismatch.")
        st.text(str(e))
