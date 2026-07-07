import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. Train the model once and remember it
@st.cache_resource
def train_model():
    data=fetch_california_housing(as_frame=True)
    df=data.frame
    X=df.drop("MedHouseVal", axis=1)
    y=df["MedHouseVal"]
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
    model=RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train,y_train)
    r2=r2_score(y_test,model.predict(X_test))
    return model,r2

model, r2=train_model()

# 2. Page Heading
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 California House Price Predictor")
st.write("Enter the detils of a housing district to predict its median house price.")
st.caption(f"Model: Random Forest | R2 Score: {r2:.2f}")
st.divider()

# 3. The 8 inputs + prediction
col1,col2=st.columns(2)
with col1:
    MedInc=st.slider("Median Income (in $10,000s)",0.5,15.0,3.5)
    HouseAge=st.slider("Median House Age", 1,52,25)
    AveRooms=st.slider("Average Rooms per Home: ", 1.0, 15.0, 5.0)
    AveBedrms=st.slider("Average Bedrooms per Home: ", 0.5, 5.0, 1.0)

with col2:
    Population = st.slider("Population", 3, 35000, 1500)
    AveOccup = st.slider("Average Occupancy", 1.0, 10.0, 3.0)
    Latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
    Longitude = st.slider("Longitude", -124.0, -114.0, -118.0)

if st.button("Predict Price", type="primary"):
    input_row=pd.DataFrame(
        [[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]],
        columns=["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                 "Population", "AveOccup", "Latitude", "Longitude"],
    )
    predicted=model.predict(input_row)[0]
    price_in_dollars=predicted * 100000

    st.divider()
    st.success(f"🏠 Estimated Median House Price: ${price_in_dollars:,.0f}")