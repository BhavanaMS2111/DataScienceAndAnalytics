import streamlit as st
import pickle
import numpy as np

# Set page configuration FIRST
st.set_page_config(page_title="Gold Price Prediction App", page_icon="💰", layout="centered")

# Load your model and other necessary code below
try:
    model = pickle.load(open("gold_price_predictor.pkl", "rb"))
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load the model: {e}")
    st.stop()

st.title("Gold Price Prediction App")
st.write("Enter USD to INR conversion rate to predict the closing gold price")

# Input and prediction
usd_to_inr = st.number_input("USD to INR Conversion Rate", min_value=50.0, max_value=200.0, step=0.01)

if st.button("Predict"):
    try:
        # Make prediction
        predicted_price_inr = model.predict(np.array([[usd_to_inr]]))
        predicted_price_usd = predicted_price_inr / usd_to_inr

        # Display the predictions
        st.success(
            f"### Predicted Closing Price of Gold: \n"
            f"💰 INR {predicted_price_inr[0]:,.2f} \n"
            f"💲 USD {predicted_price_usd[0]:,.2f}"
        )
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# Additional Information
st.markdown("---")
st.write(
    "### How the Model Works 📊"
    "- The model was trained on historical gold prices and uses linear regression to predict closing prices.\n"
    "- It considers the **USD to INR conversion rate** as the independent variable.\n"
)

# Display Model Summary
st.write(
    "### Model Information 📈"
    "- **Algorithm Used:** Linear Regression\n"
    "- **Evaluation Metric:** Mean Squared Error (MSE) and R-squared\n"
    "- **Data Range:** Historical data from 2021 to 2024"
)

# Footer with developer information
st.markdown("---")
st.write(
    "#### Developed by Bhavana \n"
    "For any queries or feedback, please contact: bhavana2111@gmail.com"
)

#AIzaSyDU5gJy8s5VZ0WBhmlwP9DDaw4tNRuNhmE
