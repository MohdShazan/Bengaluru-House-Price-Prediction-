import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model and data
model = pickle.load(open('RidgeModel.pkl', 'rb'))
data = pd.read_csv('Cleaned_data.csv')

# Set up the page configuration
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for styling with background image
st.markdown(f"""
    <style>
        body {{
            background-image: url('https://images.wallpaperscraft.com/image/single/interior_style_design_83856_1600x900.jpg');
            background-size: cover;
            background-position: center;
            color: #ffffff; /* Ensures text is visible */
        }}
        .main-content {{
            background: rgba(0, 0, 0, 0.6); /* Semi-transparent overlay */
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }}
        h1, h2, h3 {{
            color: #ffcc00;
        }}
        .css-18e3th9 {{
            padding: 10px;
        }}
        .css-1d391kg {{
            color: #ffffff;
        }}
        .css-1d391kg > h3 {{
            color: #4CAF50;
        }}
    </style>
""", unsafe_allow_html=True)

# Main App Title
st.title("🏠 Bengaluru House Price Prediction App")
st.markdown("### Use this tool to predict house prices in Bengaluru effortlessly.")

# Sidebar inputs
st.sidebar.header("Input Features")
location = st.sidebar.selectbox("Select Location", sorted(data['location'].unique()))
total_sqft = st.sidebar.number_input("Total Area (sqft)", min_value=300.0, step=50.0)
bath = st.sidebar.slider("Number of Bathrooms", 1, 10, step=1)
bhk = st.sidebar.slider("Number of Bedrooms (BHK)", 1, 10, step=1)

# Process user input
def predict_price(location, sqft, bath, bhk):
    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = model.predict(input_data)[0]
    return np.round(prediction, 2)

# Predict button
if st.sidebar.button("Predict Price"):
    price = predict_price(location, total_sqft, bath, bhk)
    st.write(f"### Predicted House Price: ₹{price} Lacs")

# Main content for details
st.markdown("## 📊 About the Data")
st.markdown("""
The prediction model is built using a curated dataset of house prices in Bengaluru. 
The data includes essential features such as:
- **Location**: The locality where the house is situated.
- **Total Area**: Measured in square feet.
- **Bathrooms**: Number of bathrooms in the property.
- **BHK**: Number of bedrooms in the house.
""")

# Display a sample of the dataset
st.markdown("### 🔍 Sample Data")
st.dataframe(data.sample(5, random_state=42))  # Display a random sample for variety

# Additional Section: Benefits of the App
st.markdown("### 🌟 Why Use This App?")
st.markdown("""
- **Easy to Use**: Input your requirements and get instant predictions.
- **Accurate Model**: Built using advanced regression techniques for high precision.
- **Real Data**: Trained on actual Bengaluru housing data for realistic estimates.
""")

# Footer
st.markdown("---")
footer = """
    <div style='text-align: center; margin-top: 30px;'>
        <h4 style='color: #ffcc00;'>Bengaluru House Price Predictor</h4>
        <p style='font-size: 14px;'>© 2024 | Powered by <strong>Streamlit</strong> and <strong>Ridge Regression</strong></p>
        <p>Designed with 💡 to help you make informed real estate decisions.</p>
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)
