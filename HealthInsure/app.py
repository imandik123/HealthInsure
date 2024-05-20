import os
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import plotly.graph_objects as go
from datetime import datetime
from streamlit.components.v1 import html

# Load the hybrid NIOA model
with open('optimized_ga_rf_model.pkl', 'rb') as f:
    optimized_model = pickle.load(f)

# Function to preprocess the user input
def preprocess_input(Age, AnyTransplants, NumberofSurgeries, chronicdiseases, AnyBloodPressureProblems, Weight, BMI):
    label_encoder = LabelEncoder()
    # Convert input values to appropriate types
    Age = int(Age)
    NumberofSurgeries = int(NumberofSurgeries)
    Weight = int(Weight)
    BMI = int(BMI)
    # Encode categorical columns
    transplants_encoded = label_encoder.fit_transform([AnyTransplants.lower()])[0]
    bpissues_encoded = label_encoder.fit_transform([AnyBloodPressureProblems.lower()])[0]
    chronicdiseases_encoded = label_encoder.fit_transform([chronicdiseases.lower()])[0]
    # Return preprocessed input as a numpy array
    return np.array([Age, transplants_encoded, NumberofSurgeries, chronicdiseases_encoded, bpissues_encoded, Weight, BMI]).reshape(1, -1)

# Function to predict premium using the Hybrid NIOA model
def predict_premium_optimized(input_data):
    prediction = optimized_model.predict(input_data)
    return prediction[0]

# Function to save user premium data history to CSV file 
def save_premium_data(data):
    # Add Timestamp to data
    data['Timestamp'] = datetime.now()
    # Convert data to DataFrame
    data_df = pd.DataFrame([data])
    # Save to CSV
    data_df.to_csv('user_premium_data.csv', mode='a', index=False, header=not os.path.isfile('user_premium_data.csv'))

# Function to load user premium data history from CSV
def load_premium_data():
    if os.path.isfile('user_premium_data.csv'):
        return pd.read_csv('user_premium_data.csv')
    else:
        return pd.DataFrame()

# Function to calculate BMI category
def calculate_bmi_category(age, bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 24.9:
        return 'Normal weight'
    elif 25 <= bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obese'

# Function to plot user premium history
def plot_premium_history(premium_data):
    st.write("Plotting premium history...")
    # Create a Plotly figure
    fig = go.Figure()
    # Add trace for optimized premium
    fig.add_trace(go.Scatter(x=premium_data['Timestamp'], y=premium_data['Predicted Premium'], mode='lines', name='Predicted Premium'))
    fig.update_layout(
        title='Premium History Over Time',
        xaxis_title='Timestamp',
        yaxis_title='Premium',
        hovermode='x',
        width=1500,  
        height=800,  
    )
    # Add play/pause buttons below the title
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="right", x=0.1, y=0.95,
                          buttons=[
                              dict(label="Play",
                                   method="animate",
                                   args=[None, {"frame": {"duration": 500, "redraw": True},
                                                "fromcurrent": True, "transition": {"duration": 300,
                                                                                     "easing": "quadratic-in-out"}}]),
                              dict(label="Pause",
                                   method="animate",
                                   args=[[None], {"frame": {"duration": 0, "redraw": True},
                                                 "mode": "immediate",
                                                 "transition": {"duration": 0}}])
                          ]
                        )]
    )
    # Add frames for animation (none in this case as we're only showing optimized premium)
    frames = []
    # Update frames
    fig.frames = frames
    # Show plot
    st.plotly_chart(fig)

# Streamlit app
def main():
    st.set_page_config(layout="wide")  # page layout set to wide to view application in full screen
    st.sidebar.title('HealthInsure')
    page = st.sidebar.radio(
        "Go to", 
        ("Predict", "Premium History"),
        help="Select 'Predict' to calculate your personalized health insurance premium or 'Premium History' to view your premium history"
    )
    # Add image banner
    st.image("HealthInsure.png", use_column_width=True) 

    # Custom CSS to increase the size of number input labels
    custom_css = """
    <style>
        /* Increase the size of number input labels */
        .stNumberInput > div > label {
            font-size: 50px !important; /* Increase font size */
        }
    </style>
    """
    # Display the custom CSS
    st.write(custom_css, unsafe_allow_html=True)

    if page == "Predict":
        st.title('Health Insurance Premium Prediction')
        st.subheader('Please enter below details to calculate your personalized health insurance premium')
        # User input section
        age = st.number_input('Age', min_value=1, max_value=100, step=1, key='age')
        transplants = st.selectbox('Have you undergone any transplants?', ('No', 'Yes'))
        noofsurgeries = st.number_input('Number of Surgeries that you have done', min_value=0, max_value=100, step=1, format='%d', key='noofsurgeries') # specify format to ensure consistency
        chronicdiseases = st.selectbox('Do you have any chronic diseases?', ('No', 'Yes'))
        bpissues = st.selectbox('Do you have any blood pressure problems ?', ('No', 'Yes'))
        weight = st.number_input('Weight(kg)', min_value=1, max_value=200, step=1, key='weight')
        bmi = st.number_input('BMI', min_value=1, max_value=200, step=1, key='bmi') 
        # Calculate BMI category
        bmi_category = calculate_bmi_category(age, bmi)
        # Display BMI category once user types BMI 
        st.write(f'BMI Category: {bmi_category}')
        # Preprocess input
        input_data = preprocess_input(age, transplants, noofsurgeries, chronicdiseases, bpissues, weight, bmi)
        # Predict premiums
        if st.button('Predict'):
            # Predict using the optimized model
            predicted_premium = predict_premium_optimized(input_data)
            predicted_premium = int(predicted_premium)
            # Display predicted premiums
            st.markdown(f'<span style="color:lightblue; font-size:28px; font-weight:bold; font-family:Times New Roman, Times, serif">Model Utilized : Hybrid NIOA Model (GA-RF)</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="color:lightblue; font-size:28px; font-weight:bold; font-family:Times New Roman, Times, serif">Premium Calculated (yearly) : {predicted_premium}</span>', unsafe_allow_html=True)

            # Store premium calculations
            premium_data = {
                'Age': age,
                'AnyTransplants': transplants,
                'Number of Surgeries': noofsurgeries,
                'Any Chronic Diseases': chronicdiseases,
                'Any Blood Pressure Problems': bpissues,
                'Weight': weight,
                'BMI': bmi,
                'BMI Category': bmi_category,
                'Predicted Premium': predicted_premium
            }
            # Create or append to the dataset
            save_premium_data(premium_data)

    elif page == "Premium History":
        st.title("User Premium History")
        premium_data = load_premium_data()
        if not premium_data.empty:
            st.write("Premium Data:")
            st.table(premium_data)
            plot_premium_history(premium_data)
        else:
            st.write("No premium data available.")

if __name__ == '__main__':
    main()