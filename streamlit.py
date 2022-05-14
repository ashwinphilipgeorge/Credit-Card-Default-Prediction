import streamlit as st
import joblib
import pandas as pd


st.title('Prediction of Credit Card Defaults')
st.text('This app was created by Ashwin Philip George for the module CET023.')
st.text('4 models are used in this app.')
 
GB_model = joblib.load("./models/gradient_boosting")
DT_model = joblib.load("./models/cart_model")
RFR_model = joblib.load("./models/random_forest")
LR_model = joblib.load("./models/linear_regression")

models = {'Gradient Boost Model': GB_model,
          'Decision Tree Model': DT_model,
          'Random Forest Model': RFR_model,
          'Logistic Regression Model': LR_model}

st.subheader("View Dataset here")
data_load_state = st.text('Loading data...')
data = pd.read_csv("./credit_card.csv")
data_load_state.text("Data has been loaded! Click the checkbox to see the data example.")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)



age = st.text_input('What is your age?')
income = st.text_input( 'What is your annual income? ($SGD)')
loan = st.text_input('How much do you intend to loan ($SGD)')



def predictor(model,age,income,loan):


    prediction = model.predict([[age,income,loan]])
    integer_value = int(prediction)
    if integer_value == 1:
        response = 'will default'
    else:
        response = 'will NOT default'
    return response

if st.button('Predict'):
    for key,item in models.items():
        answer = predictor(item,age,income,loan)
        st.write('The ' + key+ ' predicts that the user ' + answer + "on their credit card loans")