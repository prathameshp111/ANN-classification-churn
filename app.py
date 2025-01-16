import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
import pickle


# Load the trained model 
model = tf.keras.models.load_model('model.h5')

# load the encoderd

with open('Geo_encoder.pkl','rb') as file:
    geo_encoder = pickle.load(file)

with open('label_encode_gender.pkl','rb') as file:
    gen_encoder = pickle.load(file)

with open('Scalar.pkl','rb') as file:
    Scalar = pickle.load(file)

## Streamlit Appp

st.title('Customer churn Prediction')

# user input 
geography = st.selectbox('Geography',geo_encoder.categories_[0])
gender = st.selectbox('Gender',gen_encoder.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
Credit_score = st.number_input('CreditScore')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has cr Card',[0,1])
is_active_member = st.selectbox('Is active Member',[0,1])

# preparing input_data 

input_data = pd.DataFrame({
    'CreditScore':[Credit_score],
    "Gender": [gen_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]

})

# One hot encode Geography
geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=geo_encoder.get_feature_names_out())

#Combine ohe hot encoded columns with input data 
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_scaled_data = Scalar.transform(input_data)

#prediction
prediction = model.predict(input_scaled_data)
prediction_probability = prediction[0][0]

if prediction_probability > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')

