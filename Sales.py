#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports
import streamlit as st
import pandas as pd
import pickle

# setting the basic configuration of the web app. This is shown in the Tab
st.set_page_config(page_title = "Qwicka Kleen - Sales Prediction" 
                    ,page_icon = ":bar_chart:" 
                    )


# Opening intro text
st.write("# Predict Monthly Sales✨")

st.write("### Specify your different scenarios 🎛️:")

# Price of the product
customer = st.slider('How many customers per month?💲', min_value=50, max_value=1000, value=100, step=50)

# Advertisment budget
orders = st.slider('How many orders per month?💰', min_value=10, max_value=2000, value=200, step=10)


# Promotions
quantity = st.slider('What quantity of items do you expect each month?💰', min_value=100, max_value=5000, value=500, step = 20)


# Creating the dataframe to run predictions on
row = [customer, orders, quantity]
columns = ['Customers', 'Orders', 'Quantity']

sales_scenario = pd.DataFrame(dict(zip(columns, row)), index=[0])

# Show the table?
# st.table(mktg_scenario)

# Now predicting!
if st.button(label="Click to predict monthly sales"):

    # Load the model
    from sklearn.linear_model import LinearRegression
    
#     loaded_model = LinearRegression()
    
    loaded_model = pickle.load(open('lr_model_prediction.sav','rb'))
    
    # Make predictions (and get out pred probabilities)
    pred = loaded_model.predict(sales_scenario)[0]
    
    st.write(f"Your Predicted Monthly Sales is 📊: {pred:,.0f} GBP ")


