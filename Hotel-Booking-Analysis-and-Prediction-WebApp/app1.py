import numpy as np
import pickle
import streamlit as st
import random

def booking_prediction(input_data):
    model = pickle.load(open(r"C:\Users\delll\Data Analysis (hotel booking)\trained_model.sav", 'rb'))
    
    # Make prediction using the loaded model
    prediction = model.predict(input_data.reshape(1, -1))  # Reshape input to 2D array for prediction
    
    return prediction.flatten()

def main():
    
    # Setting page title and background color
    st.set_page_config(page_title="Hotel Booking Predictor", page_icon=":bar_chart:", layout="centered", initial_sidebar_state="expanded")

    # Setting background color and padding for the main content
    st.title('🏨 Hotel Booking Predictor')
    st.markdown("---")
    
    st.header('Select Hotel Details:')
    col1,col2,col3=st.columns(3)

    # Input fields for categorical features
    
    with col1:
        hotel_options = ['Resort Hotel', 'City Hotel']
        hotel = st.selectbox('Hotel Type', hotel_options)
    
    with col2:
        meal_options = ['BB', 'HB', 'FB', 'SC']
        meal = st.selectbox('Meal Type', meal_options)
        
    
    with col3:
        market_segment_options = ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary', 'Groups']
        market_segment = st.selectbox('Market Segment', market_segment_options)
    with col1:
        distribution_channel_options = ['Direct', 'Corporate', 'TA/TO', 'Undefined']
        distribution_channel = st.selectbox('Distribution Channel', distribution_channel_options)
    with col2:
        reserved_room_type_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'P']
        reserved_room_type = st.selectbox('Reserved Room Type', reserved_room_type_options)
    with col3:
        deposit_type_options = ['No Deposit', 'Non Refund', 'Refundable']
        deposit_type = st.selectbox('Deposit Type', deposit_type_options)
    with col1:
        customer_type_options = ['Transient', 'Contract', 'Transient-Party', 'Group']
        customer_type = st.selectbox('Customer Type', customer_type_options)
    st.markdown("---")
   
    

    # Input fields for numerical features
    st.header('Enter Booking Details:')
    
    col1,col2,col3=st.columns(3)
    with col1:
        year_options = ['2015', '2016', '2017', '2018','2019','2020','2021','2022','2023','2024']
        year = st.selectbox('year', year_options)
    with col2:
        month_options = ['Jan', 'Feb', 'Mar', 'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        month = st.selectbox('month', month_options)
    with col3:
        day_options = ['1', '2', '3', '4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
        day = st.selectbox('day', day_options)
    with col1:
        lead_time = st.number_input('Lead Time (days)', min_value=0, value=0)
    with col2:
        arrival_week_number = st.number_input('Arrival Week Number', min_value=1, value=1)
    
    with col3:    
        arrival_day_of_month = st.number_input('Arrival Day of Month', min_value=1, value=1)
    
    with col1:
        weekend_nights = st.number_input('Stays in Weekend Nights', min_value=0, value=0)
    with col2:
        week_nights = st.number_input('Stays in Week Nights', min_value=0, value=0)
    with col3:
        adults = st.number_input('Number of Adults', min_value=1, value=1)
    with col1:
        children = st.number_input('Number of Children', min_value=0, value=0)
    with col2:
        babies = st.number_input('Number of Babies', min_value=0, value=0)

    with col3:
        previous_cancellations = st.number_input('Previous Cancellations', min_value=0, value=0)
    with col1:
        previous_bookings_not_canceled = st.number_input('Previous Bookings Not Canceled', min_value=0, value=0)
    with col2:
        agent = st.number_input('Agent ID', min_value=0, value=0)
    with col3:
        company = st.number_input('Company ID', min_value=0, value=0)
    with col1:
        adr = st.number_input('Average Daily Rate (EUR)', min_value=0.0, value=15.0, step=0.01)
    with col2:
        car_parking_spaces = st.number_input('Required Car Parking Spaces', min_value=0, value=0)
    with col3:
        special_requests = st.number_input('Number of Special Requests', min_value=0, value=0)
    with col1:
        is_repeated_guest = st.checkbox('Repeated Guest')
    st.markdown("---")
    
    # Combine categorical and numerical inputs
    input_data = np.array([hotel, lead_time, arrival_week_number,
       arrival_day_of_month,weekend_nights, week_nights,adults, children, babies, meal,
       market_segment, distribution_channel,1 if is_repeated_guest else 0,
       previous_cancellations, previous_bookings_not_canceled,
       reserved_room_type, deposit_type, agent, company,
       customer_type, adr, car_parking_spaces,
       special_requests, year, month, day],dtype=object).reshape(1,26)
    
    # code for Prediction
    result = ''
    # Display prediction result
    if st.button('Predict Booking'):
        prediction = booking_prediction(input_data)
        print(prediction)
        if prediction[0] == 0:
            result = 'The booking is likely to be successful.'
        else:
            result = 'The booking is likely to be canceled.'
    st.success(result)

if __name__ == '__main__':
    main()
