import numpy as np
import pandas as pd
import joblib
import streamlit as st

Model = joblib.load("Model.pkl")
Inputs = joblib.load("Inputs.pkl")

def Make_Prdiction(Airline, day, month, Source, Destination, Dep_DayPart,DurationInMinutes, StopsCount, Additional_Info):
    Pr_df = pd.DataFrame(columns=Inputs)
    Pr_df.at[0,"Airline"] = Airline
    Pr_df.at[0,"day"] = day
    Pr_df.at[0,"month"] = month
    Pr_df.at[0,"Source"] = Source
    Pr_df.at[0,"Destination"] = Destination
    Pr_df.at[0,"Dep_DayPart"] = Dep_DayPart
    Pr_df.at[0,"DurationInMinutes"] = DurationInMinutes
    Pr_df.at[0,"StopsCount"] = StopsCount
    Pr_df.at[0,"Additional_Info"] = Additional_Info
    result = Model.predict(Pr_df)
    return result[0]
    
def main():
    st.title("Airline Price Prediction")
    Airline= st.selectbox("Airline",['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet',
       'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia',
       'Vistara Premium economy', 'Jet Airways Business',
       'Multiple carriers Premium economy', 'Trujet']) 
    day = st.selectbox("Day of Journey" , [1, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    month = st.selectbox("Month of Journey" , ['March', 'April', 'May', 'June'])
    Source = st.selectbox("Source" ,['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'] )
    Destination = st.selectbox("Destination" ,['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad'])
    Dep_DayPart= st.selectbox("Departure part of day",['Early Morning', 'Morning', 'Noon', 'Evening', 'Night'])
    DurationInMinutes = st.slider("Duration In Minutes", min_value=60, max_value=3000, value=60, step=1)
    StopsCount = st.selectbox("Number of stops", [0, 1, 2, 3, 4])
    Additional_Info = st.selectbox("Additional_Info" ,['No info', 'In-flight meal not included',
       'No check-in baggage included', '1 Short layover',
       '1 Long layover', 'Change airports', 'Business class',
       'Red-eye flight', '2 Long layover'] )
    if st.button("Predict"):
        Results = Make_Prdiction(Airline, day, month, Source, Destination, Dep_DayPart,DurationInMinutes, StopsCount, Additional_Info)
        st.write('Price:', Results)
             
main()
