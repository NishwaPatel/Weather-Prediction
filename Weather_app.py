import streamlit as st
import pandas as pd
import pickle

st.header("Weather Prediction")

# Load trained model from pickle file
df = pickle.load(open("df.pkl", "rb"))
#input_data = pickle.load(open("input_data.pkl", "rb"))
#new_data = pickle.load(open("new_data.pkl", "rb"))
## not nessasry to load this file(input_data,new_data),don't make this pkl file..
ridge_model = pickle.load(open("ridge_model.pkl", "rb"))
prediction = pickle.load(open("prediction.pkl", "rb"))

#input filed for users
tempmax = st.text_input('Maximum Temperature')
tempmin = st.text_input('Minimum Temperature')
temp = st.text_input('temp')
feelslikemax = st.text_input('feelslikemax')
feelslikemin = st.text_input('feelslikemin')
feelslike = st.text_input('feelslike')
dew = st.text_input('dew')
humidity = st.text_input('humidity')
precip = st.text_input('precip')
precipprob = st.text_input('precipprob')
precipcover = st.text_input('precipcover')
snow = st.text_input('snow')
snowdepth = st.text_input('snowdepth')
windgust = st.text_input('windgust')
windspeed = st.text_input('windspped')
winddir = st.text_input('winddir')
sealevelpressure = st.text_input('sealevelpressure')
cloudcover = st.text_input('cloudcover')
visibility = st.text_input('visibility')
solarradiation = st.text_input('solarradiation')
solarenergy = st.text_input('solarenergy')
uvindex = st.text_input('uvindex')
severerisk = st.text_input('severerisk')
moonphase= st.text_input('moonphase')

##  Predict on this data
if st.button('Predict'):
    input_data = {
    'tempmax': [tempmax],
    'tempmin': [tempmin],
    'temp': [temp],
    'feelslikemax': [feelslikemax],
    'feelslikemin': [feelslikemin],
    'feelslike': [feelslike],
    'dew': [dew],
    'humidity': [humidity],
    'precip': [precip],
    'precipprob': [precipprob],
    'precipcover': [precipcover],
    'snow': [snow],
    'snowdepth': [snowdepth],
    'windgust': [windgust],
    'windspeed': [windspeed],
    'winddir': [winddir],
    'sealevelpressure': [sealevelpressure],
    'cloudcover': [cloudcover],
    'visibility': [visibility],
    'solarradiation': [solarradiation],
    'solarenergy': [solarenergy],
    'uvindex': [uvindex],
    'severerisk': [severerisk],
    'moonphase': [moonphase]
    }

    new_data_for_predic = pd.DataFrame(input_data, index=[0])

    #make prediction
    predicted_temperature = ridge_model.predict(new_data_for_predic)

    #predict temperature
    st.write('Predicted Temperature:', predicted_temperature)






