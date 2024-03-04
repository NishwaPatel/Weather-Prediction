import numpy as np
import pandas as pd

df = pd.read_csv("india.csv")
print(df)
print(df.info())

print(df.apply(pd.isnull).sum())

null_pct = df.apply(pd.isnull).sum()/df.shape[0]
print(null_pct)

valid_columns = df.columns[null_pct < .05]
print(valid_columns)
print(len(valid_columns))

print(df.info())

print(df)
print(df.isnull().sum())
print(df.dtypes)

df =pd.read_csv("india.csv" , index_col = "datetime")
print(df)

print(df.index)

import matplotlib.pyplot as plt
print(df["humidity"].plot())
plt.xticks(rotation=90)

print(df["temp"].plot())
plt.xticks(rotation=90)

print(df)

## start to prectict tomoroow's temprature ---
df["target"] = df.shift(-1)["tempmax"]
print(df)


print(df[['tempmax','target']])

print(df.info())

print(df.drop('preciptype',axis=1,inplace=True))
print(df.drop('stations',axis=1,inplace=True))
print(df)

df = df.ffill()
print(df)

print(df.corr(numeric_only=True))

## Traing Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd

# Assuming you have your DataFrame 'df' containing your features and target variable
# Extract features and target variable
X = df[['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin',
       'feelslike', 'dew', 'humidity', 'precip', 'precipprob', 'precipcover',
       'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir',
       'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation',
       'solarenergy', 'uvindex', 'severerisk', 'moonphase']]
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Ridge model
ridge_model = Ridge(alpha=10.0)  # You can adjust the alpha parameter as needed

# Train the Ridge model
ridge_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = ridge_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# Taking new data for prediction...all the input values
new_data =pd.DataFrame({
    'tempmax' : [22.3],
    'tempmin': [12.4],
    'temp' : [16.8],
    'feelslikemax':[22.3],
    'feelslikemin':[12.4],
       'feelslike':[16.8],
    'dew':[10.4],
    'humidity':[67.2],
    'precip':[13],
    'precipprob':[6.5],
    'precipcover':[37.5],
       'snow':[0],
    'snowdepth':[0],
    'windgust':[40.7],
    'windspeed':[25.9],
    'winddir':[68.8],
       'sealevelpressure':[1009],
    'cloudcover':[31.3],
    'visibility':[22.7],
    'solarradiation':[196.9],
       'solarenergy':[16.9],
    'uvindex':[7],
    'severerisk':[30],
    'moonphase':[0.42]
})


predicted_temperature = ridge_model.predict(new_data)
print("Predicted temperature for tomorrow:", predicted_temperature)

import pickle

with open('df.pkl', 'wb') as f:
    pickle.dump(df, f)
with open('df.pkl', 'rb') as f:
    loaded_df = pickle.load(f)

with open('ridge_model.pkl', 'wb') as f:
    pickle.dump(ridge_model, f)
with open('ridge_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('prediction.pkl', 'wb') as f:
    pickle.dump(y_pred, f)
with open('prediction.pkl', 'rb') as f:
    loaded_model_pred = pickle.load(f)