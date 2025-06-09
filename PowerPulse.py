import pandas as pd
import numpy as np
import os
import json
import mysql.connector as db
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.model_selection import GridSearchCV

# Initialize standard scalar and minmax scalar
mm = MinMaxScaler()
ss = StandardScaler()

#Define the file path
file_path = 'household_power_consumption.txt'
# Read the file into a DataFrame and combine data & time in one column
data = pd.read_csv(file_path, delimiter=';', parse_dates=[['Date', 'Time']])
# Rename the combined column
data.rename(columns={'Date_Time': 'DateTime'}, inplace=True)

#Get the rows and columns from Data
print(data.shape)

# Check what are the values are present in the active power column
data['Global_active_power'].value_counts()

#Check how many '?' values present in active power.
data[data['Global_active_power']=='?']

#Since all other columns are exist with same values hence we can remove those rows from the data frame
data = data[data['Global_active_power'] != '?']

#Convert columns to numeric
data['Global_active_power'] = pd.to_numeric(data['Global_active_power'])
data['Global_reactive_power'] = pd.to_numeric(data['Global_reactive_power'])
data['Voltage'] = pd.to_numeric(data['Voltage'])
data['Global_intensity'] = pd.to_numeric(data['Global_intensity'])
data['Sub_metering_1'] = pd.to_numeric(data['Sub_metering_1'])
data['Sub_metering_2'] = pd.to_numeric(data['Sub_metering_2'])
data['Sub_metering_3'] = pd.to_numeric(data['Sub_metering_3'])

# Round of to same format in all columns
data['Global_active_power'] = data['Global_active_power'].round(4)
data['Global_reactive_power'] = data['Global_reactive_power'].round(4)
data['Voltage'] = data['Voltage'].round(4)
data['Global_intensity'] = data['Global_intensity'].round(4)
data['Sub_metering_1'] = data['Sub_metering_1'].round(4)
data['Sub_metering_2'] = data['Sub_metering_2'].round(4)
data['Sub_metering_3'] = data['Sub_metering_3'].round(4)

#Format date time column
data['DateTime'] = pd.to_datetime(data['DateTime'], format='%Y-%m-%d %H:%M:%S')

#Split date month year in separate column
data['Day'] = data['DateTime'].dt.day
data['Month'] = data['DateTime'].dt.month
data['Year'] = data['DateTime'].dt.year
data['Hour'] = data['DateTime'].dt.hour
data['Minute'] = data['DateTime'].dt.minute

#Find the Avg power consumed for each Month and Year
daily_avg = data.groupby(['Year', 'Month']).agg({
    'Global_active_power': 'mean',
    'Global_reactive_power': 'mean',
    'Voltage': 'mean',
    'Global_intensity': 'mean'
}).sort_values(by=['Year', 'Month'], ascending=False).reset_index().rename(columns={
    'Global_active_power': 'Daily_avg_power',
    'Global_reactive_power': 'Daily_avg_reactive_power',
    'Voltage': 'Daily_avg_voltage',
    'Global_intensity': 'Daily_avg_intensity'
})

# Define peak hour
data['Is_peak_hour'] = data['Hour'].apply(lambda x: 1 if 17 <= x <= 21 else 0)
data['Is_peak_hour'].value_counts()

#Find the Avg power consumed during peak hour
PeakHr_Avg = data[data['Is_peak_hour']==1].groupby(['Month']).agg({
    'Global_active_power': 'mean',
    'Global_reactive_power': 'mean',
    'Voltage': 'mean',
    'Global_intensity': 'mean'
}).sort_values(by=['Month'], ascending=True).reset_index().rename(columns={
    'Global_active_power': 'Daily_avg_power',
    'Global_reactive_power': 'Daily_avg_reactive_power',
    'Voltage': 'Daily_avg_voltage',
    'Global_intensity': 'Daily_avg_intensity'
})


# Convert DateTime as index to find the mean of active power usage for every hours
data.set_index('DateTime', inplace=True)
hourly_usage = data.groupby(data.index.hour)['Global_active_power'].mean()



# Unexpected Usage During Off-Hours
pd.Series(data.index.hour, index=data.index)

data['Off_hour_usage'] = (
    pd.Series(data.index.hour, index=data.index).between(0, 5)
    & (data['Global_active_power'] > 0.5))


#Voltage should typically range from 220V–240V. Outliers may indicate faults.
data['anomaly_voltage'] = (data['Voltage'] < 220) | (data['Voltage'] > 250)

# Find Correlation between dataset
cr = data.drop('DateTime', axis=1).corr()

# Plot heatmap to view the correlations and remove unwanted columns based on correlation percentage
plt.figure(figsize=(14, 5))
sns.heatmap(cr, cmap= 'coolwarm', annot= True)
plt.show()

# Remove Global_intensity due to higly correlated with active power.
data.drop(['Global_intensity'],axis=1,inplace=True)

# Find minimum value in reactive power column
min_val = data['Global_reactive_power'].min()

# Add 0.0001 if value is zero in reactive power
if min_val <= 0:
    data['Global_reactive_power'] = data['Global_reactive_power'] + abs(min_val) + 1e-4
else:
    data['Global_reactive_power'] = data['Global_reactive_power']

# Apply boxcox to handle outliers in reactive power column
v, _ = stats.boxcox(data['Global_reactive_power'])
data['Global_reactive_power'] = v


# Standardize the value of reactive power column
data['Global_reactive_power'] = mm.fit_transform(data[['Global_reactive_power']])

# Apply boxcox to handle outliers in voltage column
v, _ = stats.boxcox(data['Voltage'])
data['Voltage'] = v

# Check it is normally distributed or not.
sns.kdeplot(data['Voltage'])

# Split val and target 
val = data.drop(['DateTime', "Global_active_power"], axis = 1)
tar = data['Global_active_power']


# Train and test split 
traindata, testdata, trainlab, testlab = train_test_split( val, tar, test_size= 0.20, random_state= 46 )

# Model Selection
model = LinearRegression()
model.fit(traindata, trainlab)

# prediction 
tr_pred = model.predict(traindata)
ts_pred = model.predict(testdata)

# Performance check
mean_squared_error( trainlab, tr_pred )
root_mean_squared_error( trainlab, tr_pred )

mean_squared_error( testlab, ts_pred )
root_mean_squared_error( testlab, ts_pred )

para = {
    "fit_intercept" : [True, False],
    "copy_X": [True, False],
    "n_jobs": [3,5,12,7],
    "positive": [True, False]
}

tune = GridSearchCV(LinearRegression(), param_grid= para, cv = 5)
tune.fit(traindata, trainlab)
tune.best_params_
best_model = tune.best_estimator_

tr_pred = best_model.predict(traindata)
ts_pred = best_model.predict(testdata)

# Performance check
mean_squared_error( trainlab, tr_pred )
root_mean_squared_error( trainlab, tr_pred )

mean_squared_error( testlab, ts_pred )
root_mean_squared_error( testlab, ts_pred )


# Split val and target 
val = data.drop(['DateTime', "Global_active_power"], axis = 1)
tar = data['Global_active_power']

# Train and test split 
trdata, tsdata, trlab, tslab = train_test_split( val, tar, test_size= 0.20, random_state= 46 )

# Model Selection
model = KNeighborsRegressor(n_neighbors=5)
model.fit(trdata, trlab)

# prediction
tr_pred = model.predict(trdata)
ts_pred = model.predict(tsdata)

#Performance check
mean_squared_error( trlab, tr_pred )
root_mean_squared_error( trlab, tr_pred )

mean_squared_error( tslab, ts_pred )
root_mean_squared_error( tslab, ts_pred )


#Tuning
para = {
    "n_neighbors": list(range(3, 5, 2)),
    "metric": ["minkowski"],
    "leaf_size": [30,10,20],
    "p": [1,2]
}

# Built Model
knn = KNeighborsRegressor()
gsv = GridSearchCV(param_grid= para, cv = 5, estimator= knn)
gsv.fit(trdata, trlab)

# Prediction
tr_pred = gsv.best_estimator_.predict(trdata)
ts_pred = gsv.best_estimator_.predict(tsdata)

# Performance Check
mean_squared_error( trlab, tr_pred )
root_mean_squared_error( trlab, tr_pred )

mean_squared_error( tslab, ts_pred )
root_mean_squared_error( tslab, ts_pred )


# Split val and target 
val = data.drop(['DateTime', "Global_active_power"], axis = 1)
tar = data['Global_active_power']


#Decision tree feature selection
fs = RandomForestRegressor(n_estimators= 200, random_state= 66)
fs.fit(val, tar)

fs.feature_importances_

pd.DataFrame({
    "Columns": val.columns,
    "Score": fs.feature_importances_ * 100
}).sort_values("Score", ascending = False)


# Feature selection 
selected_col = pd.DataFrame({
    "Columns": val.columns,
    "Score": fs.feature_importances_ * 100
}).sort_values("Score", ascending = False).head(4)['Columns'].to_list()

# Split val and target based on feature selection
val = val[selected_col]
tar = data['Global_active_power']
trdata, tsdata, trlab, tslab = train_test_split(val, tar, test_size= 0.20, random_state= 60)

# Built Model
dt_model = DecisionTreeRegressor(random_state= 60)
dt_model.fit(trdata, trlab)

# Prediction
tr_pred = dt_model.predict(trdata)
ts_pred = dt_model.predict(tsdata)

# Performance Check
mean_squared_error( trlab, tr_pred )
root_mean_squared_error( trlab, tr_pred )

mean_squared_error( tslab, ts_pred )
root_mean_squared_error( tslab, ts_pred )

