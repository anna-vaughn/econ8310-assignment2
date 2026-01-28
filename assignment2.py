import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Get data.
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
data['DateTime'] = pd.to_datetime(data['DateTime']) # Convert data type to datetime.
data['hour'] = data['DateTime'].dt.hour # Get only hour.
data['day'] = data['DateTime'].dt.day # Get day (date).

# Get testing data.
## Meal is null, waiting to be filled in.
predData = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")
predData['DateTime'] = pd.to_datetime(predData['DateTime'])
predData['hour'] = predData['DateTime'].dt.hour
predData['month'] = predData['DateTime'].dt.month
predData['day'] = predData['DateTime'].dt.day

# Designate dependent (y) and independent (y) vars.
x = data[['Total', 'Discounts', 'hour', 'month' 'day']]
y = data['meal']

# Create XGBoost model.
model = XGBClassifier(
    n_estimators=150, 
    max_depth=15,
    learning_rate=0.5,
    objective='binary:logistic'
)

modelFit = model.fit(x,y)

# Test our model using the testing data
predData = predData[['Total', 'Discounts', 'hour', 'month', 'day']]
pred = modelFit.predict(predData)