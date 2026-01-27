import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Get data.
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
data['DateTime'] = pd.to_datetime(data['DateTime']) # Convert data type to datetime.
data['hour'] = data['DateTime'].dt.hour # Get only hour.
data['day'] = data['DateTime'].dt.day # Get day (date).

# Designate dependent (y) and independent (y) vars.
x = data[['Total', 'Discounts', 'hour']]
y = data['meal']

# Randomly sample our data --> 70% to train with, and 30% for testing
x, xt, y, yt = train_test_split(x, y, test_size=0.3)

# Create XGBoost model.
model = XGBClassifier(
    n_estimators=100, 
    max_depth=5,
    learning_rate=0.5
)

modelFit = model.fit(x,y)

# Test our model using the testing data.
p_test = modelFit.predict(xt)
p_test = p_test.astype(float) # To pass testValidPred.py

# limit pred array to 1000 results for submission.
pred = p_test[:1000]