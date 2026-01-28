import pandas as pd
from xgboost import XGBClassifier

# Get training data.
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")

# Get testing data.
## Meal is null, waiting to be filled in.
predData = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

# Designate dependent (y) and independent (y) vars.
x = data.drop(['id', 'DateTime', 'meal'], axis=1) # Drop string, timestamp, and dependent var.
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
predData = predData.drop(['id', 'DateTime', 'meal'], axis=1)
pred = modelFit.predict(predData)