import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Get data.
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")

# Designate dependent (y) and independent (y) vars.
x = data.drop(['id', 'DateTime', 'meal'], axis=1) # Drop string, timestamp, and dependent var.
y = data['meal']

# Randomly sample our data --> 70% to train with, and 30% for testing
x, xt, y, yt = train_test_split(x, y, test_size=0.3)

# Create XGBoost model.
model = XGBClassifier(
    n_estimators=50, 
    max_depth=10,
    learning_rate=0.5
)

modelFit = model.fit(x,y)

# Test our model using the testing data.
pred = modelFit.predict(xt)

# limit pred array to 1000 results for submission.
pred = pred[:1000]