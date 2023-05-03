import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('property_data.csv')

# Preprocess the data
data = data.dropna() # Drop any rows with missing values
data = pd.get_dummies(data, columns=['walk_score']) # Convert walk score to one-hot encoding
X = data.drop(['property_sold'], axis=1) # Features
y = data['property_sold'] # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predict the probability of property being sold
y_pred = model.predict_proba(X_test)[:,1]

# Evaluate the model
accuracy = accuracy_score(y_test, model.predict(X_test))
print('Accuracy: {:.2f}%'.format(accuracy * 100))
