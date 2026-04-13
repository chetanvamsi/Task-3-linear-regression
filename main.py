# STEP 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# STEP 2: Load dataset
data = pd.read_csv("data.csv")

print("Dataset:")
print(data.head())


# STEP 3: Convert text data to numbers (IMPORTANT)
data['mainroad'] = data['mainroad'].map({'yes':1, 'no':0})
data['guestroom'] = data['guestroom'].map({'yes':1, 'no':0})
data['basement'] = data['basement'].map({'yes':1, 'no':0})
data['hotwaterheating'] = data['hotwaterheating'].map({'yes':1, 'no':0})
data['airconditioning'] = data['airconditioning'].map({'yes':1, 'no':0})
data['prefarea'] = data['prefarea'].map({'yes':1, 'no':0})

data['furnishingstatus'] = data['furnishingstatus'].map({
    'furnished':2,
    'semi-furnished':1,
    'unfurnished':0
})


# STEP 4: Select features and target
X = data.drop('price', axis=1)   # use all features
y = data['price']


# STEP 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# STEP 6: Train model
model = LinearRegression()
model.fit(X_train, y_train)


# STEP 7: Predict
y_pred = model.predict(X_test)


# STEP 8: Evaluate
print("\nEvaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# STEP 9: Plot graph (ONLY area vs price)
# Sort values for proper line
sorted_indices = X_test['area'].argsort()

X_sorted = X_test['area'].iloc[sorted_indices]
y_sorted = y_pred[sorted_indices]

# Plot
plt.scatter(X_test['area'], y_test)
# plt.plot(X_sorted, y_sorted, color='red')

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Linear Regression (Area vs Price)")
plt.show()

# STEP 10: Coefficients
print("\nModel Details:")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)