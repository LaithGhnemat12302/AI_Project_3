import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,accuracy_score
from sklearn import metrics
from tabulate import tabulate

# Read data from file
data = pd.read_csv("Height_Weight.csv")

# 1) convert the height from inches to cms and the weight from pounds to kilograms.
data['Height'] = data['Height'] * 2.54
data['Weight'] = data['Weight'] * 0.453592

######################################################################################################################################
######################################################################################################################################
# 2) Print the main statistics of the features (i.e., mean, median, standard deviation, min, and max values) in a proper table.\
print("#################################################      Main statistics     ################################################\n")

pd.set_option('display.max_rows', None, 'display.max_columns', None)
des = data.describe(include='all').loc[['count','mean', 'std', 'min', 'max', '25%', '50%', '75%']]

print(tabulate(des,headers='keys', tablefmt='psql'))
print("\n")

######################################################################################################################################
######################################################################################################################################
# 3 to 8 questions.

def generateEvaluateModel(subset_size, model_name):
    subset_data = data.sample(n=subset_size, random_state=42)

    # Split the data into 70% training and 30% test.
    train_data, test_data = train_test_split(subset_data, test_size=0.3, random_state=42)

    # Define features and target.
    X_train = train_data[['Height']]
    y_train = train_data['Weight']
    X_test = test_data[['Height']]
    y_test = test_data['Weight']

    # Train a linear regression model.
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set.
    y_pred = model.predict(X_test)

    # Evaluate the model.
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r_square = metrics.r2_score(y_test, y_pred)

    # Print the performance metrics.
    print(f"Performance metrics for {model_name}:")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r_square}")
    print("\n#####################################################################################################################\n")

    # Plot actual vs predicted values.
    plt.figure(figsize=(10, 8))
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_pred, color='red', label='Predicted')

    plt.title(f'Actual vs Predicted values for {model_name}')
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.legend()
    plt.show()

    return {'Mean Squared Error': mse, 'Root Mean Squared Error': rmse, 'Mean Absolute Error': mae,'R-squared': r_square}
######################################################################################################################################
######################################################################################################################################

metrics_dict = {
    'M1': generateEvaluateModel(100, "M1"),
    'M2': generateEvaluateModel(1000, "M2"),
    'M3': generateEvaluateModel(5000, "M3"),
    'M4': generateEvaluateModel(len(data), "M4")
}

print("##########################################   Done by Laith Ghnemat & Salah Yaaqba   #########################################")

