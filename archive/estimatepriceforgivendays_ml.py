from pymongo import MongoClient
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from numpy.random import seed

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout
from keras.initializers import Constant

from sklearn.model_selection import GridSearchCV


import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


# TO DO: update database call like in non archived version
# database call not accurate, use collection instead



##################### Setup


# Load .env file
root_dir = os.path.dirname(os.path.abspath(__file__))
print("Loading environment variables...")
dotenv_path = os.path.join(root_dir, '.env')
load_dotenv(dotenv_path)

# MongoDB setup
# print("Setting up MongoDB connection...")
MONGO_URI = os.getenv('MONGO_URI')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')
client = MongoClient(MONGO_URI.replace("<password>", MONGO_PASSWORD))
db = client.productwatcher
handbags = db.handbags

# Ask for a number of days as an input
days = int(input("Enter the maximum number of days you are willing to sell the product: "))
buying_price = int(input("Enter the buying price in €: "))

# Fetch all the Louis Vuitton Capucines bags
print("Fetching all Louis Vuitton Capucine bags...")
lv_capucine_bags = handbags.find({"brand.name": "Louis Vuitton", "model.name": "Capucines"})
bags_count = handbags.count_documents({"brand.name": "Louis Vuitton", "model.name": "Capucines"})
print("Number of bags fetched: ", bags_count)

# Fetch all the Louis Vuitton Capucines red bags
print("Fetching all red Louis Vuitton Capucine bags...")
lv_capucine_red_bags = handbags.find({"brand.name": "Louis Vuitton", "model.name": "Capucines", "colors.all.name": "Red"})
bags_count = handbags.count_documents({"brand.name": "Louis Vuitton", "model.name": "Capucines", "colors.all.name": "Red"})
print("Number of red bags fetched: ", bags_count)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(list(lv_capucine_bags))

# Convert the price and timeToSell to numeric values
df['price'] = df['price'].apply(lambda x: x['cents']) / 100
df['timeToSell'] = df['timeToSell'].apply(int)

# Convert the data to a pandas DataFrame
dp = pd.DataFrame(list(lv_capucine_red_bags))

# Convert the price and timeToSell to numeric values
dp['price'] = dp['price'].apply(lambda x: x['cents']) / 100
dp['timeToSell'] = dp['timeToSell'].apply(int)




##################### Linear regression 

# Perform the linear regression analysis for all Capucine bags
# print("Performing linear regression analysis for all Capucine bags...")
model1 = LinearRegression()
model1.fit(df[['timeToSell']], df['price'])

# Perform the linear regression analysis for red Capucine bags    
# print("Performing linear regression analysis for red Capucine bags...")
model2 = LinearRegression()
model2.fit(dp[['timeToSell']], dp['price'])

def get_optimal_price_allmodels(days):
    return model1.predict([[days]])

def get_optimal_price_color(days):
    return model2.predict([[days]])





##################### Polynomial regression 

# Polynomial regression on all Capucine bags
# print("Performing polynomial regression analysis for all Capucine bags...")
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(df[['timeToSell']])
model3 = LinearRegression()
model3.fit(X_poly, df['price'])

# Polynomial regression on red Capucine bags
# print("Performing polynomial regression analysis for red Capucine bags...")
X_poly_red = poly.fit_transform(dp[['timeToSell']])
model4 = LinearRegression()
model4.fit(X_poly_red, dp['price'])

# Define functions to get the optimal price for all models and red only using polynomial regression
def get_optimal_price_allmodels_poly(days):
    return model3.predict(poly.transform([[days]]))

def get_optimal_price_color_poly(days):
    return model4.predict(poly.transform([[days]]))




##################### Decision tree regression 

# Define the parameter grid
param_grid = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [10, 20, 30, 40]
}

# Decision tree regression on all Capucine bags with max_depth and min_samples_split parameters
# print("Performing decision tree regression analysis for all Capucine bags...")
dt = DecisionTreeRegressor()
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5)
grid_search.fit(df[['timeToSell']], df['price'])
best_params = grid_search.best_params_
model5 = DecisionTreeRegressor(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])
model5.fit(df[['timeToSell']], df['price'])

# Decision tree regression on red Capucine bags with max_depth and min_samples_split parameters
# print("Performing decision tree regression analysis for red Capucine bags...")
grid_search.fit(dp[['timeToSell']], dp['price'])
best_params = grid_search.best_params_
model6 = DecisionTreeRegressor(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])
model6.fit(dp[['timeToSell']], dp['price'])

# Define functions to get the optimal price for all models and red only using decision tree regression
def get_optimal_price_allmodels_tree(days):
    return model5.predict([[days]])

def get_optimal_price_color_tree(days):
    return model6.predict([[days]])



##################### Random Forest regression 

# Set a seed for the random number generator to decrease randomness
seed(1)

# Shuffle the data
df = df.sample(frac=1, random_state=1)
dp = dp.sample(frac=1, random_state=1)

# Random Forest regression on all Capucine bags with max_depth and min_samples_split parameters
# print("Performing Random Forest regression analysis for all Capucine bags...")
model7 = RandomForestRegressor(max_depth=10, min_samples_split=20, random_state=1)
model7.fit(df[['timeToSell']], df['price'])

# Random Forest regression on red Capucine bags with max_depth and min_samples_split parameters
# print("Performing Random Forest regression analysis for red Capucine bags...")
model8 = RandomForestRegressor(max_depth=10, min_samples_split=20, random_state=1)
model8.fit(dp[['timeToSell']], dp['price'])

# Define functions to get the optimal price for all models and red only using Random Forest regression
def get_optimal_price_allmodels_rf(days):
    return model7.predict([[days]])

def get_optimal_price_color_rf(days):
    return model8.predict([[days]])


##################### Neural network

# Neural network
# Set a seed for the random number generator to decrease randomness
seed(1)
tf.random.set_seed(2)

# Scale data without fitting column names  
scaler_all = MinMaxScaler()
X_scaled = scaler_all.fit_transform(df[['timeToSell']])

scaler_red = MinMaxScaler() 
X_scaled_red = scaler_red.fit_transform(dp[['timeToSell']])  

# Convert price from cents to euros
df['price'] = df['price'] / 100
dp['price'] = dp['price'] / 100

# Increase the number of epochs
epochs = 200  

# Add a validation split 
validation_split = 0.2  

# Initialize the weights to small random numbers to decrease randomness
init = tf.keras.initializers.RandomNormal(seed=1)

# Neural network regression on all Capucine bags
# print("Performing neural network regression analysis for all Capucine bags...")

model9 = Sequential()
model9.add(Dense(50, input_dim=1, activation='relu', kernel_regularizer=l2(0.01), kernel_initializer=init))  # Add L2 regularization  
model9.add(Dropout(0.3))  # Increase dropout
model9.add(Dense(30, activation='relu', kernel_initializer=init)) 
model9.add(Dropout(0.3))  
model9.add(Dense(1, kernel_initializer=init))

model9.compile(loss='mean_squared_error', optimizer='adam')  
history = model9.fit(X_scaled, df['price'], epochs=epochs, verbose=0, validation_split=validation_split)  

# Neural network regression on Capucine red bags
# print("Performing neural network regression analysis for Capucine red bags...")

model10 = Sequential()  
model10.add(Dense(50, input_dim=1, activation='relu', kernel_regularizer=l2(0.01), kernel_initializer=init))  
model10.add(Dropout(0.3))   
model10.add(Dense(30, activation='relu', kernel_initializer=init))
model10.add(Dropout(0.3))   
model10.add(Dense(1, kernel_initializer=init))  

model10.compile(loss='mean_squared_error', optimizer='adam')
history_red = model10.fit(X_scaled_red, dp['price'], epochs=epochs, verbose=0, validation_split=validation_split)   

# Define functions to get the optimal price for all models and red only using neural network regression  
def get_optimal_price_allmodels_nn(days):
    return model9.predict(scaler_all.transform([[days]])) * 100  # Convert back to cents

def get_optimal_price_color_nn(days):
    return model10.predict(scaler_red.transform([[days]])) * 100  # Convert back to cents




# Profits 


profit_allmodels_lr = int(round(get_optimal_price_allmodels(days)[0])) - buying_price
profit_color_lr = int(round(get_optimal_price_color(days)[0])) - buying_price

profit_allmodels_poly = int(round(get_optimal_price_allmodels_poly(days)[0])) - buying_price
profit_color_poly = int(round(get_optimal_price_color_poly(days)[0])) - buying_price

profit_allmodels_tree = int(round(get_optimal_price_allmodels_tree(days)[0])) - buying_price
profit_color_tree = int(round(get_optimal_price_color_tree(days)[0])) - buying_price

profit_allmodels_rf = int(round(get_optimal_price_allmodels_rf(days)[0])) - buying_price
profit_color_rf = int(round(get_optimal_price_color_rf(days)[0])) - buying_price

profit_allmodels_nn = int(round(get_optimal_price_allmodels_nn(days)[0][0])) - buying_price
profit_color_nn = int(round(get_optimal_price_color_nn(days)[0][0])) - buying_price





##################### Results


print("")
print("###############       Recommended prices for all models and profits:")
print("")

# linear regression - all models
print("Linear regression - all models:", int(round(get_optimal_price_allmodels(days)[0])), "€ | +", profit_allmodels_lr, "€")
# polynomial regression - all models
print("Polynomial regression - all models:", int(round(get_optimal_price_allmodels_poly(days)[0])), "€ | +", profit_allmodels_poly, "€")
# decision tree regression - all models
print("Decision tree regression - all models:", int(round(get_optimal_price_allmodels_tree(days)[0])), "€ | +", profit_allmodels_tree, "€")
# random forest regression - all models
print("Random forest regression - all models:", int(round(get_optimal_price_allmodels_rf(days)[0])), "€ | +", profit_allmodels_rf, "€")
# neural network regression - all models
print("Neural network regression - all models:", int(round(get_optimal_price_allmodels_nn(days)[0][0])), "€ | +", profit_allmodels_nn, "€")

print("")

# linear regression - red
print("Linear regression - red:", int(round(get_optimal_price_color(days)[0])), "€ | +", profit_color_lr, "€")
# polynomial regression - red
print("Polynomial regression - red:", int(round(get_optimal_price_color_poly(days)[0])), "€ | +", profit_color_poly, "€")
# decision tree regression - red
print("Decision tree regression - red:", int(round(get_optimal_price_color_tree(days)[0])), "€ | +", profit_color_tree, "€")
# random forest regression - red
print("Random forest regression - red:", int(round(get_optimal_price_color_rf(days)[0])), "€ | +", profit_color_rf, "€")
# neural network regression - red
print("Neural network regression - red:", int(round(get_optimal_price_color_nn(days)[0][0])), "€ | +", profit_color_nn, "€")



##################### Evaluation

print("")
print("###############       Evaluate the models against the average prices:")
print("")

# Calculate the average price for all models and the red one
avg_price_all = int(round((df['price']*100).mean()))
avg_price_red = int(round((dp['price']*100).mean()))

print("Average price - all models:", avg_price_all, "€")
print("Average price - red:", avg_price_red, "€")

# Calculate the Mean Absolute Error (MAE) for each model
from sklearn.metrics import mean_absolute_error

mae_allmodels_lr = mean_absolute_error(df['price']*100, model1.predict(df[['timeToSell']]))
mae_color_lr = mean_absolute_error(dp['price']*100, model2.predict(dp[['timeToSell']]))

mae_allmodels_poly = mean_absolute_error(df['price']*100, model3.predict(poly.transform(df[['timeToSell']])))
mae_color_poly = mean_absolute_error(dp['price']*100, model4.predict(poly.transform(dp[['timeToSell']])))

mae_allmodels_tree = mean_absolute_error(df['price']*100, model5.predict(df[['timeToSell']]))
mae_color_tree = mean_absolute_error(dp['price']*100, model6.predict(dp[['timeToSell']]))

mae_allmodels_rf = mean_absolute_error(df['price']*100, model7.predict(df[['timeToSell']]))
mae_color_rf = mean_absolute_error(dp['price']*100, model8.predict(dp[['timeToSell']]))

mae_allmodels_nn = mean_absolute_error(df['price']*100, model9.predict(scaler_all.transform(df[['timeToSell']]))*100)
mae_color_nn = mean_absolute_error(dp['price']*100, model10.predict(scaler_red.transform(dp[['timeToSell']]))*100)


print("MAE - Linear regression - all models:", round(mae_allmodels_lr), "€")
print("MAE - Polynomial regression - all models:", round(mae_allmodels_poly, 2), "€")
print("MAE - Decision tree regression - all models:", round(mae_allmodels_tree), "€")
print("MAE - Random forest regression - all models:", round(mae_allmodels_rf), "€")
print("MAE - Neural network regression - all models:", round(mae_allmodels_nn), "€")

print("")

print("MAE - Linear regression - red:", round(mae_color_lr), "€")
print("MAE - Polynomial regression - red:", round(mae_color_poly), "€")
print("MAE - Decision tree regression - red:", round(mae_color_tree), "€")
print("MAE - Random forest regression - red:", round(mae_color_rf), "€")
print("MAE - Neural network regression - red:", round(mae_color_nn), "€")


diff_allmodels_lr = abs(avg_price_all - int(round(get_optimal_price_allmodels(days)[0])))
diff_color_lr = abs(avg_price_red - int(round(get_optimal_price_color(days)[0])))

diff_allmodels_poly = abs(avg_price_all - int(round(get_optimal_price_allmodels_poly(days)[0])))
diff_color_poly = abs(avg_price_red - int(round(get_optimal_price_color_poly(days)[0])))

diff_allmodels_tree = abs(avg_price_all - int(round(get_optimal_price_allmodels_tree(days)[0])))
diff_color_tree = abs(avg_price_red - int(round(get_optimal_price_color_tree(days)[0])))

diff_allmodels_rf = abs(avg_price_all - int(round(get_optimal_price_allmodels_rf(days)[0])))
diff_color_rf = abs(avg_price_red - int(round(get_optimal_price_color_rf(days)[0])))

diff_allmodels_nn = abs(avg_price_all - int(round(get_optimal_price_allmodels_nn(days)[0][0])))
diff_color_nn = abs(avg_price_red - int(round(get_optimal_price_color_nn(days)[0][0])))





# Create a dictionary to store the differences for each model
diff_allmodels = {
    'Linear': diff_allmodels_lr,
    'Polynomial': diff_allmodels_poly,
    'Decision tree': diff_allmodels_tree,
    'Random forest': diff_allmodels_rf,
    'Neural network': diff_allmodels_nn
}

diff_color = {
    'Linear': diff_color_lr,
    'Polynomial': diff_color_poly,
    'Decision tree': diff_color_tree,
    'Random forest': diff_color_rf,
    'Neural network': diff_color_nn
}

# Find the model with the smallest difference with average prices
best_model_all = min(diff_allmodels, key=diff_allmodels.get)
best_model_red = min(diff_color, key=diff_color.get)

# Calculate the predicted price and profit for the best models
if best_model_all == 'Linear':
    predicted_price_all = int(round(get_optimal_price_allmodels(days)[0]))
elif best_model_all == 'Polynomial':
    predicted_price_all = int(round(get_optimal_price_allmodels_poly(days)[0]))
elif best_model_all == 'Decision tree':
    predicted_price_all = int(round(get_optimal_price_allmodels_tree(days)[0]))
elif best_model_all == 'Random forest':
    predicted_price_all = int(round(get_optimal_price_allmodels_rf(days)[0]))
else:  # Neural network
    predicted_price_all = int(round(get_optimal_price_allmodels_nn(days)[0][0]))

if best_model_red == 'Linear':
    predicted_price_red = int(round(get_optimal_price_color(days)[0]))
elif best_model_red == 'Polynomial':
    predicted_price_red = int(round(get_optimal_price_color_poly(days)[0]))
elif best_model_red == 'Decision tree':  
    predicted_price_red = int(round(get_optimal_price_color_tree(days)[0]))
elif best_model_red == 'Random forest':
    predicted_price_red = int(round(get_optimal_price_color_rf(days)[0]))
else:  # Neural network
    predicted_price_red = int(round(get_optimal_price_color_nn(days)[0][0]))


profit_all = predicted_price_all - buying_price
profit_red = predicted_price_red - buying_price

print("")
print("Closest model to average price for all models:", best_model_all, "with a difference of", round(diff_allmodels[best_model_all], 2), "€, a price of", predicted_price_all, "€ and a profit of", profit_all, "€")
print("Closest model to average price for red bags:", best_model_red, "with a difference of", round(diff_color[best_model_red], 2), "€, a price of", predicted_price_red, "€ and a profit of", profit_red, "€")
