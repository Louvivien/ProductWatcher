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
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout
from keras.initializers import Constant
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import logging
import sys
import gc



import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


import tracemalloc
tracemalloc.start()


# Set up logging
root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def set_up(Brand, Model, Color):


    ##################### Setup
    # Increase Gunicorn timeout
    timeout = 120

    # Load .env file
    logging.info("Loading environment variables...")
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path)

    # MongoDB setup
    logging.info("Setting up MongoDB connection...")
    MONGO_URI = os.getenv('MONGO_URI')
    MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')
    client = MongoClient(MONGO_URI.replace("<password>", MONGO_PASSWORD))
    db = client.productwatcher
    handbags = db.handbags





    # # Ask for inputs
    # Brand = input("Enter the brand of the product: ")
    # Model = input("Enter the model of the product: ")
    # Color = input("Enter the color of the product: ")
    # buying_price = int(input("Enter the buying price in €: "))
    # days = int(input("Enter the maximum number of days you are willing to sell the product: "))

    ##################### Getting data

    try:
        # Fetch all the sold bags for the brand and the model
        logging.info("Fetching all %s %s bags...", Brand, Model)
        bags = handbags.find({"brand.name": {"$regex": Brand, "$options": 'i'}, "model.name": {"$regex": Model, "$options": 'i'}})
        bags_count = handbags.count_documents({"brand.name": {"$regex": Brand, "$options": 'i'}, "model.name": {"$regex": Model, "$options": 'i'}})
        logging.info("Number of %s %s bags fetched: %s", Brand, Model, bags_count)


        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Fetching - Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

        if bags_count == 0:
            logging.info("No data in the database for this item")
            exit()

        # Fetch all the sold bags for the brand, the model and the color
        logging.info("Fetching all %s %s %s bags...", Brand, Model, Color)

        bags_color = handbags.find({"brand.name": {"$regex": Brand, "$options": 'i'}, "model.name": {"$regex": Model, "$options": 'i'}, "colors.all.name": {"$regex": Color, "$options": 'i'}})
        bags_color_count = handbags.count_documents({"brand.name": {"$regex": Brand, "$options": 'i'}, "model.name": {"$regex": Model, "$options": 'i'}, "colors.all.name": {"$regex": Color, "$options": 'i'}})
        logging.info("Number of %s %s %s bags fetched: %s", Brand, Model, Color, bags_color_count)


        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Fetching - Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

        color_data_exists = True
        if bags_color_count == 0:
            logging.info("No data for this specific color")
            color_data_exists = False

        # Convert the data to a pandas DataFrame
        df = pd.DataFrame(list(bags))
        del bags  # Free up memory
        gc.collect()  # Explicit garbage collection

        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Dataframe - Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

        # Convert the price and timeToSell to numeric values
        if 'price' in df.columns:
            df['price'] = df['price'].apply(lambda x: x['cents'] if isinstance(x, dict) and 'cents' in x else np.nan) / 100
        else:
            logging.info("Price field is missing or not in the expected format in some entries.")

        if 'timeToSell' in df.columns:
            df['timeToSell'] = pd.to_numeric(df['timeToSell'], errors='coerce')
        else:
            logging.info("timeToSell field is missing in some entries.")

        if color_data_exists:
            # Convert the data to a pandas DataFrame
            dp = pd.DataFrame(list(bags_color))
            del bags_color  # Free up memory
            gc.collect()  # Explicit garbage collection

            # Convert the price and timeToSell to numeric values
            if 'price' in dp.columns:
                dp['price'] = dp['price'].apply(lambda x: x['cents'] if isinstance(x, dict) and 'cents' in x else np.nan) / 100
            else:
                logging.info("Price field is missing or not in the expected format in some entries.")

            if 'timeToSell' in dp.columns:
                dp['timeToSell'] = pd.to_numeric(dp['timeToSell'], errors='coerce')
            else:
                logging.info("timeToSell field is missing in some entries.")


    except Exception as e:
        logging.info("An error occurred while getting the data:", str(e))
        exit()

    return (handbags, color_data_exists, bags_count, bags_color_count, df, dp)


def train_linear_model(Model, Color, color_data_exists, df, dp):
    

    ##################### Linear regression 

    # Perform the linear regression analysis for all model bags
    logging.info("Performing linear regression analysis for all %s bags...", Model)
    model1 = LinearRegression()


    model1.fit(df[['timeToSell']], df['price'])
    del df  # Free up memory
    gc.collect()  # Explicit garbage collection

    if color_data_exists:
        # Perform the linear regression analysis for model bags in the color   
        logging.info("Performing linear regression analysis for %s %s bags...", Color, Model)
        model2 = LinearRegression()
        model2.fit(dp[['timeToSell']], dp['price'])
        del dp  # Free up memory
        gc.collect()  # Explicit garbage collection

    def get_optimal_price_allmodels(days):
        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Linear Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        return model1.predict([[days]])

    def get_optimal_price_color(days):
        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Linear Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        return model2.predict([[days]])


    return (get_optimal_price_allmodels, get_optimal_price_color, model1, model2)


def train_polynomial_model( Model, Color, color_data_exists, df, dp):

    ##################### Polynomial regression 

    # Polynomial regression on all model bags
    logging.info("Performing polynomial regression analysis for all %s bags...", Model)

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(df[['timeToSell']])
    model3 = LinearRegression()
    model3.fit(X_poly, df['price'])
    del df  # Free up memory
    gc.collect()  # Explicit garbage collection

    if color_data_exists:
        # Polynomial regression on model bags in the color
        logging.info("Performing polynomial regression analysis  for %s %s bags...", Color, Model )

        X_poly_red = poly.fit_transform(dp[['timeToSell']])
        model4 = LinearRegression()
        model4.fit(X_poly_red, dp['price'])
        del dp  # Free up memory
        gc.collect()  # Explicit garbage collection

    # Define functions to get the optimal price for all models and color only using polynomial regression
    def get_optimal_price_allmodels_poly(days):
        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Polynomial Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        return model3.predict(poly.transform([[days]]))

    def get_optimal_price_color_poly(days):
        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Polynomial Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        return model4.predict(poly.transform([[days]]))

    return (get_optimal_price_allmodels_poly, get_optimal_price_color_poly, model3, model4, poly)

def train_decision_model( Model, Color, color_data_exists, df, dp):

    ##################### Decision tree regression 

    # Define the parameter grid
    param_grid = {
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [10, 20, 30, 40]
    }

    # Decision tree regression on all model bags with max_depth and min_samples_split parameters
    logging.info("Decision Performing decision tree regression analysis for all %s bags...", Model)
    dt = DecisionTreeRegressor()
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5)
    grid_search.fit(df[['timeToSell']], df['price'])
    best_params = grid_search.best_params_
    model5 = DecisionTreeRegressor(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])
    model5.fit(df[['timeToSell']], df['price'])
    del df  # Free up memory
    gc.collect()  # Explicit garbage collection

    if color_data_exists:
        # Decision tree regression on model bags in the color with max_depth and min_samples_split parameters
        logging.info("Decision Performing decision tree regression analysis  for %s %s bags...", Color, Model)
        grid_search.fit(dp[['timeToSell']], dp['price'])
        best_params = grid_search.best_params_
        model6 = DecisionTreeRegressor(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])
        model6.fit(dp[['timeToSell']], dp['price'])
        del dp  # Free up memory
        gc.collect()  # Explicit garbage collection

    # Define functions to get the optimal price for all models and color only using decision tree regression
    def get_optimal_price_allmodels_tree(days):
        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Decision Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        return model5.predict([[days]])

    def get_optimal_price_color_tree(days):
        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Decision Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        return model6.predict([[days]])

    return (get_optimal_price_allmodels_tree, get_optimal_price_color_tree, model5, model6)

def train_forest_model( Model, Color, color_data_exists, df, dp):
    ##################### Random Forest regression 

    # Set a seed for the random number generator to decrease randomness
    seed(1)

    # Shuffle the data
    df = df.sample(frac=1, random_state=1)
    if color_data_exists:
        dp = dp.sample(frac=1, random_state=1)

    # Random Forest regression on all model bags with max_depth and min_samples_split parameters
    logging.info("Performing Random Forest regression analysis for all %s bags...", Model)
    model7 = RandomForestRegressor(max_depth=10, min_samples_split=20, random_state=1)
    model7.fit(df[['timeToSell']], df['price'])
    del df  # Free up memory
    gc.collect()  # Explicit garbage collection

    if color_data_exists:
        # Random Forest regression on model bags in the color with max_depth and min_samples_split parameters
        logging.info("Performing Random Forest regression analysis for %s %s bags...", Color, Model)
        model8 = RandomForestRegressor(max_depth=10, min_samples_split=20, random_state=1)
        model8.fit(dp[['timeToSell']], dp['price'])
        del dp  # Free up memory
        gc.collect()  # Explicit garbage collection

    # Define functions to get the optimal price for all models and color only using Random Forest regression
    def get_optimal_price_allmodels_rf(days):
        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Forest Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        return model7.predict([[days]])

    def get_optimal_price_color_rf(days):
        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Forest  Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        if color_data_exists:
            return model8.predict([[days]])
        else:
            logging.info("Color data does not exist.")

    return (get_optimal_price_allmodels_rf, get_optimal_price_color_rf, model7, model8)


def train_neural_model( color_data_exists, df, dp):

    #################### Neural network

    # Neural network

    # Set a seed for the random number generator to decrease randomness
    seed(1)
    tf.random.set_seed(2)

    # Scale data without fitting column names  
    scaler_all = MinMaxScaler()
    # logging.info("df type: ",type(df))
    # logging.info("df : ",df)
    # logging.info("df[['timeToSell']]: ",df[['timeToSell']],)
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

    # Set a smaller batch size
    batch_size = 15  # Adjust this value as needed

    # Initialize the weights to small random numbers to decrease randomness
    init = tf.keras.initializers.RandomNormal(seed=1)

    # Create a callback to save the model weights after each epoch
    checkpoint = ModelCheckpoint('model_weights.h5', save_weights_only=True)

    # Neural network regression on all model bags
    model9 = Sequential()
    model9.add(Dense(50, input_dim=1, activation='relu', kernel_regularizer=l2(0.01), kernel_initializer=init))  # Add L2 regularization  
    model9.add(Dropout(0.3))  # Increase dropout
    model9.add(Dense(30, activation='relu', kernel_initializer=init)) 
    model9.add(Dropout(0.3))  
    model9.add(Dense(1, kernel_initializer=init))

    model9.compile(loss='mean_squared_error', optimizer='adam')  
    history = model9.fit(X_scaled, df['price'], epochs=epochs, batch_size=batch_size, verbose=0, validation_split=validation_split, callbacks=[checkpoint])  

    if color_data_exists:
        # Neural network regression on all model bags in the color
        model10 = Sequential()  
        model10.add(Dense(50, input_dim=1, activation='relu', kernel_regularizer=l2(0.01), kernel_initializer=init))  
        model10.add(Dropout(0.3))   
        model10.add(Dense(30, activation='relu', kernel_initializer=init))
        model10.add(Dropout(0.3))   
        model10.add(Dense(1, kernel_initializer=init))  

        model10.compile(loss='mean_squared_error', optimizer='adam')
        history_red = model10.fit(X_scaled_red, dp['price'], epochs=epochs, batch_size=batch_size, verbose=0, validation_split=validation_split, callbacks=[checkpoint])   

    # Define functions to get the optimal price for all models and color only using neural network regression  
    def get_optimal_price_allmodels_nn(days):
        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Neural Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        return model9.predict(scaler_all.transform([[days]])) * 100  # Convert back to cents

    def get_optimal_price_color_nn(days):
        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Neural Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        return model10.predict(scaler_red.transform([[days]])) * 100  # Convert back to cents

    return (get_optimal_price_allmodels_nn, get_optimal_price_color_nn, model9, model10, scaler_all, scaler_red)



def calculate_profits(buying_price, days, color_data_exists, get_optimal_price_allmodels, get_optimal_price_allmodels_poly, get_optimal_price_allmodels_tree, get_optimal_price_allmodels_rf, get_optimal_price_allmodels_nn, get_optimal_price_color, get_optimal_price_color_poly, get_optimal_price_color_tree, get_optimal_price_color_rf, get_optimal_price_color_nn):
    # Profits 


    profit_allmodels_lr = int(round(get_optimal_price_allmodels(days)[0])) - buying_price
    profit_allmodels_poly = int(round(get_optimal_price_allmodels_poly(days)[0])) - buying_price
    profit_allmodels_tree = int(round(get_optimal_price_allmodels_tree(days)[0])) - buying_price
    profit_allmodels_rf = int(round(get_optimal_price_allmodels_rf(days)[0])) - buying_price
    profit_allmodels_nn = int(round(get_optimal_price_allmodels_nn(days)[0][0])) - buying_price

    if color_data_exists:
        profit_color_lr = int(round(get_optimal_price_color(days)[0])) - buying_price
        profit_color_poly = int(round(get_optimal_price_color_poly(days)[0])) - buying_price
        profit_color_tree = int(round(get_optimal_price_color_tree(days)[0])) - buying_price
        profit_color_rf = int(round(get_optimal_price_color_rf(days)[0])) - buying_price    
        profit_color_nn = int(round(get_optimal_price_color_nn(days)[0][0])) - buying_price

    current, peak = tracemalloc.get_traced_memory()
    logging.info(f"Profits Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    return (profit_allmodels_lr, profit_allmodels_poly, profit_allmodels_tree, profit_allmodels_rf, profit_allmodels_nn, profit_color_lr, profit_color_poly, profit_color_tree, profit_color_rf, profit_color_nn)


def results(Color, days, color_data_exists, get_optimal_price_allmodels, get_optimal_price_allmodels_poly, get_optimal_price_allmodels_tree, get_optimal_price_allmodels_rf, get_optimal_price_allmodels_nn, get_optimal_price_color, get_optimal_price_color_poly, get_optimal_price_color_tree, get_optimal_price_color_rf, get_optimal_price_color_nn, profit_allmodels_lr, profit_allmodels_poly, profit_allmodels_tree, profit_allmodels_rf, profit_allmodels_nn, profit_color_lr, profit_color_poly, profit_color_tree, profit_color_rf, profit_color_nn):

    ##################### Results


    logging.info("")
    logging.info("###############       Recommended prices for all models and profits:")
    logging.info("")

    # linear regression - all models
    logging.info("Linear regression - all models: % s€ | + %s €", int(round(get_optimal_price_allmodels(days)[0])), profit_allmodels_lr)
    # polynomial regression - all models
    logging.info("Polynomial regression - all models: % s€ | + %s €",  int(round(get_optimal_price_allmodels_poly(days)[0])), profit_allmodels_poly)
    # decision tree regression - all models
    logging.info("Decision tree regression - all models: % s€ | + %s €", int(round(get_optimal_price_allmodels_tree(days)[0])), profit_allmodels_tree)
    # random forest regression - all models
    logging.info("Random forest regression - all models: % s€ | + %s €",  int(round(get_optimal_price_allmodels_rf(days)[0])), profit_allmodels_rf)
    # neural network regression - all models
    logging.info("Neural network regression - all models: % s€ | + %s €", int(round(get_optimal_price_allmodels_nn(days)[0][0])), profit_allmodels_nn)

    logging.info("")

    if color_data_exists:
        # linear regression - color
        logging.info("Linear regression  - %s : %s € | + %s €", Color, int(round(get_optimal_price_color(days)[0])), profit_color_lr)
        # polynomial regression - color
        logging.info("Polynomial regression  - %s : %s € | + %s €", Color, int(round(get_optimal_price_color_poly(days)[0])), profit_color_poly)
        # decision tree regression - color
        logging.info("Decision tree regression  - %s : %s € | + %s €", Color, int(round(get_optimal_price_color_tree(days)[0])), profit_color_tree)
        # random forest regression - color
        logging.info("Random forest regression  - %s : %s € | + %s €", Color, int(round(get_optimal_price_color_rf(days)[0])), profit_color_rf )
        # neural network regression - color
        logging.info("Neural network regression - %s : %s € | + %s €", Color, int(round(get_optimal_price_color_nn(days)[0][0])), profit_color_nn)




def evaluate(Color, days, color_data_exists, df, dp, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, scaler_all, scaler_red, get_optimal_price_allmodels, get_optimal_price_allmodels_poly, get_optimal_price_allmodels_tree, get_optimal_price_allmodels_rf, get_optimal_price_allmodels_nn, get_optimal_price_color, get_optimal_price_color_poly, get_optimal_price_color_tree, get_optimal_price_color_rf, get_optimal_price_color_nn, poly):

    ##################### Evaluation

    logging.info("")
    logging.info("###############       Evaluate the models against the average prices:")
    logging.info("")

    # Calculate the average price for all models and the color one
    avg_price_all = int(round((df['price']*100).mean()))
    if color_data_exists:
        avg_price_color = int(round((dp['price']*100).mean()))

    logging.info("Average price - all models: %s €", avg_price_all)
    if color_data_exists:
        logging.info("Average price - %s : %s €", Color, avg_price_color)

    current, peak = tracemalloc.get_traced_memory()
    logging.info(f"Evaluate Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    # Calculate the Mean Absolute Error (MAE) for each model
    from sklearn.metrics import mean_absolute_error

    mae_allmodels_lr = mean_absolute_error(df['price']*100, model1.predict(df[['timeToSell']]))
    mae_allmodels_poly = mean_absolute_error(df['price']*100, model3.predict(poly.transform(df[['timeToSell']])))
    mae_allmodels_tree = mean_absolute_error(df['price']*100, model5.predict(df[['timeToSell']]))
    mae_allmodels_rf = mean_absolute_error(df['price']*100, model7.predict(df[['timeToSell']]))
    mae_allmodels_nn = mean_absolute_error(df['price']*100, model9.predict(scaler_all.transform(df[['timeToSell']]))*100)
    if color_data_exists:
        mae_color_lr = mean_absolute_error(dp['price']*100, model2.predict(dp[['timeToSell']]))
        mae_color_poly = mean_absolute_error(dp['price']*100, model4.predict(poly.transform(dp[['timeToSell']])))
        mae_color_tree = mean_absolute_error(dp['price']*100, model6.predict(dp[['timeToSell']]))
        mae_color_rf = mean_absolute_error(dp['price']*100, model8.predict(dp[['timeToSell']]))
        mae_color_nn = mean_absolute_error(dp['price']*100, model10.predict(scaler_red.transform(dp[['timeToSell']]))*100)

    current, peak = tracemalloc.get_traced_memory()
    logging.info(f"Evaluate 2 Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    logging.info("MAE - Linear regression  - all models: %s €", round(mae_allmodels_lr))
    logging.info("MAE - Polynomial regression  - all models: %s €", round(mae_allmodels_poly, 2))
    logging.info("MAE - Decision tree regression  - all models: %s €", round(mae_allmodels_tree))
    logging.info("MAE - Random forest regression  - all models: %s €", round(mae_allmodels_rf))
    logging.info("MAE - Neural network regression - all models: %s €", round(mae_allmodels_nn))

    logging.info("")

    if color_data_exists:
        logging.info("MAE - Linear regression  - %s : %s €", Color, round(mae_color_lr))
        logging.info("MAE - Polynomial regression  - %s : %s €", Color, round(mae_color_poly))
        logging.info("MAE - Decision tree regression  - %s : %s €", Color, round(mae_color_tree))
        logging.info("MAE - Random forest regression  - %s : %s €", Color, round(mae_color_rf),)
        logging.info("MAE - Neural network regression - %s : %s €", Color, round(mae_color_nn))


    diff_allmodels_lr = abs(avg_price_all - int(round(get_optimal_price_allmodels(days)[0])))
    diff_allmodels_poly = abs(avg_price_all - int(round(get_optimal_price_allmodels_poly(days)[0])))
    diff_allmodels_tree = abs(avg_price_all - int(round(get_optimal_price_allmodels_tree(days)[0])))
    diff_allmodels_rf = abs(avg_price_all - int(round(get_optimal_price_allmodels_rf(days)[0])))
    diff_allmodels_nn = abs(avg_price_all - int(round(get_optimal_price_allmodels_nn(days)[0][0])))
    if color_data_exists:
        diff_color_lr = abs(avg_price_color - int(round(get_optimal_price_color(days)[0])))
        diff_color_poly = abs(avg_price_color - int(round(get_optimal_price_color_poly(days)[0])))
        diff_color_tree = abs(avg_price_color - int(round(get_optimal_price_color_tree(days)[0])))
        diff_color_rf = abs(avg_price_color - int(round(get_optimal_price_color_rf(days)[0])))
        diff_color_nn = abs(avg_price_color - int(round(get_optimal_price_color_nn(days)[0][0])))
    
    current, peak = tracemalloc.get_traced_memory()
    logging.info(f"Evaluate Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    # Create a dictionary to store the differences for each model
    diff_allmodels = {
        'Linear': diff_allmodels_lr,
        'Polynomial': diff_allmodels_poly,
        'Decision tree': diff_allmodels_tree,
        'Random forest': diff_allmodels_rf,
        'Neural network': diff_allmodels_nn
    }

    if color_data_exists:
        diff_color = {
            'Linear': diff_color_lr,
            'Polynomial': diff_color_poly,
            'Decision tree': diff_color_tree,
            'Random forest': diff_color_rf,
            'Neural network': diff_color_nn
        }

    return (diff_allmodels, diff_color, avg_price_all, avg_price_color)


def best(Color, buying_price, days, color_data_exists, diff_allmodels, diff_color, get_optimal_price_allmodels, get_optimal_price_allmodels_poly, get_optimal_price_allmodels_tree, get_optimal_price_allmodels_rf, get_optimal_price_allmodels_nn, get_optimal_price_color, get_optimal_price_color_poly, get_optimal_price_color_tree, get_optimal_price_color_rf, get_optimal_price_color_nn, bags_count, bags_color_count, avg_price_all,avg_price_color):

    # Find the model with the smallest difference with average prices
    best_model_all = min(diff_allmodels, key=diff_allmodels.get)
    if color_data_exists:
        best_model_color = min(diff_color, key=diff_color.get)

    # Calculate the predicted price and profit for the best models
    if best_model_all == 'Linear':
        predicted_price_all = int(round(get_optimal_price_allmodels(days)[0]))
    elif best_model_all == 'Polynomial':
        predicted_price_all = int(round(get_optimal_price_allmodels_poly(days)[0]))
    elif best_model_all == 'Decision tree':
        predicted_price_all = int(round(get_optimal_price_allmodels_tree(days)[0]))
    elif best_model_all == 'Random forest':
        predicted_price_all = int(round(get_optimal_price_allmodels_rf(days)[0]))
    # else:  # Neural network
        predicted_price_all = int(round(get_optimal_price_allmodels_nn(days)[0][0]))

    if color_data_exists:
        if best_model_color == 'Linear':
            predicted_price_color = int(round(get_optimal_price_color(days)[0]))
        elif best_model_color == 'Polynomial':
            predicted_price_color = int(round(get_optimal_price_color_poly(days)[0]))
        elif best_model_color == 'Decision tree':  
            predicted_price_color = int(round(get_optimal_price_color_tree(days)[0]))
        elif best_model_color == 'Random forest':
            predicted_price_color = int(round(get_optimal_price_color_rf(days)[0]))
        # else:  # Neural network
            predicted_price_color = int(round(get_optimal_price_color_nn(days)[0][0]))


    profit_all = predicted_price_all - buying_price
    if color_data_exists:
        profit_color = predicted_price_color - buying_price

    logging.info("")
    logging.info("Closest model to average price for all models: %s with a difference of %s €, a price of %s € and a profit of %s €", best_model_all, round(diff_allmodels[best_model_all], 2), predicted_price_all, profit_all )

    if color_data_exists:
        logging.info("Closest model to average price for %s bags: %s with a difference of %s €, a price of %s € and a profit of %s €", Color, best_model_color, round(diff_color[best_model_color], 2), predicted_price_color, profit_color)


    tracemalloc.stop()
    

# Return the results as a dictionary
    return {
            'bags_count': bags_count,
            'bags_color_count': bags_color_count if color_data_exists else None,
            'color_data_exists': color_data_exists,
            'avg_price_all': avg_price_all,
            'avg_price_color': avg_price_color if color_data_exists else None,
            'best_model_all': best_model_all,
            'diff_allmodels': round(diff_allmodels[best_model_all], 2),
            'best_model_color': best_model_color if color_data_exists else None,
            'diff_color_models': round(diff_color[best_model_color], 2) if color_data_exists else None,
            'predicted_price_all': predicted_price_all,
            'predicted_price_color': predicted_price_color if color_data_exists else None,
            'profit_all': round(profit_all, 2),
            'profit_color': round(profit_color, 2) if color_data_exists else None,
        }