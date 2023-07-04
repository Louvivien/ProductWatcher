from flask import Flask, render_template, request, jsonify
from flask import Response

from flask import make_response
from flask import Flask, render_template, request, redirect, url_for


from datetime import date, datetime

import pymongo  

import requests
import json
from flask_bootstrap import Bootstrap
import curlify
import logging
import sys

from bson.code import Code
from operator import itemgetter
from dotenv import load_dotenv
import os

from pymongo import MongoClient

import plotly.express as px
import pandas as pd

import pycountry
import time


from bson import ObjectId

from scripts.load_offers import search_vestiaire, search_stockx, search_reoriginal
import threading
import gc

from scripts.estimatepriceforgivendays_anyproduct import estimate_price


from flask import request, jsonify


from scheduler_tasks import scheduler, call_product_detail


from urllib.parse import unquote
from unidecode import unidecode

from flask_caching import Cache

import atexit




# Define the list of products to watch
from config import products





# Load .env file
root_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(root_dir, '.env')
load_dotenv(dotenv_path)

# MongoDB setup
MONGO_URI = os.getenv('MONGO_URI')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')
client = MongoClient(MONGO_URI.replace("<password>", MONGO_PASSWORD))
db = client.productwatcher
handbags = db.handbags  

# This convert ObjectId / necessary for this route '/sales_stats/data'
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        elif isinstance(o, datetime):
            return o.isoformat()
        elif isinstance(o, date):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)

def jsonify(*args, **kwargs):
    return Response(json.dumps(dict(*args, **kwargs), cls=JSONEncoder), mimetype='application/json')


app = Flask(__name__)

app.json_encoder = JSONEncoder
Bootstrap(app)
cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.getenv('CACHE_REDIS_URL')
})


# scheduler.start()



 ## Tool to clear cache
@app.route('/clear_cache', methods=['GET'])
def clear_cache():
    print("Clearing cache")
    cache.clear()
    return "Cache has been cleared"


 ## Add a new product to watch
@app.route('/', methods=['GET', 'POST'])
def product_list():
    if request.method == 'POST':
        new_product = request.form.get('new_product')
        if new_product:
            products.append(new_product)
            cache.delete('product_list')  # Clear the cache when a new product is added
        return redirect(url_for('product_list'))  # Redirect to the GET request

    is_cached = False
    cache_key = 'product_list'
    product_list_html = cache.get(cache_key)
    if product_list_html is not None:
        is_cached = True
    else:
        product_list_html = render_template('product_list.html', products=products)
        cache.set(cache_key, product_list_html, timeout=36000)  # Cache the data for 36000 seconds

    return product_list_html, 200, {'From-Cache': str(is_cached)}


################## Buy ##################

## Load offers page
@app.route('/product_detail/<brand>/<model>', methods=['GET'])
def product_detail(brand, model):
    cache_key = f'product_detail_{brand}_{model}'
    page = cache.get(cache_key)
    if page is None:
        page = render_template('offers.html', brand=brand, model=model)
        cache.set(cache_key, page, timeout=3600)  # Cache the page for 3600 seconds
        is_cached = False
    else:
        is_cached = True
    response = make_response(page)
    response.headers['X-From-Cache'] = str(is_cached)
    return response

# Load offers data 1
@app.route('/product_detail/data/stockx/<brand>/<model>', methods=['GET'])
def get_stockx_data(brand, model):
    print("load stockx data")
    cache_key = f'stockx_data_{brand}_{model}'
    stockx_data = cache.get(cache_key)
    if stockx_data is None:
        stockx_result = search_stockx(brand, model)
        if stockx_result is not None:
            stockx_data, debug_info = stockx_result
            for item in stockx_data:
                item['source'] = 'StockX'
            cache.set(cache_key, stockx_data, timeout=3600)  # Cache the data for 3600 seconds
        else:
            stockx_data = []
        is_cached = False
    else:
        is_cached = True
    return jsonify({ "from_cache": is_cached, "stockx_data": stockx_data })


# Load offers data 2
@app.route('/product_detail/data/vestiaire/<brand>/<model>', methods=['GET'])
def get_vestiaire_data(brand, model):
    print("load vestiaire data")
    is_cached = False
    cache_key = f'vestiaire_data_{brand}_{model}'
    vestiaire_data = cache.get(cache_key)
    if vestiaire_data is not None:
        is_cached = True
    else:
        vestiaire_result = search_vestiaire(brand, model)
        if vestiaire_result is not None:
            vestiaire_data = vestiaire_result[0]
            for item in vestiaire_data:
                item['source'] = 'VC'
            cache.set(cache_key, vestiaire_data, timeout=3600)  # Cache the data for 3600 seconds
        else:
            vestiaire_data = []
    return jsonify(from_cache=is_cached, vestiaire_data=vestiaire_data)


# Load offers data 3
@app.route('/product_detail/data/original/<brand>/<model>', methods=['GET'])
def get_original_data(brand, model):
    print("load original data")
    is_cached = False
    cache_key = f'original_data_{brand}_{model}'
    original_data = cache.get(cache_key)
    if original_data is not None:
        is_cached = True
    else:
        original_result = search_reoriginal(brand, model)
        if original_result is not None:
            original_data, _, _ = original_result
            for item in original_data:
                item['source'] = 'Original'
            cache.set(cache_key, original_data, timeout=3600)  # Cache the data for 3600 seconds
        else:
            original_data = []
    return jsonify( from_cache=is_cached, original_data=original_data)

    
 # Load offers colors
@app.route('/get_image_color', methods=['POST'])
def get_color():
    print("get color")
    from scripts.getcolor import get_image_color
    data = request.get_json()
    image_url = data.get('image_url')
    if not image_url:
        return jsonify({'error': 'Missing image_url parameter'}), 400

    cache_key = f'color_{image_url}'
    color = cache.get(cache_key)
    is_cached = False
    if color is not None:
        is_cached = True
    else:
        try:
            color = get_image_color(image_url)
            cache.set(cache_key, color, timeout=36000)  # Cache the data for 3600 seconds
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({ 'from_cache': is_cached, 'color': color})   
    

# Load offers profits
@app.route('/get_profit/<brand>/<model>/<path:color>/<buying_price>', methods=['GET'])
def get_profit(brand, model, color, buying_price):
    print("get_profit route called")  
    brand = unidecode(unquote(brand))  
    print("Logging color") 
    print(f"Color: {color}")
    cache_key = f'profit_{brand}_{model}_{color}_{buying_price}'
    profit_data = cache.get(cache_key)
    is_cached = False
    if profit_data is not None:
        is_cached = True
    else:
        try:
            profit_data = estimate_price(brand, model, color, float(buying_price), 30)
            cache.set(cache_key, profit_data, timeout=36000)  # Cache the data for 36000 seconds
        except Exception as e:
            print(f"An error occurred in get_profit: {e}")  
            logging.error(f"An error occurred: {e}")
            return jsonify({"error": str(e)}), 500

    return jsonify(profit_data, from_cache=is_cached)
    

################## Sell ##################


# Get Sales Items for all models
@app.route('/sales_stats/allmodels', methods=['GET'])
def sales_stats_allmodels():
    page = request.args.get('page', default = 1, type = int)
    per_page = request.args.get('per_page', default = 10, type = int)
    cache_key = f'sales_stats_allmodels_{page}_{per_page}'
    is_cached = False
    all_stats = cache.get(cache_key)
    all_colors = set()

    if all_stats is not None:
        is_cached = True
    else:
        collections = client.productwatcher.list_collection_names()  # Get all collection names
        all_stats = []
        for collection_name in collections:
            collection = db[collection_name]
            all_products = list(collection.find().skip((page - 1) * per_page).limit(per_page))  # Get all products with pagination

            # Get all unique colors in the current collection
            colors = collection.distinct('colors.all.name')
            all_colors.update(colors)

            sold_items = [item for item in all_products if item.get('sold')]

            all_stats.append({
                'collection_name': collection_name,
                'all_products': all_products,
                'currency': "EUR"
            })
        cache.set(cache_key, all_stats, timeout=36000)  # Cache the data for 36000 seconds

    return render_template('sales_stats_allmodels.html', all_stats=all_stats, colors=all_colors, from_cache=is_cached)




# Get Sales Items for a specific product
@app.route('/sales_stats/<brand>/<model>', methods=['GET'])
def sales_stats(brand, model):
    cache_key = f'sales_stats_{brand}_{model}'
    cached_data = cache.get(cache_key)

    if cached_data is not None:
        from_cache = True
        all_products = cached_data
    else:
        from_cache = False
        all_products = list(handbags.find({'collection': brand + " " + model}))  # Get all products from the new collection
        cache.set(cache_key, all_products, timeout=36000)

    return render_template('sales_stats.html', brand=brand, model=model, all_products=all_products, currency="EUR", from_cache=from_cache)




# Get Sales Items data 
@app.route('/sales_stats/data', methods=['GET'])
def sales_stats_data():
    page = request.args.get('page', default = 1, type = int)
    per_page = request.args.get('per_page', default = 10, type = int)
    search_value = request.args.get('search[value]', default = '', type = str)
    timeToSell_min = request.args.get('timeToSell_min', default = None, type = int)
    timeToSell_max = request.args.get('timeToSell_max', default = None, type = int)
    likes_min = request.args.get('likes_min', default = None, type = int)
    likes_max = request.args.get('likes_max', default = None, type = int)
    price_min = request.args.get('price_min', default = None, type = int)
    price_max = request.args.get('price_max', default = None, type = int)

  # Get order parameters from request
    order_column = request.args.get('order[0][column]', default = None, type = int)
    order_dir = request.args.get('order[0][dir]', default = None, type = str)

    # Define a list of column names in the same order as in the DataTables initialization
    column_names = ['id', 'brand', 'model', 'name', 'color', 'price', 'likes', 'timeToSell', 'link']

    # Get checkbox value from request
    checkbox_values = request.args.get('checkbox_values', default = None, type = str)


    collections = client.productwatcher.list_collection_names()
    all_products_data = []

    for collection_name in collections:
        collection = db[collection_name]


        # Build the MongoDB query
        query = {}

        if search_value:
            query['$or'] = [
                {'id': {'$regex': search_value, '$options': 'i'}},
                {'brand.name': {'$regex': search_value, '$options': 'i'}},
                {'model.name': {'$regex': search_value, '$options': 'i'}},
                {'name': {'$regex': search_value, '$options': 'i'}},
                {'link': {'$regex': search_value, '$options': 'i'}}
            ]

        # Add checkbox filter to MongoDB query if it exists
        if checkbox_values is not None and checkbox_values != '':
            color_names = checkbox_values.split('|')
            query['colors.all.name'] = {'$in': color_names}

        if timeToSell_min is not None and timeToSell_max is not None:
            query['timeToSell'] = {'$gte': timeToSell_min, '$lte': timeToSell_max}

        if likes_min is not None and likes_max is not None:
            query['likes'] = {'$gte': likes_min, '$lte': likes_max}

        if price_min is not None and price_max is not None:
            query['price.cents'] = {'$gte': price_min*100, '$lte': price_max*100}

        # Add order parameters to MongoDB query if they exist
        if order_column is not None and order_dir is not None:
            sort = [(column_names[order_column], pymongo.ASCENDING if order_dir == 'asc' else pymongo.DESCENDING)]


        all_products = list(collection.find(query).sort(sort).skip((page - 1) * per_page).limit(per_page))

        for product in all_products:
            product_data = {
                "image": 'https://images.vestiairecollective.com/produit/' + str(product.get('id', '')) + '-1_3.jpg',
                "id": product.get('id', ''),
                "brand": product.get('brand', {}).get('name', '') if 'brand' in product else '',
                "model": product.get('model', {}).get('name', '') if 'model' in product else '',
                "name": product.get('name', ''),
                "color": product.get('colors', {}).get('all', [{}])[0].get('name', '') if product.get('colors') and product.get('colors').get('all') else '',
                "price": product.get('price', {}).get('cents', 0)/100 if product.get('price') and product.get('price').get('cents') else '',
                "likes": product.get('likes', ''),
                "timeToSell": product.get('timeToSell', ''),
                "link": 'https://fr.vestiairecollective.com/' + product.get('link', '') if product.get('link') else ''
            }
            all_products_data.append(product_data)


    records_total = collection.count_documents({})
    records_filtered = collection.count_documents({})

    response = {
        "draw": int(request.args.get('draw', default = 1)),
        "recordsTotal": records_total,
        "recordsFiltered": records_filtered,
        "data": all_products_data
    }

    return jsonify(response)

################## Estimate Prices ##################

## Estimate Price page
@app.route('/estimate_price/', methods=['GET'])
def estimate_price_page():
    return render_template('estimate_price.html')

# This dictionary will hold the status of each request
status_dict = {}

# Estimate Price route ML
@app.route('/estimate_price_ml/<brand>/<model>/<color>/<buying_price>/<days>', methods=['GET'])
def estimateML(brand, model, color, buying_price, days):
    # Convert the buying_price and days to int as they are passed as strings in the URL
    buying_price = int(buying_price)
    days = int(days)

    # Create a unique id for each request
    request_id = f"{brand}_{model}_{color}_{buying_price}_{days}"
    status_dict[request_id] = "Processing"

    # Start a new thread to process the request
    threading.Thread(target=process_request, args=(request_id, brand, model, color, buying_price, days)).start()

    # Return the initial response
    return jsonify({"status": "Processing", "request_id": request_id})

# Estimate ML Price request handling
@app.route('/status/<request_id>', methods=['GET'])
def get_status(request_id):
    # Return the status of the request
    return jsonify({"status": status_dict.get(request_id, "Not found")})

def process_request(request_id, brand, model, color, buying_price, days):
    try:
        from scripts.estimatepriceforgivendays_anyproduct_ml import (set_up, train_linear_model, train_forest_model, train_polynomial_model, train_decision_model, train_neural_model, calculate_profits, results, evaluate, best)

        status_dict[request_id] = "Setting up"
        set_up_result = set_up(brand, model, color)
        handbags, color_data_exists, bags_count, bags_color_count, df, dp = set_up_result

        del set_up_result
        gc.collect()

        status_dict[request_id] = "Training linear model"
        train_linear_model_result = train_linear_model(model, color, color_data_exists, df, dp)
        get_optimal_price_allmodels, get_optimal_price_color, model1, model2 = train_linear_model_result

        del train_linear_model_result
        gc.collect()

        status_dict[request_id] = "Training polynomial model"
        train_polynomial_model_result = train_polynomial_model(model, color, color_data_exists, df, dp)
        get_optimal_price_allmodels_poly, get_optimal_price_color_poly, model3, model4, poly = train_polynomial_model_result

        del train_polynomial_model_result
        gc.collect()

        status_dict[request_id] = "Training decision tree model"
        train_decision_model_result = train_decision_model(model, color, color_data_exists, df, dp)
        get_optimal_price_allmodels_tree, get_optimal_price_color_tree, model5, model6 = train_decision_model_result

        del train_decision_model_result
        gc.collect()

        status_dict[request_id] = "Training forest model"
        train_forest_model_result = train_forest_model(model, color, color_data_exists, df, dp)
        get_optimal_price_allmodels_rf, get_optimal_price_color_rf, model7, model8 = train_forest_model_result

        del train_forest_model_result
        gc.collect()

        status_dict[request_id] = "Training neural model"
        train_neural_model_result = train_neural_model(color_data_exists, df, dp)
        get_optimal_price_allmodels_nn, get_optimal_price_color_nn, model9, model10, scaler_all, scaler_red = train_neural_model_result

        del train_neural_model_result
        gc.collect()

        status_dict[request_id] = "Calculating profits"
        calculate_profits_result = calculate_profits(buying_price, days, color_data_exists, get_optimal_price_allmodels, get_optimal_price_allmodels_poly, get_optimal_price_allmodels_tree, get_optimal_price_allmodels_rf, get_optimal_price_allmodels_nn, get_optimal_price_color, get_optimal_price_color_poly, get_optimal_price_color_tree, get_optimal_price_color_rf, get_optimal_price_color_nn)
        profit_allmodels_lr, profit_allmodels_poly, profit_allmodels_tree, profit_allmodels_rf, profit_allmodels_nn, profit_color_lr, profit_color_poly, profit_color_tree, profit_color_rf, profit_color_nn = calculate_profits_result

        del calculate_profits_result
        gc.collect()

        status_dict[request_id] = "Getting results"
        results_result = results(color, days, color_data_exists, get_optimal_price_allmodels, get_optimal_price_allmodels_poly, get_optimal_price_allmodels_tree, get_optimal_price_allmodels_rf, get_optimal_price_allmodels_nn, get_optimal_price_color, get_optimal_price_color_poly, get_optimal_price_color_tree, get_optimal_price_color_rf, get_optimal_price_color_nn, profit_allmodels_lr, profit_allmodels_poly, profit_allmodels_tree, profit_allmodels_rf, profit_allmodels_nn, profit_color_lr, profit_color_poly, profit_color_tree, profit_color_rf, profit_color_nn)

        del results_result
        gc.collect()

        status_dict[request_id] = "Evaluating"
        evaluate_result = evaluate(color, days, color_data_exists, df, dp, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, scaler_all, scaler_red, get_optimal_price_allmodels, get_optimal_price_allmodels_poly, get_optimal_price_allmodels_tree, get_optimal_price_allmodels_rf, get_optimal_price_allmodels_nn, get_optimal_price_color, get_optimal_price_color_poly, get_optimal_price_color_tree, get_optimal_price_color_rf, get_optimal_price_color_nn, poly)
        diff_allmodels, diff_color, avg_price_all, avg_price_color = evaluate_result

        del evaluate_result
        gc.collect()

        status_dict[request_id] = "Getting best model"
        best_result = best(color, buying_price, days, color_data_exists, diff_allmodels, diff_color, get_optimal_price_allmodels, get_optimal_price_allmodels_poly, get_optimal_price_allmodels_tree, get_optimal_price_allmodels_rf, get_optimal_price_allmodels_nn, get_optimal_price_color, get_optimal_price_color_poly, get_optimal_price_color_tree, get_optimal_price_color_rf, get_optimal_price_color_nn, bags_count, bags_color_count, avg_price_all,avg_price_color)
        best_result['color'] = color

  

        # Update the status to complete and store the result
        status_dict[request_id] = {"status": "Complete", "result": best_result}

        del best_result
        gc.collect()

        # Log the final status
        print(f"Final status for request {request_id}: {status_dict[request_id]}")

    except Exception as e:
        # If an error occurs, update the status to "Error" and store the error message
        status_dict[request_id] = {"status": "Error", "error": str(e)}
        print(f"Error occurred for request {request_id}: {e}")
        
from flask import request

# Estimate Price route 
@app.route('/estimate_price/<brand>/<model>/<color>/<buying_price>/<days>', methods=['POST'])
def estimate(brand, model, color, buying_price, days):
    # Convert the buying_price and days to int as they are passed as strings in the URL
    buying_price = int(buying_price)
    days = int(days)

    # Get the data from the request body
    data = request.get_json()

    # Extract the parameters from the data
    modelName = data.get('modelName')
    material = data.get('material')
    condition = data.get('condition')
    vintage = data.get('vintage') == 'true'

    # Call the function from your script and get the result
    result = estimate_price(brand, model, color, buying_price, days, material, condition, modelName, vintage)

    # Return the result as JSON
    return jsonify(result)





if __name__ == '__main__':
    # Register the scheduler.shutdown function to be called when the Python interpreter is about to exit
    atexit.register(scheduler.shutdown)

    app.run(debug=False)