from flask import Flask, render_template, request, jsonify
from flask import Response
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

from bson import ObjectId

from scripts.load_offers import search_vestiaire, search_stockx
import threading
import gc

from scripts.estimatepriceforgivendays_anyproduct import estimate_price

from apscheduler.schedulers.background import BackgroundScheduler







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


# Set up logging
root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


# Call root route to check if the server is running
def call_root():
    base_url = os.getenv('BASE_URL', 'http://localhost:5000/')
    response = requests.get(base_url)
    print(f"Response from root: {response.text}")

scheduler = BackgroundScheduler()
scheduler.add_job(call_root, 'interval', minutes=13)
scheduler.start()

# Define the list of products
products = [
    {'brand': 'Hermes', 'model': 'Evelyn'},
    {'brand': 'Hermes', 'model': 'Kelly'},
    {'brand': 'Hermes', 'model': 'Birkin'},
    {'brand': 'Hermes', 'model': 'Picotin'},
    {'brand': 'Chanel', 'model': 'Flap'},
    {'brand': 'Chanel', 'model': 'Double Flap'},
    {'brand': 'Chanel', 'model': 'Boy'},
    {'brand': 'Chanel', 'model': '19'},
    {'brand': 'Chanel', 'model': '2.55'},
    {'brand': 'Chanel', 'model': 'mini Flap'},
    {'brand': 'Dior', 'model': 'Lady Dior'},
    {'brand': 'Dior', 'model': 'Diorama'},
    {'brand': 'Bottega Veneta', 'model': 'Cassette'},
    {'brand': 'Chanel', 'model': 'V Stitch'},
    {'brand': 'Louis Vuitton', 'model': 'Capucines'},
    {'brand': 'Louis Vuitton', 'model': 'Twist Chain'}
]


 ## Add a new product to watch
@app.route('/', methods=['GET', 'POST'])
def product_list():
    if request.method == 'POST':
        new_product = request.form.get('new_product')
        if new_product:
            products.append(new_product)

    return render_template('product_list.html', products=products)


## Load offers page
@app.route('/product_detail/<brand>/<model>', methods=['GET'])
def product_detail(brand, model):
    return render_template('offers.html', brand=brand, model=model)

# Load offers data 1
@app.route('/product_detail/data/stockx/<brand>/<model>', methods=['GET'])
def get_stockx_data(brand, model):
    print("load stockx data")
    stockx_result = search_stockx(brand, model)
    if stockx_result is not None:
        stockx_data, debug_info = stockx_result
        for item in stockx_data:
            item['source'] = 'StockX'
    else:
        stockx_data = []
    return jsonify(stockx_data=stockx_data)

# Load offers data 2
@app.route('/product_detail/data/vestiaire/<brand>/<model>', methods=['GET'])
def get_vestiaire_data(brand, model):
    print("load vestiaire data")
    vestiaire_result = search_vestiaire(brand, model)
    if vestiaire_result is not None:
        vestiaire_data = vestiaire_result[0]
        for item in vestiaire_data:
            item['source'] = 'VC'
    else:
        vestiaire_data = []
    return jsonify(vestiaire_data=vestiaire_data)




# Get Sales Items for all models
@app.route('/sales_stats/allmodels', methods=['GET'])
def sales_stats_allmodels():
    page = request.args.get('page', default = 1, type = int)
    per_page = request.args.get('per_page', default = 10, type = int)

    collections = client.productwatcher.list_collection_names()  # Get all collection names
    all_stats = []
    all_colors = set()

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

    return render_template('sales_stats_allmodels.html', all_stats=all_stats, colors=all_colors)


## Get Sales Items for a specific product
@app.route('/sales_stats/<brand>/<model>', methods=['GET'])
def sales_stats(brand, model):
    all_products = list(handbags.find({'collection': brand + " " + model}))  # Get all products from the new collection
    sold_items = [item for item in all_products if item.get('sold')]

    # calculate average price
    total_price = 0
    for item in sold_items:
        price = item.get('price')
        if isinstance(price, dict) and 'cents' in price and isinstance(price['cents'], (int, float)):
            total_price += price['cents']
        else:
            app.logger.warning(f"Unexpected price type for item {item['_id']}: {type(price)} with value {price}")

    average_price = round(total_price / len(sold_items) / 100, 2) if sold_items else 0  # divide by 100 to convert cents to euros

    # calculate best selling color
    color_counts = {}
    for item in sold_items:
        color = item.get('colors')
        if isinstance(color, dict) and 'all' in color and isinstance(color['all'], list) and color['all']:
            color_name = color['all'][0].get('name')
            if color_name in color_counts:
                color_counts[color_name] += 1
            else:
                color_counts[color_name] = 1
    best_selling_color = max(color_counts.items(), key=itemgetter(1))[0]

    # get top 5 liked products
    top_5_liked_products = sorted(sold_items, key=lambda x: x['likes'], reverse=True)[:5]

    # calculate average time to sell
    total_time_to_sell = sum(item['timeToSell'] for item in sold_items)
    average_time_to_sell = round(total_time_to_sell / len(sold_items))

    return render_template('sales_stats.html', brand=brand, model=model, average_time_to_sell=average_time_to_sell, best_selling_color=best_selling_color, average_price=average_price, top_5_liked_products=top_5_liked_products, all_products=all_products, currency="EUR")

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
        
# Estimate Price route 
@app.route('/estimate_price/<brand>/<model>/<color>/<buying_price>/<days>', methods=['GET'])
def estimate(brand, model, color, buying_price, days):
    # Convert the buying_price and days to int as they are passed as strings in the URL
    buying_price = int(buying_price)
    days = int(days)

    # Call the function from your script and get the result
    result = estimate_price(brand, model, color, buying_price, days)

    # Return the result as JSON
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)

    try:
        # This is here to simulate application activity (which keeps the main thread alive).
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        # Not strictly necessary if daemonic mode is enabled but should be done if possible
        scheduler.shutdown()




