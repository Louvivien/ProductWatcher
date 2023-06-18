from flask import Flask, render_template, request, jsonify
from flask import Response
from datetime import date, datetime

import pymongo  # Add this line

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

from load_offers import search_vestiaire, search_stockx
from estimatepriceforgivendays_anyproduct import estimate_price



# Load .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
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


 # Add a new product to watch
@app.route('/', methods=['GET', 'POST'])
def product_list():
    if request.method == 'POST':
        new_product = request.form.get('new_product')
        if new_product:
            products.append(new_product)

    return render_template('product_list.html', products=products)

 # Get all offers from APIs
@app.route('/product_detail/<brand>/<model>', methods=['GET'])
def product_detail(brand, model):
    stockx_result = search_stockx(brand, model)
    if stockx_result is None:
        return render_template('error.html', message="All requests failed due to proxy errors.", data=[])  
    else:
        queries, debug_info = stockx_result
    vestiaire_result = search_vestiaire(brand, model)
    return render_template('home.html', stockx_data=queries, vestiaire_data=vestiaire_result[0], debug_info=debug_info)

# Get Sales Items for a specific product
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


# Get Sales Items data (for all models)
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


@app.route('/estimate_price/', methods=['GET'])
def estimate_price_page():
    return render_template('estimate_price.html')





@app.route('/estimate_price/<brand>/<model>/<color>/<buying_price>/<days>', methods=['GET'])
def estimate(brand, model, color, buying_price, days):
    # Convert the buying_price and days to int as they are passed as strings in the URL
    buying_price = int(buying_price)
    days = int(days)

    # Call the estimate_price function
    result = estimate_price(brand, model, color, buying_price, days)

    # Return the result as a JSON response
    return jsonify(result)


@app.route('/estimate_price/', methods=['GET'])





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

        # # calculate average price
        # total_price = 0
        # for item in sold_items:
        #     price = item.get('price')
        #     if isinstance(price, dict) and 'cents' in price and isinstance(price['cents'], (int, float)):
        #         total_price += price['cents']
        #     else:
        #         app.logger.warning(f"Unexpected price type for item {item['_id']}: {type(price)} with value {price}")

        # average_price = round(total_price / len(sold_items) / 100, 2) if sold_items else 0  # divide by 100 to convert cents to euros

        # # calculate best selling color
        # color_counts = {}
        # for item in sold_items:
        #     color = item.get('colors')
        #     if isinstance(color, dict) and 'all' in color and isinstance(color['all'], list) and color['all']:
        #         color_name = color['all'][0].get('name')
        #         if color_name in color_counts:
        #             color_counts[color_name] += 1
        #         else:
        #             color_counts[color_name] = 1
        # best_selling_color = max(color_counts.items(), key=itemgetter(1))[0]

        # # get top 5 liked products
        # top_5_liked_products = sorted(sold_items, key=lambda x: x['likes'], reverse=True)[:5]

        # # calculate average time to sell
        # total_time_to_sell = sum(item['timeToSell'] for item in sold_items)
        # average_time_to_sell = round(total_time_to_sell / len(sold_items))

        all_stats.append({
            'collection_name': collection_name,
            # 'average_time_to_sell': average_time_to_sell,
            # 'best_selling_color': best_selling_color,
            # 'average_price': average_price,
            # 'top_5_liked_products': top_5_liked_products,
            'all_products': all_products,
            'currency': "EUR"
        })

    return render_template('sales_stats_allmodels.html', all_stats=all_stats, colors=all_colors)




if __name__ == '__main__':
    app.run(debug=True)












# @app.route('/dashboard1/<brand>/<model>', methods=['GET'])
# def dashboard1(brand, model):
    # all_products = list(handbags.find({'collection': brand + " " + model}))  # Get all products from the new collection


#     # Convert the list of dictionaries to a pandas DataFrame
#     df = pd.DataFrame(all_products)

#     # Flatten the nested fields in the DataFrame
#     df['price_euros'] = df['price'].apply(lambda x: x['cents'] / 100)  # convert cents to euros
#     df['brand_name'] = df['brand'].apply(lambda x: x['name'])
#     df['creation_date'] = pd.to_datetime(df['creationDate'], unit='ms')  # convert timestamp to datetime

#     # Convert ISO-2 country codes to ISO-3
#     df['country_iso3'] = df['country'].apply(lambda x: pycountry.countries.get(alpha_2=x).alpha_3)

#     # Create the plots
#     fig1 = px.histogram(df, x="price_euros", nbins=20, title="Price Distribution")
#     fig2 = px.scatter(df, x="price_euros", y="likes", title="Price vs Likes")
#     fig3 = px.pie(df, names="brand_name", title="Products by Brand")
#     fig4 = px.line(df, x="creation_date", y="price_euros", title="Price Trend Over Time")
#     fig5 = px.scatter_geo(df, locations="country_iso3", color="price_euros", title="Geographic Distribution of Products")

#     # Convert the plots to HTML and return
#     plot1 = fig1.to_html(full_html=False)
#     plot2 = fig2.to_html(full_html=False)
#     plot3 = fig3.to_html(full_html=False)
#     plot4 = fig4.to_html(full_html=False)
#     plot5 = fig5.to_html(full_html=False)

#     return render_template('dashboard1.html', plot1=plot1, plot2=plot2, plot3=plot3, plot4=plot4, plot5=plot5)



# @app.route('/dashboard2/<brand>/<model>', methods=['GET'])
# def dashboard2(brand, model):
    # all_products = list(handbags.find({'collection': brand + " " + model}))  # Get all products from the new collection


#     # Convert the list of dictionaries to a pandas DataFrame
#     df = pd.DataFrame(all_products)

#     # Flatten the nested fields in the DataFrame
#     df['price_euros'] = df['price'].apply(lambda x: x['cents'] / 100)  # convert cents to euros
#     df['brand_name'] = df['brand'].apply(lambda x: x['name'])
#     df['creation_date'] = pd.to_datetime(df['creationDate'], unit='ms')  # convert timestamp to datetime

#     # Convert ISO-2 country codes to ISO-3
#     df['country_iso3'] = df['country'].apply(lambda x: pycountry.countries.get(alpha_2=x).alpha_3)

#     # Create the data for the charts
#     data1 = df['price_euros'].tolist()
#     data2 = df[['price_euros', 'likes']].values.tolist()
#     data3 = df['brand_name'].value_counts().reset_index().values.tolist()
#     data4 = df[['creation_date', 'price_euros']].values.tolist()
#     data5 = df['country_iso3'].value_counts().reset_index().values.tolist()

#     return render_template('dashboard2.html', data1=data1, data2=data2, data3=data3, data4=data4, data5=data5)
