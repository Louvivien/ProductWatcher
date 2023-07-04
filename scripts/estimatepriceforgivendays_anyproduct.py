# Improvements
# recommended price all should take into account number of days
# ideallly give all intermediate results for each parameters



from pymongo import MongoClient  
import sys
import os  
import statistics  
from dotenv import load_dotenv  
import numpy as np  
import webcolors
import urllib.parse
import logging
import math
import redis
from dotenv import load_dotenv
import os
import json
from bson import json_util

# Load .env file
root_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(root_dir)  
dotenv_path = os.path.join(parent_dir, '.env')
print(dotenv_path)
load_dotenv(dotenv_path)

# Set up logging
root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Set up the DB
MONGO_URI = os.getenv('MONGO_URI')   
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD') 
client = MongoClient(MONGO_URI.replace("<password>", MONGO_PASSWORD))  
db = client.productwatcher  
handbags = db.handbags  

# Set up Redis for caching
r = redis.Redis.from_url(os.getenv('CACHE_REDIS_URL'))

def get_from_cache_or_db(query):
    # Create a unique key for this query
    query_key = str(query)

    # # Try to get the result from the cache
    # result = r.get(query_key)

    # if result is not None:
    #     # If the result is in the cache, return it
    #     return json.loads(result, object_hook=json_util.object_hook)
    # else:
        # Convert all string values in the query to case-insensitive regex
    for key, value in query.items():
            if isinstance(value, str):
                query[key] = {"$regex": f"^{value}$", "$options": "i"}

        # If the result is not in the cache, get it from the database
    result = list(handbags.find(query))
        
        # # Store the result in the cache, with an expiration time of 1 hour (3600 seconds)
        # r.set(query_key, json.dumps(result, default=json_util.default), ex=3600)
    return result


def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2     
        gd = (g_c - requested_color[1]) ** 2     
        bd = (b_c - requested_color[2]) ** 2     
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]
def estimate_price(brand, model, color=None, buying_price=None, days=None, material=None, condition=None, modelName=None, vintage=False):  
    print(f"Parameters: brand={brand}, model={model}, color={color}, buying_price={buying_price}, days={days}, material={material}, condition={condition}, modelName={modelName}, vintage={vintage}")

    # Create a unique key for this query
    query_key = f"{brand}_{model}"

    # General request
    max_time_to_sell = days 

    same_brand_model_general = get_from_cache_or_db({"collection": {"$regex": f"{brand} {model}", "$options": "i"}})
    same_brand_model_general_prices = [bag['price']['cents']/100 for bag in same_brand_model_general]
    avg_price_same_brand_model_general = statistics.mean(same_brand_model_general_prices) if same_brand_model_general_prices else 0 
       
    same_brand_model = get_from_cache_or_db({         
        "collection": {"$regex": f"{brand} {model}", "$options": "i"},         
        "timeToSell": {"$lte": max_time_to_sell}     
    })

    same_brand_model_prices = [bag['price']['cents']/100 for bag in same_brand_model]     

    if same_brand_model_prices:  
        lower_bound = np.percentile(same_brand_model_prices, 1)     
        upper_bound = np.percentile(same_brand_model_prices, 99)  
        filtered_prices = [price for price in same_brand_model_prices if lower_bound <= price <= upper_bound]     
        avg_price_same_brand_model = statistics.mean(filtered_prices) if filtered_prices else 0 
    else:
        print("No prices found for the same brand and model. Setting related variables to 0.")
        lower_bound = 0
        upper_bound = 0
        avg_price_same_brand_model = 0
        
    # Specific request 
    # same_brand_model_request_general = []  
    
    # Additional options
    additional_options = {}
    if color not in [None, "None", ""]:
        color = urllib.parse.unquote(color)
        # Check if the color string contains a slash
        if "/" in color:
            print(f"Color {color} contains a slash. Ignoring color option.")
        else:
            try:
                requested_color = webcolors.name_to_rgb(color)
                closest_named_color = closest_color(requested_color)
                additional_options["color"] = {"$regex": closest_named_color, "$options": "i"}
            except ValueError:
                print(f"Color {color} not recognized. Ignoring color option.")
    if material not in [None, ""]:
        additional_options["material"] = {"$regex": f"^{material}$", "$options": "i"}
    if condition not in [None, ""]:
        additional_options["condition.label"] = {"$regex": f"^{condition}$", "$options": "i"}
    if modelName not in [None, ""]:
        additional_options["modelName"] = {"$regex": f"^{modelName}$", "$options": "i"}
    if vintage is not None:
        additional_options["vintage"] = vintage


    same_brand_model_request = get_from_cache_or_db({        
        "collection": {"$regex": f"{brand} {model}", "$options": "i"},         
        **additional_options,        
        "timeToSell": {"$lte": max_time_to_sell}     
    })  



    same_brand_model_request_for_period = [bag for bag in same_brand_model_request if bag['timeToSell'] <= max_time_to_sell]    
    same_brand_model_request_for_period_prices = [bag['price']['cents']/100 for bag in same_brand_model_request if bag['timeToSell'] <= max_time_to_sell]
    if same_brand_model_request_for_period_prices:
        lower_bound = np.percentile(same_brand_model_request_for_period_prices, 1)
        upper_bound = np.percentile(same_brand_model_request_for_period_prices, 99)
        filtered_prices = [price for price in same_brand_model_request_for_period_prices if lower_bound <= price <= upper_bound]
        avg_price_same_brand_model_request_for_period = statistics.mean(filtered_prices) if filtered_prices else 0
    else:
        print("No prices found for the same brand and model with the specific request for the period. Setting related variables to 0.")
        lower_bound = 0
        upper_bound = 0
        avg_price_same_brand_model_request_for_period = 0

    rec_price_all = avg_price_same_brand_model * 0.9 
    profit_all = rec_price_all - buying_price
    rec_price_request = avg_price_same_brand_model_request_for_period * 0.9

    if avg_price_same_brand_model_request_for_period != 0:
        profit_request = rec_price_request - buying_price
    else:
        profit_request = 0

    print(f"Number of bags: {int(len(same_brand_model_general))}")
    print(f"Average price: {int(round(avg_price_same_brand_model_general))}€")
    print(f"Average price - for the period: {int(round(avg_price_same_brand_model))}€")
    print(f"Recommended price - all: {int(round(rec_price_all))}€")
    print(f"Profit- all: {int(round(profit_all))}€")
    print(f"")

    # print(f"Number of bags - specific request: {int(len(same_brand_model_request_general))}")
    # print(f"Average price - specific request: {int(round(avg_price_same_brand_model_request))}€")
    print(f"Number of bags - specific request - for the period: {int(len(same_brand_model_request_for_period))}")
    print(f"Average price - specific request - for the period: {int(round(avg_price_same_brand_model_request_for_period))}€")
    print(f"Recommended price - specific request: {int(round(rec_price_request))}€")
    print(f"Profit - specific request: {int(round(profit_request))}€")
    print(f"")

    result = {        
        "Number of bags": int(len(same_brand_model_general)),        
        "Average price": int(round(avg_price_same_brand_model_general)),        
        "Average price - for the period": int(round(avg_price_same_brand_model)),     
        "Recommended price - all": int(round(rec_price_all)),        
        "Profit- all": int(round(profit_all)),
        # "Number of bags - specific request": int(len(same_brand_model_request_general)),                      
        "Number of bags - specific request - for the period": int(len(same_brand_model_request_for_period)),
        # "Average price - specific request": int(round(avg_price_same_brand_model_request)), 
        "Average price - specific request - for the period": int(round(avg_price_same_brand_model_request_for_period)), 
        "Recommended price - specific request - for the period": int(round(rec_price_request)),        
        "Profit - specific request - for the period": int(round(profit_request)), 
    }

    return result




if __name__ == "__main__":
    result = estimate_price(
        brand="Hermes", 
        model="Kelly", 
        color="None", 
        buying_price=2000, 
        days=30, 
        # material="Leather", 
        # condition="Never worn", 
        # modelName="Kelly 32", 
        # vintage=True
    )
    print(result)