from pymongo import MongoClient  
import sys

import os  

import statistics  

from dotenv import load_dotenv  

import numpy as np  

import webcolors
import urllib.parse
import logging


# Set up logging
root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)




def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2     
        gd = (g_c - requested_color[1]) ** 2     
        bd = (b_c - requested_color[2]) ** 2     
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def estimate_price(brand, model, color, buying_price, days):  
    root_dir = os.path.dirname(os.path.abspath(__file__))

    parent_dir = os.path.dirname(root_dir)

    dotenv_path = os.path.join(parent_dir, '.env')

    load_dotenv(dotenv_path)  

    MONGO_URI = os.getenv('MONGO_URI')   

    MONGO_PASSWORD = os.getenv('MONGO_PASSWORD') 

    client = MongoClient(MONGO_URI.replace("<password>", MONGO_PASSWORD))  

    db = client.productwatcher  

    handbags = db.handbags  

    max_time_to_sell = days 

    same_brand_model_general = list(handbags.find({"collection": {"$regex": f"{brand} {model}", "$options": "i"}})) 

    same_brand_model_general_prices = [bag['price']['cents']/100 for bag in same_brand_model_general]

    
    avg_price_same_brand_model_general = statistics.mean(same_brand_model_general_prices) if same_brand_model_general_prices else 0 
    
    same_brand_model_color_general = []  

    # Unquote the color string
    color = urllib.parse.unquote(color)

    # Check if the color string contains a slash
    if "/" in color:
        logging.info(f"Color {color} contains a slash. Setting all color-related variables to 0.")
        closest_named_color = "0"
        avg_price_same_brand_model_color_general = 0
        avg_price_same_brand_model_color = 0
        rec_price_color = 0 
        profit_color = 0
    else:
        try:
            requested_color = webcolors.name_to_rgb(color)
            closest_named_color = closest_color(requested_color)
        except ValueError:
            logging.info(f"Color {color} not recognized. Setting all color-related variables to 0.")
            closest_named_color = "0"
            avg_price_same_brand_model_color_general = 0
            avg_price_same_brand_model_color = 0
            rec_price_color = 0 
            profit_color = 0



    if closest_named_color != "0": 
        same_brand_model_color_general = list(handbags.find({  
            "collection": {"$regex": f"{brand} {model}", "$options": "i"},  
            "colors.all.name": {"$regex": closest_named_color, "$options": "i"}  
        }))  
        same_brand_model_color_general_prices = [bag['price']['cents']/100 for bag in same_brand_model_color_general] 
        avg_price_same_brand_model_color_general = statistics.mean(same_brand_model_color_general_prices) if same_brand_model_color_general_prices and len(same_brand_model_color_general) != 0 else 0
    else:
        logging.info(f"Closest named color {closest_named_color} is not a string. Setting all color-related variables to 0.")
        avg_price_same_brand_model_color_general = 0
        avg_price_same_brand_model_color = 0
        rec_price_color = 0 
        profit_color = 0


    same_brand_model = list(handbags.find({         
            "collection": {"$regex": f"{brand} {model}", "$options": "i"},         
            "timeToSell": {"$lte": max_time_to_sell}     
        })) 

    same_brand_model_prices = [bag['price']['cents']/100 for bag in same_brand_model]     

    if same_brand_model_prices:  # Add this check
        lower_bound = np.percentile(same_brand_model_prices, 1)     
        upper_bound = np.percentile(same_brand_model_prices, 99)  
        filtered_prices = [price for price in same_brand_model_prices if lower_bound <= price <= upper_bound]     
        avg_price_same_brand_model = statistics.mean(filtered_prices) if filtered_prices else 0 
    else:
        logging.info("No prices found for the same brand and model. Setting related variables to 0.")
        lower_bound = 0
        upper_bound = 0
        avg_price_same_brand_model = 0


    same_brand_model_color = list(handbags.find({        
            "collection": {"$regex": f"{brand} {model}", "$options": "i"},         
            "colors.all.name": {"$regex": color, "$options": "i"},         
            "timeToSell": {"$lte": max_time_to_sell}     
        }))  
    

    same_brand_model_color_prices = [bag['price']['cents']/100 for bag in same_brand_model_color]
    if same_brand_model_color_prices:
        lower_bound = np.percentile(same_brand_model_color_prices, 1)
        upper_bound = np.percentile(same_brand_model_color_prices, 99) 
        filtered_prices = [price for price in same_brand_model_color_prices if lower_bound <= price <= upper_bound]
        avg_price_same_brand_model_color = statistics.mean(filtered_prices) if filtered_prices and len(same_brand_model_color) != 0 else 0
    else:
        logging.info("No prices found for the same brand and model color. Setting related variables to 0.")
        lower_bound = 0
        upper_bound = 0
        avg_price_same_brand_model_color = 0
        profit_color = 0 

    rec_price_all = avg_price_same_brand_model * 0.9 

    profit_all = rec_price_all - buying_price

    rec_price_color = avg_price_same_brand_model_color * 0.9
  
    profit_color = rec_price_color - buying_price
    


    logging.info(f"Number of bags same brand same model: {len(same_brand_model_general)}")
    logging.info(f"Number of bags same brand same model same color: {len(same_brand_model_color_general)}")  
    logging.info(f"")
    logging.info(f"General average price for same brand same model: {avg_price_same_brand_model_general}€")
    logging.info(f"General average price for same brand same model same color: {avg_price_same_brand_model_color_general}€")    
    logging.info(f"")
    logging.info(f"Recommended price (based on all bags): {rec_price_all}€, Profit: {profit_all}€")
    logging.info(f"Recommended price (bags same color): {rec_price_color}€, Profit: {profit_color}€")

    return {        
            "Number of bags": len(same_brand_model_general),        
            "Number of bags - color": len(same_brand_model_color_general),        
            "Average price": avg_price_same_brand_model_general,        
            "Average price - color": avg_price_same_brand_model_color_general,        
            "Average price - for the period": avg_price_same_brand_model,        
            "Average price - color - for the period": avg_price_same_brand_model_color,        
            "Recommended price - all": rec_price_all,        
            "Profit- all": profit_all,        
            "Recommended price - color": rec_price_color,        
            "Profit - color": profit_color    
        }