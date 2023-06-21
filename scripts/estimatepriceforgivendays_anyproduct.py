from pymongo import MongoClient
import os
import statistics
from dotenv import load_dotenv
# TO DO: Update database call like the one for all product

def estimate_price(brand, model, color, buying_price, days):

    # Load .env file
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print("Loading environment variables...")
    dotenv_path = os.path.join(root_dir, '.env')
    load_dotenv(dotenv_path)

    # MongoDB setup
    MONGO_URI = os.getenv('MONGO_URI')
    MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')
    client = MongoClient(MONGO_URI.replace("<password>", MONGO_PASSWORD))
    db = client.productwatcher
    handbags = db.handbags

    # Input parameters
    max_time_to_sell = days
    
    # Query for same brand and model without time constraint
    same_brand_model_general = list(handbags.find({"collection": {"$regex": f"{brand} {model}", "$options": "i"}}))

    # Calculate general average price for same brand and model
    same_brand_model_general_prices = [bag['price']['cents']/100 for bag in same_brand_model_general]
    avg_price_same_brand_model_general = statistics.mean(same_brand_model_general_prices) if same_brand_model_general_prices else 0

    # Query for same brand, model, and color without time constraint
    same_brand_model_color_general = list(handbags.find({
        "collection": {"$regex": f"{brand} {model}", "$options": "i"},
        "colors.all.name": {"$regex": color, "$options": "i"}
    }))

    # Calculate general average price for same brand, model, and color
    same_brand_model_color_general_prices = [bag['price']['cents']/100 for bag in same_brand_model_color_general]
    avg_price_same_brand_model_color_general = statistics.mean(same_brand_model_color_general_prices) if same_brand_model_color_general_prices else 0

    # Query for same brand and model sold within max_time_to_sell
    same_brand_model = list(handbags.find({
        "collection": {"$regex": f"{brand} {model}", "$options": "i"},
        "timeToSell": {"$lte": max_time_to_sell}
    }))
    # Calculate average price for same brand and model
    same_brand_model_prices = [bag['price']['cents']/100 for bag in same_brand_model]
    avg_price_same_brand_model = statistics.mean(same_brand_model_prices) if same_brand_model_prices else 0

    # Query for same brand, model, and color sold within max_time_to_sell
    same_brand_model_color = list(handbags.find({
        "collection": {"$regex": f"{brand} {model}", "$options": "i"},
        "colors.all.name": {"$regex": color, "$options": "i"},
        "timeToSell": {"$lte": max_time_to_sell}
    }))
    
    # Calculate average price for same brand, model, and color
    same_brand_model_color_prices = [bag['price']['cents']/100 for bag in same_brand_model_color]
    avg_price_same_brand_model_color = statistics.mean(same_brand_model_color_prices) if same_brand_model_color_prices else 0

    # Calculate recommended price and profit for all bags
    rec_price_all = avg_price_same_brand_model * 0.9  # Assuming we want to price 10% below average for quick sale
    profit_all = rec_price_all - buying_price

    # Calculate recommended price and profit for bags of same color
    rec_price_color = avg_price_same_brand_model_color * 0.9  # Assuming we want to price 10% below average for quick sale
    profit_color = rec_price_color - buying_price

    # Print results
    print(f"Number of bags same brand same model: {len(same_brand_model_general)}")
    print(f"Number of bags same brand same model same color: {len(same_brand_model_color_general)}")
    print(f"")
    print(f"General average price for same brand same model: {avg_price_same_brand_model_general}€")
    print(f"General average price for same brand same model same color: {avg_price_same_brand_model_color_general}€")
    print(f"")
    print(f"Recommended price (based on all bags): {rec_price_all}€, Profit: {profit_all}€")
    print(f"Recommended price (bags same color): {rec_price_color}€, Profit: {profit_color}€")

    # Return the results as a dictionary
    return {
            "Number of bags": len(same_brand_model_general),
            "Number of bags - color": len(same_brand_model_color_general),
            "Average price": avg_price_same_brand_model_general,
            "Average price - color": avg_price_same_brand_model_color_general,
            "Recommended price - all": rec_price_all,
            "Profit- all": profit_all,
            "Recommended price - color": rec_price_color,
            "Profit - color": profit_color
        }