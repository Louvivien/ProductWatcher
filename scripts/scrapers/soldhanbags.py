from pymongo import MongoClient
import json
from dotenv import load_dotenv
import os
import time
import random
import logging
from random import randint
from fake_useragent import UserAgent
from datetime import datetime
from urllib.parse import quote



import http.client

import redis




# Load .env file
root_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(root_dir)
parent_dir = os.path.dirname(parent_dir)

dotenv_path = os.path.join(parent_dir, '.env')
print(dotenv_path)
load_dotenv(dotenv_path)

# Set up Redis for caching
r = redis.Redis.from_url(os.getenv('CACHE_REDIS_URL'))

# MongoDB setup
MONGO_URI = os.getenv('MONGO_URI')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')
client = MongoClient(MONGO_URI.replace("<password>", MONGO_PASSWORD))
db = client.productwatcher

# Setup logging
# logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


# Setup UserAgent
ua = UserAgent()



# Function to generate a cURL command from a requests response object
def curl_from_response(response, method, url, headers, body):
    command = "curl -X {method} -H {headers} '{uri}'"
    if body:  # Only add -d option if there is a payload
        command = "curl -X {method} -H {headers} -d '{data}' '{uri}'"
    headers_str = ['"{0}: {1}"'.format(k, v) for k, v in headers.items()]
    headers_str = " -H ".join(headers_str)
    return command.format(method=method, headers=headers_str, data=body, uri=url)


# Function to get sold products from vestiairecollective
def get_sold_products(brand, model):
    # Create a unique key for this query
    query_key = f"sold_products_{brand}_{model}"

    # Try to get the result from the cache
    result = r.get(query_key)
    
    # if result is not None:
    #     # If the result is in the cache, return it
    #     return json.loads(result)
    
    try:
        conn = http.client.HTTPSConnection("search.vestiairecollective.com")
        brand_encoded = quote(brand)
        model_encoded = quote(model)
        headers = {'User-Agent': ua.random, 'Content-Type': 'application/json'}
        url = "/v1/product/search"
        sold_products = []
        offset = 0
        while True:
            data = {
                    "pagination": {
                        "offset": offset,
                        "limit": 100
                    },
                    "fields": [
                        "name",
                        "description",
                        "brand",
                        "model",
                        "condition",
                        "country",
                        "price",
                        "discount",
                        "link",
                        "sold",
                        "likes",
                        "seller",
                        "directShipping",
                        "local",
                        "pictures",
                        "colors",
                        "size",
                        "stock"
                    ],
                    "facets": {
                        "fields": [
                        "brand",
                        "universe",
                        "color",
                        "categoryLvl0",
                        "priceRange",
                        "price",
                        "condition",
                        "region",
                        "watchMechanism",
                        "discount",
                        "sold",
                        "sellerBadge",
                        "materialLvl0"
                        ],
                        "stats": [
                        "price"
                        ]
                    },
                    "q": f"{brand} {model}",
                    "sortBy": "relevance",
                    "filters": {
                        "catalogLinksWithoutLanguage": [
                        "/women-bags/handbags/"
                        ],
                        "universe.id": [
                        "1"
                        ],
                        "sold": [
                        "1"
                        ]
                    },
                    "locale": {
                        "country": "FR",
                        "currency": "EUR",
                        "language": "en",
                        "sizeType": "FR"
                    }
                    }
            conn.request("POST", url, body=json.dumps(data), headers=headers)
            response = conn.getresponse()
            data = json.loads(response.read().decode())

            items = data['items']

            current_page = data['paginationStats']['offset'] // 100 + 1
            total_pages = data['paginationStats']['totalPages']
            logging.info(f"Retrieving products for page {current_page}/{total_pages}...")

            for item in items:
                    item['collection'] = brand + " " + model  
                    sold_products.append(item)

            if total_pages == 0:
                break
            if total_pages == current_page:
                break
            offset += 100
            # Random sleep to avoid being detected as a bot
            time.sleep(randint(1,3))
        logging.info(f"Retrieved {len(sold_products)} sold products for {brand} {model}")
      
        # # Store the result in the cache, with an expiration time of 30 minutes
        # r.set(query_key, json.dumps(sold_products), ex=1800)
  
        return sold_products
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

# Function to get additional product details
def get_product_details(product_id, retries=3):
    for i in range(retries):
        try:
            # Random sleep to avoid bot detection
            time.sleep(random.uniform(1, 3))

            conn = http.client.HTTPSConnection("apiv2.vestiairecollective.com")
            headers = {'User-Agent': ua.random}
            url = f"/products/{product_id}?isoCountry=FR&x-siteid=1&x-language=en&x-currency=EUR"
            conn.request("GET", url, headers=headers)
            response = conn.getresponse()
            response_data = response.read().decode()

            # Check if the response is a valid JSON string
            try:
                data = json.loads(response_data)
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON received for product {product_id}. Retrying ({i+1}/{retries})...")
                continue

            # Convert creationDate and soldDate to datetime objects
            creation_date = datetime.strptime(data['data']['creationDate'], "%Y-%m-%dT%H:%M:%SZ")
            sold_date = datetime.strptime(data['data']['soldDate'], "%Y-%m-%dT%H:%M:%SZ")

            # Calculate time to sell
            time_to_sell = sold_date - creation_date

            details = {
                "creationDate": creation_date,
                "soldDate": sold_date,
                "timeToSell": time_to_sell.days,  # Convert timedelta to number of days
                "measurements": data['data'].get('measurements', None),
                "measurementFormatted": data['data'].get('measurementFormatted', None),
                "unit": data['data'].get('unit', None),
                "material": data['data'].get('material', {}).get('name', None),
                "modelName": data['data'].get('model', {}).get('name', None),
                "vintage": "vintage" in data['data'].get('tags', []), 
                "color": data['data'].get('color', {}).get('name', None) 

            }


            logging.info(f"Retrieved details for product {product_id}. Time to sell: {time_to_sell.days} days")
            return details

        except Exception as e:
            logging.error(f"Failed to fetch details for product {product_id}. Retrying ({i+1}/{retries})...", exc_info=True)
            # Wait for a bit before retrying
            time.sleep(2)
    else:
        # If we've exhausted all retries and still failed, log a message and return
        logging.warning(f"Failed to fetch details for product {product_id} after {retries} attempts. Skipping...")


# Main function
def main(brand, model):
    try:
        # Get models
        # logging.info(f"Getting models...")
        # models = get_models(brand, model)
        collection = db["handbags"]

        # Get sold products
        logging.info(f"Getting sold products...")
        sold_products = get_sold_products(brand, model)
        # Add sold products to collection
        logging.info(f"Adding sold products to the database...")
        for product in sold_products:
            # Check if product already exists in the collection
            if collection.find_one({"id": product['id']}):
                logging.info(f"Product {product['id']} already exists in the database. Skipping...")
                continue
            # Get additional details
            logging.info(f"Getting additional details for product {product['id']}...")
            details = get_product_details(product['id'])
            if details is not None:  # Check if details is not None before updating
                # Add details to product
                product.update(details)
                # Add product to collection
                collection.insert_one(product)
                logging.info(f"Added product {product['id']} to the database")
            else:
                logging.warning(f"Could not retrieve details for product {product['id']}. Skipping...")
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

# Run the script
main("Hermes", "Kelly")
