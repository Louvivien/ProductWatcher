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




# Load .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

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





# Function to extract models from eBay
def get_models(brand, model):
    try:
        conn = http.client.HTTPSConnection("www.ebay.com")
        brand_encoded = quote(brand)
        model_encoded = quote(model)
        headers = {
                'authority': 'www.ebay.com',
                'accept': '*/*',
                'cache-control': 'no-cache',
                'pragma': 'no-cache',
                'referer': "https://www.ebay.com/sch/i.html?_fsrp=1&_from=R40&_nkw={brand}+{model}&_sacat=0&LH_Sold=1&rt=nc&_oaa=1&_dcat=169291".format(brand=brand, model=model),
                'sec-ch-ua-full-version': '"114.0.5735.106"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-model': '""',
                'sec-ch-ua-platform-version': '"12.6.5"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'x-requested-with': 'XMLHttpRequest',
                'User-Agent': ua.random,
                'Cookie': '__deba=OH9EbZd2CoWoYUNpbimbaoLyC_7t7Wv8oNC_87lDaMxvQ_HHkvsvqUSySNteDQSQWiP3ms0kLC5xqOQEeY2siogkSZp4GNO8aVvCTEaxLKHC5xHjSmsDUvAlQQsigp4P2HUUX7nXo10fSYckCM7qWA==; __uzma=497fe911-5b45-475c-abde-d88a75158a0d; __uzmb=1686237210; __uzmc=2542512492180; __uzmd=1686301844; __uzme=0223; __uzmf=7f6000c29712ca-d5b1-4784-a289-f4c5838bf289168623721003164634252-9d24dd2e3ba6d10b124; ak_bmsc=26C214431EF86207F0366187658499F0~000000000000000000000000000000~YAAQZ4QVAo4is26IAQAAcCRsnxTsI6byoozSOTbxVvWVFV6c9FzEpx3BFiU5Hi3LnIpqGEwQ9zMjdZz1kWANzpxvFRwopyZ9F+FEqLP3YC9UJvPkGB3xApJqKA3USaEdGDu2xYeoWt8Waeo9s09NcGsNhx8gfKnB5isymvaMZCjYK7Lr3dolGBOgf2/0VREzO2ZSPj2yrfIxIudxriYjGHxLBFfQniyDqH8H0bT+mDa31CQ1vr3CfWXdDDmp9rxlbbRggG06ZIhu8hVTrAkkqMtH/c8q/+smtRSgccIGD2augEFsU1xI4F1MwG+64Cz+L2dJsbSKzzgJLw7KbbMwOn4IJMvbIuOnhpAyV0eTOrM2rOaT0M/Voom3GUAlCz/rCNqAWTbg1zHXxcTd; bm_sv=5AC5A3C6ABB14408F00C111A04F46B82~YAAQZ4QVAo8is26IAQAAcCRsnxSVCLXRb+quQ7NMW+wI/7k8Z14ShGTOaOxPd9LQxZNxZsa6ACxrFo/VNXekw30B4eZPZLb+N7Ee9+ABk6v6LVAAo4N+usIfavW+3ADtP/F4CS+Ius1Lbt3m+qaxJTGWWyWL3j7rmWGZlOywOFxL8iNJSpVtoS6kfJl/TAEALJjvEIbi+pOJamZYpsFMuhDCh99Ot4NV0m0tO9FZZWX2+ly11al+ESgewbiwXg==~1; dp1=bu1p/QEBfX0BAX19AQA**68455394^pbf/%23a000e0000001000200000066642014^bl/FR68455394^; ebay=%5Ejs%3D1%5Esbf%3D%23000000%5E; nonsession=BAQAAAYhF58hiAAaAADMABWZkIBQ3NTAxNQDKACBoRVOUOWI5MWUxYWQxODgwYTQ0ZTAxZDJjMmUzZmZmODBiY2QAywABZILznDURRbiFYqWwMTMb+G4a3Yu+wyLuyA**; ns1=BAQAAAYhF58hiAAaAANgAU2ZkIBRjNjl8NjAxXjE2ODYyMzcyMTgwNzNeXjFeM3wyfDV8NHw3fDEwfDQyfDQzfDExXl5eNF4zXjEyXjEyXjJeMV4xXjBeMV4wXjFeNjQ0MjQ1OTA3NXq1/TdTTuiGs141oU0ocy2Wm1lV; s=CgADuAIRkg0RUMwZodHRwczovL3d3dy5lYmF5LmNvbS9zY2gvaS5odG1sP19mc3JwPTEmX2Zyb209UjQwJl9ua3c9aGVybWVzK2V2ZWx5bmUmX3NhY2F0PTAmTEhfU29sZD0xJnJ0PW5jJl9vYWE9MSZfZGNhdD0xNjkyOTEjaXRlbTUyOWQyODdjYjUHAPgAIGSDRFQ5YjkxZTFhZDE4ODBhNDRlMDFkMmMyZTNmZmY4MGJjZABW5tY*',
}
        url = f"/sch/ajax/refine?no_encode_refine_params=1&_fsrp=1&rt=nc&_from=R40&_nkw={brand_encoded}+{model_encoded}&_sacat=0&&_oaa=1&LH_Sold=1&_aspectname=aspect-Model&modules=SEARCH_REFINEMENTS_MODEL_V2%3Afa"
        conn.request("GET", url, headers=headers)


        response = conn.getresponse()
        response_content = response.read().decode()
        data = json.loads(response_content)
        # logging.info(curl_from_response(response, "GET", "https://www.ebay.com" + url, headers, ""))

        data = data['group'][0]
        # logging.info("data: %s", data)

        
        models = []
        for group in data['entries']:
            if 'fieldId' in group and group['fieldId'] == 'aspect-Model':
                for entry in group['entries']:
                    if entry['paramValue'] != '!':
                        models.append(entry['label']['textSpans'][0]['text'])

                logging.info(f"models: {models}")



        logging.info(f"Retrieved {len(models)} models for {brand} {model}")
        return models
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

# Function to get sold products from vestiairecollective
def get_sold_products(brand, model):
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
                "pagination": {"offset": offset, "limit": 100},
                "fields": ["name", "description", "brand", "model", "country", "price", "discount", "link", "sold", "likes",
                           "seller", "directShipping", "local", "pictures", "colors", "size", "stock", "universeId"],
                "facets": {
                    "fields": ["brand", "universe", "country", "stock", "color", "categoryLvl0", "priceRange", "price",
                               "condition", "region", "watchMechanism", "discount", "sold", "localCountries",
                               "sellerBadge", "isOfficialStore", "materialLvl0", "dealEligible"],
                    "stats": ["price"]},
                "q": f"{brand} {model}",
                "sortBy": "relevance",
                "filters": {"sold": ["1"]},
                "locale": {"country": "FR", "currency": "EUR", "language": "en", "sizeType": "FR"}
            }
            conn.request("POST", url, body=json.dumps(data), headers=headers)
            response = conn.getresponse()
            data = json.loads(response.read().decode())

            items = data['items']

            current_page = data['paginationStats']['offset'] // 100 + 1
            total_pages = data['paginationStats']['totalPages']
            logging.info(f"Retrieving products for page {current_page}/{total_pages}...")

            # Add model_name field to each item
            for item in items:
                item['model_name'] = 'unknown'  # default value
                if item['sold']:
                    sold_products.append(item)

            if total_pages == 0:
                break
            if total_pages == current_page:
                break
            offset += 100
            # Random sleep to avoid being detected as a bot
            time.sleep(randint(1,3))
        logging.info(f"Retrieved {len(sold_products)} sold products for {brand} {model}")
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
                "unit": data['data'].get('unit', None)
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
        logging.info(f"Getting models...")
        models = get_models(brand, model)
        # Create collection
        logging.info(f"Creating collection...")
        collection = db[brand + " " + model]

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
main("Chanel", "Double Flap")
