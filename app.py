from flask import Flask, render_template, request
import undetected_chromedriver as uc
import re
import json
from flask_bootstrap import Bootstrap
import curlify
import logging
import sys

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import TimeoutException



app = Flask(__name__)
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
    'Hermes Evelyn',
    'Hermes Kelly',
    'Hermes Birkin',
    'Hermes Pikotin',
    'Chanel Flap',
    'Chanel Double Flap',
    'Chanel Boy',
    'Chanel 19',
    'Chanel 2.55',
    'Chanel mini Flap',
    'Dior Lady Dior',
    'Dior Diorama',
    'Bottega Veneta Cassette',
    'Chanel V Stitch',
    'Louis Vuitton Capucines',
    'Louis Vuitton Twist Chain'
]

def search_stockx(product_name):
    logging.info('Loading undetected Chrome')

    driver = uc.Chrome(headless=True)
    driver.set_page_load_timeout(30)
    logging.info('Loaded Undetected chrome')

    product_name = product_name.replace(' ', '%20')
    driver.get('https://stockx.com/fr-fr/search?s=' + product_name)
    
    try:
        WebDriverWait(driver,15).until(EC.title_contains('StockX'))
    except TimeoutException:
        pass
    


    logging.info("response content: %s", driver.page_source)

    # Find the script tag with the id '__NEXT_DATA__'
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', driver.page_source, re.DOTALL)

    edges = None
    if match:
        data_str = match.group(1)
        data = json.loads(data_str)

        # Navigate to the 'results' key
        queries = data['props']['pageProps']['req']['appContext']['states']['query']['value']['queries']

        # Access the 5th query directly
        query = queries[4]['state']['data']['browse']['results']
        edges = query['edges']  # return the list of edges

    logging.info("edges : %s", edges)

    if not edges:
        return None, curlify.to_curl(driver.current_url)
    else:
        return edges, None

@app.route('/')
def product_list():
    if request.method == 'POST':
        new_product = request.form.get('new_product')
        if new_product:
            products.append(new_product)

    return render_template('product_list.html', products=products)

@app.route('/product/<product_name>')
def product_detail(product_name):
    edges, debug_info = search_stockx(product_name)
    return render_template('home.html', edges=edges if edges else [], debug_info=debug_info)

if __name__ == '__main__':
    app.run(debug=True)
