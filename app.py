from flask import Flask, render_template, request
import undetected_chromedriver as uc
import re
import json
from flask_bootstrap import Bootstrap
import curlify


app = Flask(__name__)
Bootstrap(app)


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
    options = uc.ChromeOptions()
    options.headless = True
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = uc.Chrome(options=options)

    product_name = product_name.replace(' ', '%20')
    driver.get('https://stockx.com/fr-fr/search?s=' + product_name)

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

    print("edges :", edges)  # Add this line to print the edges variable

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
