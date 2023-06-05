
# Works locally but not when deployed


from flask import Flask, render_template, request
import requests
from flask import Flask, render_template, request
import requests
import re
import json
from flask_bootstrap import Bootstrap
import curlify


import requests
from requests.exceptions import ProxyError



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


def get_proxies():
    url = 'https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=FR&ssl=FR&anonymity=FR&_ga=2.134393777.1587810449.1684520809-1182041995.1684520809'
    headers = {
        'authority': 'api.proxyscrape.com',
        'accept': 'text/plain, */*; q=0.01',
        'accept-language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
        'cache-control': 'no-cache',
        'origin': 'https://proxyscrape.com',
        'pragma': 'no-cache',
        'referer': 'https://proxyscrape.com',
        'sec-ch-ua': '"Google Chrome";v="113", "Chromium";v="113", "Not-A.Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
    }
    response = requests.get(url, headers=headers)
    proxies = [{'ip': line.split(':')[0], 'port': line.split(':')[1]} for line in response.text.split('\n') if line]
    return proxies

def search_stockx(product_name):
    headers = {
                'authority': 'stockx.com',
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'accept-language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
                'cache-control': 'no-cache',
                'pragma': 'no-cache',
                'sec-ch-ua': '"Google Chrome";v="113", "Chromium";v="113", "Not-A.Brand";v="24"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"macOS"',
                'sec-fetch-dest': 'document',
                'sec-fetch-mode': 'navigate',
                'sec-fetch-site': 'none',
                'sec-fetch-user': '?1',
                'upgrade-insecure-requests': '1',
                'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
                'cookie': '_ga=GA1.1.91221291.1685454860; _ga_TYYSNQDG4W=GS1.1.1685951323.5.1.1685951428.0.0.0; _ga=GA1.1.91221291.1685454860; _ga_TYYSNQDG4W=GS1.1.1685957993.5.0.1685957993.0.0.0; stockx_device_id=cc820410-0abf-4c5c-9cbd-c88ed825bde8; _pxvid=6a92d6e2-fef1-11ed-aab3-727262436761; pxcts=6a92ea80-fef1-11ed-aab3-727262436761; __pxvid=6b052045-fef1-11ed-bbb5-0242ac120002; __ssid=a5905c2e1cd358405c9c42541e6ca61; rskxRunCookie=0; rCookie=7kqmhuji4sr4x30pngwkyuliacajc9; language_code=fr; stockx_selected_locale=fr; stockx_selected_currency=EUR; stockx_selected_region=FR; stockx_dismiss_modal=true; stockx_dismiss_modal_set=2023-05-30T13%3A54%3A05.807Z; stockx_dismiss_modal_expiration=2024-05-30T13%3A54%3A05.807Z; OptanonAlertBoxClosed=2023-05-30T13:54:20.108Z; ajs_anonymous_id=b5f08b3e-25ad-4732-a345-2aadf229b02c; _gcl_au=1.1.204760704.1685454861; _pin_unauth=dWlkPU5tRXlZalJrWWpNdE16STROeTAwTW1VeExXRTJaVFF0TVdKaU9XWTJZV1F5WWpGbA; rbuid=rbos-a2ff6e52-f94f-4e51-9f6c-2cbcf3ff26ca; QuantumMetricUserID=5738dfd2590c2758c3b98c83cb61a175; _rdt_uuid=1685454864958.171c16d1-5f0e-4860-91bb-68ba5ca7ecbf; IR_gbd=stockx.com; __pdst=395dbe3ea0114ac0a1f240fc17faa765; stockx_preferred_market_activity=sales; _ga=GA1.1.91221291.1685454860; _gid=GA1.2.1582542843.1685951321; ajs_user_id=24d3b066-037e-11ee-8a06-12f12be0eb51; stockx_session=a5820759-f850-430e-849a-9ae3acc4b559; _ga_TYYSNQDG4W=GS1.1.1685955057.4.1.1685955140.0.0.0; _ga=GA1.2.91221291.1685454860; OptanonConsent=isGpcEnabled=0&datestamp=Mon+Jun+05+2023+10%3A52%3A34+GMT%2B0200+(Central+European+Summer+Time)&version=202211.2.0&isIABGlobal=false&hosts=&consentId=b43eb36e-9005-4168-869c-fc95d1beb3ed&interactionCount=2&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0005%3A1%2CC0004%3A1%2CC0003%3A1&AwaitingReconsent=false&geolocation=FR%3BIDF; stockx_product_visits=1; IR_9060=1685955159865%7C4294847%7C1685955159865%7C%7C; IR_PI=7cb47d44-fef1-11ed-823f-2b97337fbd0e%7C1686041559865; forterToken=e3337fe8c3234ff78ac6ef95bc078e5e_1685955157234__UDF43-m4_13ck; lastRskxRun=1685955163055; stockx_homepage=accessories; _uetsid=655cb570037511ee90476b7fd1eaef07; _uetvid=7a3ade40fef111ed82d4419ea07e8fc9; _pxde=19550942aa8095600c88ca96e76877f3fd5b06669520e67748ab2850ff00a9d4:eyJ0aW1lc3RhbXAiOjE2ODU5NTc5ODc1ODMsImZfa2IiOjB9; __cf_bm=0oS862M9UxIq3aqOYcD2wzOfgNQiZUwwkbb4SXM9lxI-1685957987-0-AU9waJgfCeWlY464BONwFeTcnharxPGbCL453Su75GprLtAKYbPfXzQdHXdWXhWoi9+8m/1ksQVR30zhnZUmDFM=; QuantumMetricSessionID=6a800d446625a1be05728d4604ee8bdd; stockx_session_id=047aab36-447b-4ca6-8e33-efb230a13f30; _dd_s=',
            }

    proxies = get_proxies()
    working_proxy = None
    for proxy in proxies:
        try:
            proxy_dict = {
                'http': f'http://{proxy["ip"]}:{proxy["port"]}',
                'https': f'https://{proxy["ip"]}:{proxy["port"]}',
            }
            product_name = product_name.replace(' ', '%20')
            response = requests.get('https://stockx.com/fr-fr/search?s=' + product_name, headers=headers, proxies=proxy_dict)
            # If the request is successful, break the loop
            working_proxy = proxy
            break
        except ProxyError:
            # If the request fails, continue to the next proxy
            continue

    if working_proxy is None:
        raise Exception("No working proxy found.")
    else:
        print(f"Working proxy found: {working_proxy['ip']}:{working_proxy['port']}")


    # Find the script tag with the id '__NEXT_DATA__'
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', response.text, re.DOTALL)

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
        return None, curlify.to_curl(response.request)
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

