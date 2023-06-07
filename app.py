# blocked when deployed


from flask import Flask, render_template, request
import requests
import re
import json
from flask_bootstrap import Bootstrap
import curlify
import logging
import sys
import socket
import urllib3
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter



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

# Define the retry strategy
retry_strategy = Retry(
    total=10,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["HEAD", "GET", "OPTIONS"]
)

adapter = HTTPAdapter(max_retries=retry_strategy)
http = urllib3.PoolManager(retries=retry_strategy)



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
    logging.info("response content: %s", response.content)
    proxies = [{'ip': line.split(':')[0].strip(), 'port': line.split(':')[1].strip()} for line in response.text.split('\n') if line]
    return proxies

proxies = get_proxies()


def is_proxy_working(proxy_ip, proxy_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)  
    try:
        logging.info(f"Attempting to connect to proxy {proxy_ip}:{proxy_port}...")
        sock.connect((proxy_ip, int(proxy_port)))
        sock.close()
        logging.info(f"Successfully connected to proxy {proxy_ip}:{proxy_port}")
        return True
    except Exception as e:
        logging.info(f"Failed to connect to proxy {proxy_ip}:{proxy_port}")
        return False

working_proxy = None
for proxy in proxies:
    if is_proxy_working(proxy["ip"], proxy["port"]):
        working_proxy = proxy
        break

if working_proxy is None:
    raise Exception("No working proxy found.")
else:
    logging.info(f"Working proxy found: {working_proxy['ip']}:{working_proxy['port']}")

    
def switch_proxy():
    '''
    Switches to a different proxy.

    This function iterates over the list of proxies and tries to connect to each one.
    If a connection is successful, it sets the working proxy to the current one and breaks the loop.
    If no working proxy is found, it raises an exception.

    Returns:
        None
    '''
    proxies = get_proxies()

    global working_proxy
    for proxy in proxies:
        if proxy != working_proxy and is_proxy_working(proxy["ip"], proxy["port"]):
            working_proxy = proxy
            break

    if working_proxy is None:
        raise Exception("No working proxy found.")
    else:
        logging.info(f"Switched to a new working proxy: {working_proxy['ip']}:{working_proxy['port']}")

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


    # Set up the proxy
    proxies = {
        'http': f"http://{working_proxy['ip']}:{working_proxy['port'].strip()}",
        'https': f"http://{working_proxy['ip']}:{working_proxy['port'].strip()}",
    }

    for _ in range(10):
        product_name = product_name.replace(' ', '%20')
        response = None  

        try:
            response = requests.get('https://stockx.com/fr-fr/search?s=' + product_name, headers=headers, proxies=proxies, verify=False, timeout=10)

            curl_command = curlify.to_curl(response.request)
            proxy_string = f"-x http://{working_proxy['ip']}:{working_proxy['port'].strip()}"
            curl_command_with_proxy = f"{curl_command} {proxy_string}"
            logging.info("Curl command with proxy: %s", curl_command_with_proxy)
            logging.info("Response content: %s", response.content)

            if "captcha-error" in response.text or "<h1>Access Denied</h1>" in response.text or "Enable JavaScript and cookies" in response.text:
                logging.info("Captcha error or Access Denied detected, switching proxy...")
                switch_proxy()
                continue
        except requests.exceptions.HTTPError as errh:
            logging.error("Http Error:", errh)
        except requests.exceptions.ProxyError:
            logging.error("Proxy error occurred, switching proxy...")
            logging.error("Failed request details: %s", curlify.to_curl(response.request) if response else "No response received")
            logging.error("Response content: %s", response.content if response else "No response received")
            switch_proxy()
            continue
        except requests.exceptions.ConnectionError as errc:
            logging.error("Error Connecting: %s", errc)
            logging.error("Failed request details: %s", curlify.to_curl(response.request) if response else "No response received")
        except requests.exceptions.Timeout as errt:
            logging.error("Timeout Error:", errt)
        except requests.exceptions.RequestException as err:
            logging.error("Something went wrong", err)

        if response is None:
            return None, "All requests failed due to proxy errors."
        
        # rest of your code here

        
        
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

        logging.info("edges : %s", edges)

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
    result = search_stockx(product_name)
    if result is None:
        return render_template('error.html', message="All requests failed due to proxy errors.", edges=[])  # pass an empty list for edges
    else:
        edges, debug_info = result
        return render_template('home.html', edges=edges if edges else [], debug_info=debug_info)


if __name__ == '__main__':
    app.run(debug=True)
