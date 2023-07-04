# Improvements
# load list of products dynamically


import requests
import random
import time
import threading
from datetime import datetime, timedelta
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the absolute paths
working_proxies_file = os.path.join(script_dir, "workingproxies.txt")
proxy_list_file = os.path.join(script_dir, "proxylist.txt")
blacklist_file = os.path.join(script_dir, "blacklist.txt")




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

def get_proxies(url):
    try:
        response = requests.get(url)
        print(f"Retrieved proxies from {url}")
        return [line.strip() for line in response.text.split("\n") if line.strip() != ""]
    except Exception as e:
        print(f"Error retrieving proxies from {url}: {e}")
        if "Failed to establish a new connection" in str(e):
            return None  
        return []


def find_working_proxy(proxies):
    url = "http://httpbin.org/ip"
    for proxy in proxies:
        try:
            response = requests.get(url, proxies={"http": proxy, "https": proxy}, timeout=30)
            if response.status_code == 200:
                # print(f"Found working proxy: {proxy}")
                return proxy
        except Exception as e:
            # print(f"Error with proxy {proxy}: {e}")
            # print(f"")
            return None
    print("No working proxy found")
    return None

def search_stockx(product, proxy):
    product_name = product['brand'] + ' ' + product['model']
    product_name = product_name.replace(' ', '%20')
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
    proxies = {
        'https': proxy,
    }
    try:
        response = requests.get('https://stockx.com/api/browse?_search=' +  product_name, headers=headers, proxies=proxies, timeout=10)
        if "captcha-error" in response.text or "<h1>Access Denied</h1>" in response.text or "Enable JavaScript and cookies" in response.text:
            # print(f"Proxy {proxy} failed to bypass Cloudflare. Adding to blacklist.")
            with open(blacklist_file, "a") as file:
                file.write(proxy + "\n")
            return False
        else:
            print(f"Proxy {proxy} successfully bypassed Cloudflare 😃")
            print(f"")
            return True
    except Exception as e:
        # print(f"Error with proxy {proxy}: {e}")
        # print(f"")
        if "Failed to establish a new connection" in str(e):
            return None  # Return None if this specific error occurs
        return False

def check_working_proxies():
    while True:
        with open(working_proxies_file, "r") as file:
            working_proxies = [proxy.strip() for proxy in file.readlines()]
        for proxy in working_proxies:
            if not search_stockx(random.choice(products), proxy):
                # print(f"Removing non-working proxy: {proxy}")
                # print(f"")
                working_proxies.remove(proxy)
        with open(working_proxies_file, "w") as file:
            for proxy in working_proxies:
                file.write(proxy + "\n")
        print("Finished checking working proxies. Waiting for 30 minutes before next check.")
        print(f"")
        time.sleep(1800) 

        
def main():
    threading.Thread(target=check_working_proxies).start()
    while True:
        with open(proxy_list_file, "r") as file:
            proxy_list = [line.strip() for line in file.readlines()]
        if not proxy_list:
            print("No more proxy URLs left in the list.")
            break
        # Choose a random URL that is not marked as useless
        url = random.choice([url for url in proxy_list if not url.startswith("USELESS:")])
        proxies = get_proxies(url)
        if proxies is None:  # If get_proxies returned None, skip this iteration
            continue
        working_proxies_found = False
        for proxy in proxies:
            with open(blacklist_file, "r") as file:
                blacklist = file.readlines()
            if proxy + "\n" in blacklist:
                # print(f"Skipping blacklisted proxy: {proxy}")
                # print(f"")
                continue
            with open(working_proxies_file, "r") as file:
                working_proxies = file.readlines()
            if search_stockx(random.choice(products), proxy):
                with open(working_proxies_file, "a") as file:
                    if proxy + "\n" not in working_proxies:
                        print(f"Adding new working proxy: {proxy}")
                        print(f"")
                        file.write(proxy + "\n")
                        working_proxies_found = True
            if search_stockx(random.choice(products), proxy) is None:
                continue  
        if not working_proxies_found:
            print(f"No working proxies found in {url}. Marking as useless.")
            print(f"")
            proxy_list.remove(url)
            proxy_list.append("USELESS:" + url)  # Mark the URL as useless
            with open(proxy_list_file, "w") as file:
                for proxy_url in proxy_list:
                    file.write(proxy_url + "\n")



if __name__ == "__main__":
    main()
