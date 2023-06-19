
import logging
import sys
import requests
import json
import curlify

    




# Set up logging
root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)



def search_vestiaire(brand, model):
    url = 'https://search.vestiairecollective.com/v1/product/search'
    headers = {
  'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
  'x-correlation-id': 'ac0d9fb0-2082-4b8e-a4fc-5ae43124c437',
  'sec-ch-ua-mobile': '?0',
  'x-datadog-origin': 'rum',
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
  'x-datadog-sampling-priority': '1',
  'Content-Type': 'application/json',
  'Accept': 'application/json, text/plain, */*',
  'Referer': 'https://fr.vestiairecollective.com/',
  'x-usertoken': 'anonymous-f3d761e0-98c7-48f0-a6f2-85bac75b9125',
  'x-datadog-parent-id': '8660110310529178523',
  'x-datadog-trace-id': '2824068943479274331',
  'sec-ch-ua-platform': '"macOS"',
  'Cookie': '__cf_bm=bjIcuIchzNacvWkSlOvUExxulmCAbvPsaRKX3nQfPiQ-1686133140-0-ASdq1dUNzx5rtqX9mPYoe3Mi7Z4xsmneXOoO8dDV2qnQZyboxtujXVWjGx9z/nVc7XlJ/F1Pq57KUeJ2PLTn6Pw=; _cfuvid=MwnWu_bAJ1wW7wax40HU1RMcKlSS1vEcvcP.b5TRClI-1686131820523-0-604800000; __cflb=04dToTDVtDoa5B61C23N844SspREcYmcBrFVgbKUvT'
}
    data = {
  "pagination": {
    "offset": 0,
    "limit": 48
  },
  "fields": [
    "name",
    "description",
    "brand",
    "model",
    "country",
    "price",
    "discount",
    "link",
    "sold",
    "likes",
    "editorPicks",
    "shouldBeGone",
    "seller",
    "directShipping",
    "local",
    "pictures",
    "colors",
    "size",
    "stock",
    "universeId"
  ],
  "facets": {
    "fields": [
      "brand",
      "universe",
      "country",
      "stock",
      "color",
      "categoryLvl0",
      "priceRange",
      "price",
      "condition",
      "region",
      "editorPicks",
      "watchMechanism",
      "discount",
      "sold",
      "directShippingEligible",
      "directShippingCountries",
      "localCountries",
      "sellerBadge",
      "isOfficialStore",
      "materialLvl0",
      "size0",
      "size1",
      "size2",
      "size3",
      "size4",
      "size5",
      "size6",
      "size7",
      "size8",
      "size9",
      "size10",
      "size11",
      "size12",
      "size13",
      "size14",
      "size15",
      "size16",
      "size17",
      "size18",
      "size19",
      "size20",
      "size21",
      "size22",
      "size23",
      "dealEligible"
    ],
    "stats": [
      "price"
    ]
  },
  "q": brand + " " + model,
  "sortBy": "relevance",
  "filters": { },
 
  "locale": {
    "country": "FR",
    "currency": "EUR",
    "language": "fr",
    "sizeType": "FR"
  },
  "mySizes": None
}

  
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        print(response.content)
        data = json.loads(response.content)
        queries = data['items']

        #     # Navigate to the 'results' key
        #     queries = data['props']['pageProps']['req']['appContext']['states']['query']['value']['queries']

        #     # Access the 5th query directly
        #     query = queries[4]['state']['data']['browse']['results']
        #     edges = query['edges']  # return the list of edges
        # logging.info("vestiaire : %s", queries)

        # for query in queries:
        #     query['collection'] = brand + " " + model
        #     handbags.insert_one(query)
        return queries, None
    else:
        return None

def search_stockx(brand, model):
    product_name = f"{brand} {model}"  
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



    for _ in range(10):
        product_name = product_name.replace(' ', '%20')
        response = None  

        try:
            response = requests.get('https://stockx.com/api/browse?_search=' + product_name, headers=headers,timeout=10)

            # curl_command = curlify.to_curl(response.request)
            # logging.info("curl_command: %s", curl_command)
            # logging.info("Response content: %s", response.content)

            if "captcha-error" in response.text or "<h1>Access Denied</h1>" in response.text or "Enable JavaScript and cookies" in response.text:
                                return None, logging.error("Captcha error or Access Denied detected, switching proxy?")

        except requests.exceptions.HTTPError as errh:
            logging.error("Http Error:", errh)
        except requests.exceptions.ProxyError:
            logging.error("Proxy error occurred, switching proxy...")
            logging.error("Failed request details: %s", curlify.to_curl(response.request) if response else "No response received")
            logging.error("Response content: %s", response.content if response else "No response received")

        except requests.exceptions.ConnectionError as errc:
            logging.error("Error Connecting: %s", errc)
            logging.error("Failed request details: %s", curlify.to_curl(response.request) if response else "No response received")
        except requests.exceptions.Timeout as errt:
            logging.error("Timeout Error:", errt)
        except requests.exceptions.RequestException as err:
            logging.error("Something went wrong", err)

        if response is None:
            return None, "All requests failed due to proxy errors."
        # else:
        #     queries, debug_info = stockx_result
        #     for query in queries:
        #         query['collection'] = brand + " " + model
        #         handbags.insert_one(query)
        #     return queries, None

        
        
        # # Find the script tag with the id '__NEXT_DATA__'
        # match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', response.text, re.DOTALL)

        # edges = None
        # if match:
        #     data_str = match.group(1)
        data = json.loads(response.content)
        queries = data['Products']
        # queries = queries[1]

        #     # Navigate to the 'results' key
        #     queries = data['props']['pageProps']['req']['appContext']['states']['query']['value']['queries']

        #     # Access the 5th query directly
        #     query = queries[4]['state']['data']['browse']['results']
        #     edges = query['edges']  # return the list of edges
        # logging.info("stockx : %s", queries)

        # if not edges:
        #     return None, curlify.to_curl(response.request)
        # else:
        return queries, None

