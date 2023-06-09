# blocked when deployed



from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import httpx
import json
from parsel import Selector
from nested_lookup import nested_lookup
import logging
import sys



app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Set up logging
root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# create HTTPX client with headers that resemble a web browser
client = httpx.AsyncClient(
    http2=True,
    follow_redirects=True,
    timeout=60.0, 
    limits=httpx.Limits(max_connections=100),
    headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    },
)

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

def parse_nextjs(html: str) -> dict:
    """extract nextjs cache from page"""
    selector = Selector(html)
    data = selector.css("script#__NEXT_DATA__::text").get()
    if not data:
        data = selector.css("script[data-name=query]::text").get()
        data = data.split("=", 1)[-1].strip().strip(";")
    data = json.loads(data)
    logging.info("data", data)
    return data


async def scrape_product(product_name: str) -> dict:
    """scrape a single stockx product page for product data"""
    product_name = product_name.replace(' ', '%20')
    url = 'https://stockx.com/api/browse?_search=' + product_name
    try:
        response = await client.get(url)
        response.raise_for_status()
    except httpx.PoolTimeout:
        logging.info(f"Timeout occurred while trying to connect to {url}")
        return {}
    except httpx.HTTPStatusError as exc:
        logging.info(f"An HTTP error occurred while making a request to {url}: {exc}")
        return {}

    data = parse_nextjs(response.text)
    # extract all products datasets from page cache
    # products = nested_lookup("product", data)
    # find the current product dataset
    # try:
    #     product = next(p for p in products if p.get("urlKey") in str(response.url))
    # except StopIteration:
    #     raise ValueError("Could not find product dataset in page cache", response)
    # Navigate to the 'results' key
    # queries = data['props']['pageProps']['req']['appContext']['states']['query']['value']['queries']
    # query = queries[4]['state']['data']['browse']['results']
    # edges = query['edges']
    logging.info("data :", data)   

    return data




@app.get("/", response_class=HTMLResponse)
async def product_list(request: Request):
    return templates.TemplateResponse("product_list.html", {"request": request, "products": products})

@app.get("/product/{product_name}", response_class=HTMLResponse)
async def product_detail(request: Request, product_name: str):
    result = await scrape_product(product_name)
    return templates.TemplateResponse("home.html", {"request": request, "data": result if result else [], "debug_info": None})


