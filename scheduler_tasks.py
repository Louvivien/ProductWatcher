from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
import requests
import os
from config import products
from selenium import webdriver
from datetime import datetime, timedelta
import time
from webdriver_manager.chrome import ChromeDriverManager
import subprocess
import random
from threading import Lock


# Create a global lock
lock = Lock()

# Keep track of the subprocess
proxies_process = None

def call_root():
    random_delay = random.randint(1, 3)
    time.sleep(random_delay)  
    base_url = os.getenv('BASE_URL', 'http://localhost:5000')
    response = requests.get(base_url)
    if response.status_code == 200:
        print(f"Uptime check page up and running")
        
def call_product_detail():
    global proxies_process
    # Acquire the lock before running the function
    with lock:
        # If the proxies script is running, stop it
        if proxies_process is not None:
            proxies_process.terminate()
            proxies_process = None
            
        random_delay = random.randint(1, 3)
        time.sleep(random_delay)  
        base_url = os.getenv('BASE_URL', 'http://localhost:5000')
        
        # Setup Chrome options to run in headless mode and optimize memory usage
        from selenium.webdriver.chrome.options import Options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")  # Disable GPU hardware acceleration
        chrome_options.add_argument("--disable-software-rasterizer")  # Disable software rasterizer
        chrome_options.add_argument("--disk-cache-size=10000000")  # Limit disk cache size to 10MB
        prefs = {"profile.managed_default_content_settings.images": 2}  # Disable images
        chrome_options.add_experimental_option("prefs", prefs)

        # Set mobile emulation
        mobile_emulation = {"deviceName": "iPhone X"}  # You can change this to any other supported device
        chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)
        
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)  # This line will handle the driver download
        try:
            total_products = len(products)
            print(f"Total number of products to process: {total_products}")
            for index, product in enumerate(products, start=1):
                brand = product['brand']
                model = product['model']
                try:
                    driver.get(f"{base_url}/product_detail/{brand}/{model}")
                    print(f"Processing product Buy {index} of {total_products}  😃")
                    print(f"Before sleep at {datetime.now().time()}")
                    time.sleep(1200+random.randint(1, 3))  # Delay for 20 minutes
                    print(f"After sleep at {datetime.now().time()}")
                    print(f"Processing product Sell {index} of {total_products}  😃")
                    driver.get(f"{base_url}/sales_stats/{brand}/{model}")
                    time.sleep(90 +random.randint(1, 3))


                except Exception as e:
                    print(f"Error while trying to access product detail page for {brand} {model}: {e}")
        except Exception as e:
            print(f"Error while initializing the webdriver: {e}")
        finally:
            driver.quit()

        
        

def run_proxies_script():
    global proxies_process
    # Only run the function if the lock is not acquired
    if not lock.locked():

        try:
            subprocess.Popen(["python", "./scripts/scrapers/proxies.py"])
            print("Started proxies.py script.")
        except Exception as e:
            print(f"Error while trying to run proxies.py script: {e}")



# Call the function once before adding it to the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(call_root, 'interval', minutes=13)

scheduler.add_job(call_product_detail, DateTrigger(run_date=datetime.now()))
scheduler.add_job(run_proxies_script, DateTrigger(run_date=datetime.now()))

scheduler.add_job(call_product_detail, 'interval', hours=3, start_date=datetime.now() + timedelta(hours=1))
scheduler.add_job(run_proxies_script, 'interval', hours=1, start_date=datetime.now() + timedelta(minutes=5))