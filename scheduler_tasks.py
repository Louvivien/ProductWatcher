from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
import requests
import os
from config import products
from selenium import webdriver
from datetime import datetime, timedelta
import time
from webdriver_manager.chrome import ChromeDriverManager

def call_root():
    base_url = os.getenv('BASE_URL', 'http://localhost:5000')
    response = requests.get(base_url)
    if response.status_code == 200:
        print(f"Uptime check page up and running")

def call_product_detail():
    base_url = os.getenv('BASE_URL', 'http://localhost:5000')
    
    # Setup Chrome options to run in headless mode
    from selenium.webdriver.chrome.options import Options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    
    
    driver = webdriver.Chrome(ChromeDriverManager().install())  # This line will handle the driver download
    try:
        for product in products:
            brand = product['brand']
            model = product['model']
            try:
                driver.get(f"{base_url}/product_detail/{brand}/{model}")
                print(f"Before sleep at {datetime.now().time()}")
                time.sleep(1200)  # Delay for 20 minutes
                print(f"After sleep at {datetime.now().time()}")
            except Exception as e:
                print(f"Error while trying to access product detail page for {brand} {model}: {e}")
    except Exception as e:
        print(f"Error while initializing the webdriver: {e}")
    finally:
        driver.quit()


# Call the function once before adding it to the scheduler


scheduler = BackgroundScheduler()
scheduler.add_job(call_root, 'interval', minutes=13)
scheduler.add_job(call_product_detail, DateTrigger(run_date=datetime.now()))
scheduler.add_job(call_product_detail, 'interval', hours=1, start_date=datetime.now() + timedelta(hours=1))
