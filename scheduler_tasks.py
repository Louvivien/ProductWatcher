from apscheduler.schedulers.background import BackgroundScheduler
import requests
import os
from config import products
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from webdriver_manager.chrome import ChromeDriverManager

def call_root():
    base_url = os.getenv('BASE_URL', 'http://localhost:5000')
    response = requests.get(base_url)
    if response.status_code == 200:
        print(f"Uptime check page up and running")

def call_product_detail():
    base_url = os.getenv('BASE_URL', 'http://localhost:5000')
    driver = webdriver.Chrome(ChromeDriverManager().install())  # This line will handle the driver download
    try:
        for product in products:
            brand = product['brand']
            model = product['model']
            try:
                driver.get(f"{base_url}/product_detail/{brand}/{model}")
                WebDriverWait(driver, 1200).until(expected_conditions.presence_of_element_located((By.ID, 'title-search'))) 
            except Exception as e:
                print(f"Error while trying to access product detail page for {brand} {model}: {e}")
    except Exception as e:
        print(f"Error while initializing the webdriver: {e}")
    finally:
        driver.quit()

# Call the function once before adding it to the scheduler
call_product_detail()

scheduler = BackgroundScheduler()
scheduler.add_job(call_root, 'interval', minutes=13)
scheduler.add_job(call_product_detail, 'interval', hours=1)
