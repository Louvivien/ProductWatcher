from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# MongoDB setup
MONGO_URI = os.getenv('MONGO_URI')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')
client = MongoClient(MONGO_URI.replace("<password>", MONGO_PASSWORD))
db = client.productwatcher

# Get the list of collections in the database
collections = db.list_collection_names()

# Create a new collection 'handbags'
handbags = db['handbags']

print("Merging collections...")

# Loop through all collections
for collection_name in collections:
    collection = db[collection_name]
    
    # Get all documents from the collection
    documents = collection.find()
    
    # Insert the documents into the 'handbags' collection
    handbags.insert_many(documents)

print("Merging completed. The 'handbags' collection now contains all documents from the 16 collections.")