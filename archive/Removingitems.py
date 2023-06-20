from pymongo import MongoClient
from dotenv import load_dotenv
import os
import re

# Load .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# MongoDB setup
MONGO_URI = os.getenv('MONGO_URI')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')
client = MongoClient(MONGO_URI.replace("<password>", MONGO_PASSWORD))
db = client.productwatcher

# Get all collection names
collection_names = db.list_collection_names()

total_collections = len(collection_names)
print(f'Total collections: {total_collections}')

# Define the words to match in the 'name' field
words = ['wallet', 'jacket', 'watch', 'pants', 'shoe']
pattern = '|'.join(words)  # Create a pattern string like 'wallet|jacket|watch|pants|shoe'
regex = re.compile(pattern, re.IGNORECASE)  # Create a regex object, case-insensitive

for i, collection_name in enumerate(collection_names, start=1):
    collection = db[collection_name]
    
    # Delete all documents in the collection where 'name' field contains any of the specified words
    result = collection.delete_many({'name': regex})
    
    print(f'Deleted {result.deleted_count} documents from collection {i} of {total_collections}: {collection_name}')

print('Deletion complete')
