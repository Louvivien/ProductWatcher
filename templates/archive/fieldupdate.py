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

# Get all collection names
collection_names = db.list_collection_names()

total_collections = len(collection_names)
print(f'Total collections: {total_collections}')

for i, collection_name in enumerate(collection_names, start=1):
    collection = db[collection_name]
    
    # Get all documents in the collection
    documents = collection.find({})
    total_documents = collection.count_documents({})
    print(f'Updating collection {i} of {total_collections}: {collection_name}, {total_documents} documents to update')
    
    for j, document in enumerate(documents, start=1):
        # Update the document
        collection.update_one(
            {'_id': document['_id']},
            {
                '$rename': {
                    'universeId': 'collection_temp'  # Rename 'universeId' to 'collection_temp'
                }
            }
        )
        
        collection.update_one(
            {'_id': document['_id']},
            {
                '$set': {
                    'collection': collection_name  # Set the value of 'collection' to the collection name
                }
            }
        )
        
        collection.update_one(
            {'_id': document['_id']},
            {
                '$unset': {
                    'collection_temp': ""  # Remove the 'collection_temp' field
                }
            }
        )
        
        if j % 100 == 0:  # Print progress every 100 documents
            print(f'Updated {j} of {total_documents} documents in collection {collection_name}')

print('Update complete')