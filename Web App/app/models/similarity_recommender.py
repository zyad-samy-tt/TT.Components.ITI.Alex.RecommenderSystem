from pinecone import Pinecone
from config import Config

pc = Pinecone(api_key=Config.PINECONE_API_KEY)

index_name = Config.PINECONE_INDEX_NAME
index = pc.Index(index_name)

def get_similar_products(product_id, num_results=5):
    query_response = index.query(id = product_id , top_k=num_results)
    results = query_response['matches']
    ids = [result.id for result in results]
    return results