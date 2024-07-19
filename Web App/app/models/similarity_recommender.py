from pinecone import Pinecone
from config import Config

pc = Pinecone(api_key=Config.PINECONE_API_KEY)

models = {
    'roberta': Config.PINECONE_INDEX_NAME1,
    'fclip': Config.PINECONE_INDEX_NAME2,
}

# index_name = Config.PINECONE_INDEX_NAME
# index = pc.Index(index_name)


def get_index_for_model(model_name):
    index_name = models.get(model_name)
    if not index_name:
        raise ValueError(f"Model '{model_name}' is not recognized.")
    return pc.Index(index_name)


# def get_similar_products(product_id, num_results=5):
#     query_response = index.query(id=product_id, top_k=num_results)
#     results = query_response['matches']
#     ids = [result.id for result in results]
#     return results


def get_similar_products(index, product_id, num_results=5):
    query_response = index.query(id=product_id, top_k=num_results)
    results = query_response.get('matches', [])
    return results
