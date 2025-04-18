import requests
import time
import aiohttp
import asyncio
from typing import List, Union


def get_embedding(query: str):
    base_url = "http://localhost:8000/retrieval/embeddings"

    body = {
        "type": "default",
        "sentences": [
            query
        ]
    }
    response = requests.post(base_url, json=body)

    return response.json()['data'][0]

def get_embeddings(query: list[str]):
    base_url = "http://localhost:8000/retrieval/embeddings"

    body = {
        "type": "default",
        "sentences": query
    }
    response = requests.post(base_url, json=body)

    return response.json()['data']

def get_specialist_id(query: str, max_retries=3, retry_delay=1):
    base_url = "https://nampham1106-search.hf.space/specialist/predict"
    
    body = {
        "query": query
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(base_url, json=body, timeout=30)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return response.json()['data']['ids']
        except (requests.RequestException, KeyError, ValueError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise Exception(f"Failed to get specialist ID after {max_retries} attempts: {str(e)}")
        
async def get_specialist_ids(queries: list[str], max_retries=3, retry_delay=1):
    base_url = "http://localhost:8000/specialist/batch_predict"
    
    body = {
        "queries": queries
    }

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(base_url, json=body, timeout=30) as response:
                    response.raise_for_status()  # Raise exception for 4XX/5XX responses
                    result = await response.json()
                    
                    # Extract both ids and logs from the response
                    predictions = []
                    for item in result['data']:
                        predictions.append({
                            'ids': item['ids'],
                            'logs': item['logs']
                        })
                    return predictions
        except (aiohttp.ClientError, KeyError, ValueError) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            raise Exception(f"Failed to get specialist IDs after {max_retries} attempts: {str(e)}")

if __name__ == "__main__":
    # print(get_embeddings('beenhj vien cho ray'))
    print(get_specialist_id('đau đầu'))