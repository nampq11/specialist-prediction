import requests


def get_embeddings(query: str):
    base_url = "http://localhost:8000/retrieval/embeddings"

    body = {
        "type": "default",
        "sentences": [
            query
        ]
    }
    response = requests.post(base_url, json=body)

    return response.json()['data'][0]


if __name__ == "__main__":
    print(get_embeddings('beenhj vien cho ray'))