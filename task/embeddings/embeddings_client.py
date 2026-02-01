import json

import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


class DialEmbeddingsClient:
    """Client for generating embeddings using DIAL API"""
    
    def __init__(self, deployment_name: str, api_key: str):
        """
        Initialize the embeddings client
        
        Args:
            deployment_name: Name of the DIAL embedding model deployment
            api_key: API key for authentication
        """
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")
        
        self._endpoint = DIAL_EMBEDDINGS.format(model=deployment_name)
        self._api_key = api_key
    
    def get_embeddings(self, texts: list[str], dimensions: int = 1536) -> dict[int, list[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to generate embeddings for
            dimensions: Dimension size for embeddings (default: 1536)
        
        Returns:
            Dictionary mapping index to embedding vector
        """
        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json"
        }
        
        request_data = {
            "input": texts,
            "dimensions": dimensions
        }
        
        response = requests.post(url=self._endpoint, headers=headers, json=request_data, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            embeddings_data = data.get("data", [])
            
            # Build dictionary with index as key and embedding as value
            result = {}
            for item in embeddings_data:
                index = item.get("index")
                embedding = item.get("embedding")
                result[index] = embedding
            
            return result
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
