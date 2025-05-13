import requests
import base64

class OllamaClient:
    """
    Client for interacting with the Ollama API.
    """
    def __init__(self, base_url):
        self.base_url = base_url

    def list_models(self):
        """
        Fetches the list of available models from the Ollama API.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", verify=False)
            response.raise_for_status()
            model_name = []
            data = response.json()
            for model in data['models']:
                model_name.append(model['name'])
            return model_name
        except requests.RequestException as e:
            print(f"Error listing models: {e}")
            return []

    def generate_text(self, model, prompt):
        """
        Generates text using the specified Ollama model.
        """
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, headers=headers, verify=False)
            response.raise_for_status()
            data = response.json()
            return data['response']
        except requests.RequestException as e:
            print(f"Error generating text: {e}")
            return None

    def generate_text_with_images(self, model, prompt, images):
        """
        Generates text using the specified Ollama model with images.
        """
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "images": images
        }
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, headers=headers, verify=False)
            response.raise_for_status()
            data = response.json()
            return data['response']
        except requests.RequestException as e:
            print(f"Error generating text with images: {e}")
            return None
