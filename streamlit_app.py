import streamlit as st
from ollama_client import OllamaClient
import base64

class StreamlitApp:
    """
    Streamlit application for interacting with Ollama models.
    """
    def __init__(self, ollama_url):
        self.ollama_client = OllamaClient(ollama_url)
        self.available_models = []

    def load_models(self):
        """Loads the available models from the Ollama API."""
        self.available_models = self.ollama_client.list_models()
        st.session_state.models = self.available_models
        return self.available_models

    def run(self):
        """Runs the Streamlit application."""
        st.title("Self Hosted Ollama Model Playground")
        if "models" not in st.session_state:
            self.load_models()
        if not st.session_state.models:
            st.warning("No models found. Check your Ollama API endpoint OR Refresh this Page")
            return
        selected_model = st.selectbox("Select Model:", st.session_state.models)
        prompt = st.text_area("Enter Prompt:", "Tell me a story about a cat.")
        
        uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        images_base64 = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                base64_str = base64.b64encode(bytes_data).decode("utf-8")
                images_base64.append(base64_str)
        
        if st.button("Generate"):
            with st.spinner("Generating..."):
                if images_base64:
                    generated_text = self.ollama_client.generate_text_with_images(selected_model, prompt, images_base64)
                else:
                    generated_text = self.ollama_client.generate_text(selected_model, prompt)
                if generated_text:
                    st.write(generated_text)

def main():
    ollama_url = "https://localmodelscligpu.jollywater-dfbf95e7.australiaeast.azurecontainerapps.io" # Replace with your Ollama API URL
    app = StreamlitApp(ollama_url)
    app.run()

if __name__ == "__main__":
    main()
