import streamlit as st
from ollama_client import OllamaClient 

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

        if st.button("Generate"):
            with st.spinner("Generating..."):
                generated_text = self.ollama_client.generate_text(selected_model, prompt)
                if generated_text:
                    st.write(generated_text)

def main():

    ollama_url = "https://localmodelscligpu.jollywater-dfbf95e7.australiaeast.azurecontainerapps.io"  # Replace with your Ollama API URL
    app = StreamlitApp(ollama_url)
    app.run()

if __name__ == "__main__":
    main()
