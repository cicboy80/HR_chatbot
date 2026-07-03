import os

# Provide dummy values so the app modules import without real credentials.
# load_dotenv() does not override variables that are already set, and the
# unit tests never make real API calls.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")
os.environ.setdefault("WEAVIATE_URL", "https://test-cluster.example")
os.environ.setdefault("WEAVIATE_API_KEY", "test-weaviate-key")
