import os
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
