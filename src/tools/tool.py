from urllib import response
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Correct spelling here 👇
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Initialize tool
tool = TavilySearch(api_key=tavily_api_key)

# Run a query
response = tool.run("What is the capital of France?")
print(response["results"][0]["content"])
