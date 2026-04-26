from dotenv import load_dotenv
import os
load_dotenv()
from langchain_tavily import TavilySearch

def tavily_search(query):
    tavily_tool = TavilySearch(max_results=10)
    tavily_response = tavily_tool.invoke(query, search_depth="advanced")
    serialized_tavily_response = "\n\n".join(f"Source: {doc.get('url','no sources')}\n\nContent: {doc.get('contents','no contents')}" for doc in tavily_response["results"])
    return serialized_tavily_response
