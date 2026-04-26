from bs4 import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader

bs4_strainer = SoupStrainer(class_=("post-title","post-header","post-content"))
web_content = WebBaseLoader(
    web_path = "https://docs.langchain.com/langsmith/observability-quickstart",
    bs_kwargs = {"parse_only": bs4_strainer}
)

file_content = web_content.load()

print(file_content)

