import uuid

import dotenv
from dotenv import load_dotenv
import os

from langchain_core import documents
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(override=True, verbose=True)
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from mcp_client import llm, main
from tavily_tool import tavily_search
from webloader import file_content

prompt_template = ChatPromptTemplate.from_template(
    "You are one helpful agent to help user search the content of the given query"
    "You can use the tavily_search, web_server tools to finish the process."
    "You will return the answers from tools and max results are 10."
)


class CustomeAgent(AgentState):
    user_id:str
    thread_id:str
    query:str

checkpointer = InMemorySaver()
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    verbose = True
)
texts = text_spliter.split_documents(documents=file_content)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = InMemoryVectorStore.from_documents(documents = texts,embeddings=embeddings)

@tool("retriever", description="retrieve the content of the given query")
async def retrieve_content(query):
    """Retrieve the content of the given query"""
    retriever = vectorstore.similarly_search(query, k=2)
    serialized_retriever = "\n\n".join(f"Source: {doc.get('source')}\n\nContent: {doc.page_content}" for doc in retriever["results"])
    return serialized_retriever


def rag_agent(user_id, thread_id, query):
    agent = create_agent(
        model=llm,
        tools= [tavily_search,main],
        checkpoint= checkpointer,
    )

    if not thread_id:
        thread_id = str(uuid.uuid4())


    if not user_id:
        user_id = str(uuid.uuid4())


    if not query:
        print(f'Please provide a query and try again.')

    full_prompt = prompt_template.format(query=query)
    init_state = {"messages":[HumanMessage(content=full_prompt)]}
    config = {"configurable":{"thread_id":thread_id, "user_id":user_id}}

    agent_response = agent.invoke(init_state, config=config)
    return agent_response


if __name__ == "__main__":
    print(f'Hello, RAG Agent!')
    print("-" * 40)
    u_id = input("Please provide a user id or press enter to finish the agent: ")
    thread = input("Please provide a thread id or press enter to finish the agent: ")
    q = input("Please provide a query or press enter to finish the agent: ")
    while u_id and thread and q:
        try:
            response_output = rag_agent(user_id=u_id, thread_id=thread, query=q)
            print(response_output["messages"][-1].content)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            print("-"*40)