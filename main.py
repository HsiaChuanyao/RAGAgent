import uuid

from dotenv import load_dotenv
load_dotenv(override=True, verbose=True)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
load_dotenv(override=True, verbose=True)
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from mcp_client import llm, main
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
)
texts = text_spliter.split_documents(documents=file_content)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = InMemoryVectorStore.from_documents(documents = texts,embedding=embeddings)

@tool
def tavily_search(query):
    """tavily search
    Args:
        query (str): query
    Returns:
        return the answer to the query
    """
    tavily_tool = TavilySearch(max_results=10)
    tavily_response = tavily_tool.invoke(query, search_depth="advanced")
    serialized_tavily_response = "\n\n".join(f"Source: {doc.get('url','no sources')}\n\nContent: {doc.get('contents','no contents')}" for doc in tavily_response["results"])
    return serialized_tavily_response



@tool
def retrieve_content(query):
    """retrieve the content of the given query"""
    retriever = vectorstore.similarly_search(query, k=2)
    serialized_retriever = "\n\n".join(f"Source: {doc.get('source')}\n\nContent: {doc.page_content}" for doc in retriever["results"])
    return serialized_retriever


def rag_agent(user_id, thread_id, query):
    """rag agent"""
    agent = create_agent(
        model=llm,
        tools= [tavily_search,main],
    )

    if not thread_id:
        thread_id = str(uuid.uuid4())


    if not user_id:
        user_id = str(uuid.uuid4())


    if not query:
        print(f'Please provide a query and try again.')

    full_prompt = prompt_template.format(query=query)
    init_state = {"messages":[HumanMessage(content=full_prompt)]}
    config = {"configurable":{"thread_id":thread_id, "user_id":user_id,"checkpointer": checkpointer }, }

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
        break