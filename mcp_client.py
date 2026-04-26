from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0)

async def main(query: str):
    async with MultiServerMCPClient(
            {
                "web_server":{
                    "transport":"stdio",
                    "command": "python",
                    "args":["/Users/reganhsia/Desktop/PythonProject/rag_agent/servers/web_server.py"]
                }
            }
    ) as client:
        tool = await client.get_tools(query)
        agent = create_agent(llm, tool)
        client_response = await agent.ainvoke({"input":query})
        print(client_response.get("outputs","error"))




