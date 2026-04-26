from fastmcp import FastMCP

mcp = FastMCP("web-server")

@mcp.tool()
def web_server(query:str):
    """search for the information online based on query"""
    web_answer = f'Stimulate information for: {query}'
    return web_answer

