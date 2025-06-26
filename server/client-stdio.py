import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys

async def main():
    # Define server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],  # Make sure this points to your PDF extractor server
    )

    # Connect to the server
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Call the PDF Extractor Tool
            pdf_path = sys.argv[1]
            pages = sys.argv[2] if len(sys.argv) > 2 else ""
            result = await session.call_tool(
                "extract_pdf_contents",
                arguments={"pdf_path": pdf_path, "pages": pages}
            )
            print(f"Extracted text:\n{result.content[0].text}")

if __name__ == "__main__":
    asyncio.run(main())