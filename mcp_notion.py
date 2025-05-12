"""
Notion MCP Read-Only Integration

This module implements a simplified read-only Model-Call Protocol (MCP) for Notion integration,
allowing AI assistants to retrieve data from Notion databases through a standardized interface.
"""

from mcp.server import Server
from mcp.types import (
    Resource, 
    Tool,
    TextContent,
    EmbeddedResource
)
import os
import json
import httpx
from typing import Any, Dict, List, Sequence, Optional
from dotenv import load_dotenv
from pathlib import Path
import logging
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('notion_mcp_readonly')

# Initialize server
server = Server("notion-reader")

class NotionReader:
    """Read-only client for interacting with the Notion API"""
    
    def __init__(self, api_key: Optional[str] = None, notion_version: str = "2022-06-28"):
        """
        Initialize the Notion read-only client
        
        Args:
            api_key: Notion API key (optional, defaults to env var)
            notion_version: Notion API version
        """
        # Find and load .env file from project root if API key not provided
        if api_key is None:
            project_root = Path(__file__).parent.parent
            env_path = project_root / '.env'
            if env_path.exists():
                load_dotenv(env_path)
            else:
                logger.warning(f"No .env file found at {env_path}, using environment variables")
        
        # Set API key
        self.api_key = api_key or os.getenv("NOTION_API_KEY")
        
        # Validate configuration
        if not self.api_key:
            raise ValueError("NOTION_API_KEY not found in environment variables or .env file")
        
        # API configuration
        self.notion_version = notion_version
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Notion-Version": self.notion_version
        }
        
        # Client for connection pooling
        self._client = None
    
    @property
    def client(self):
        """Get HTTP client with connection pooling"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def close(self):
        """Close HTTP client"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def query_database(self, database_id: str, filter_params: Optional[Dict] = None, 
                           sorts: Optional[List[Dict]] = None, page_size: int = 100, 
                           start_cursor: Optional[str] = None) -> Dict:
        """
        Query a Notion database
        
        Args:
            database_id: The ID of the database to query
            filter_params: Filters to apply to the query
            sorts: Sort orders to apply to the query
            page_size: Number of results per page
            start_cursor: Cursor for pagination
            
        Returns:
            The query response
        """
        query = {"page_size": page_size}
        
        if filter_params:
            query["filter"] = filter_params
            
        if sorts:
            query["sorts"] = sorts
        else:
            # Default sort by created time
            query["sorts"] = [{"timestamp": "created_time", "direction": "descending"}]
            
        if start_cursor:
            query["start_cursor"] = start_cursor
        
        response = await self.client.post(
            f"{self.base_url}/databases/{database_id}/query",
            headers=self.headers,
            json=query
        )
        response.raise_for_status()
        return response.json()
    
    async def get_page(self, page_id: str) -> Dict:
        """
        Get a page
        
        Args:
            page_id: The ID of the page
            
        Returns:
            The page
        """
        response = await self.client.get(
            f"{self.base_url}/pages/{page_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    async def get_block_children(self, block_id: str, page_size: int = 100,
                              start_cursor: Optional[str] = None) -> Dict:
        """
        Get children blocks of a block
        
        Args:
            block_id: The ID of the block
            page_size: Number of results per page
            start_cursor: Cursor for pagination
            
        Returns:
            The block children
        """
        params = {"page_size": page_size}
        if start_cursor:
            params["start_cursor"] = start_cursor
            
        response = await self.client.get(
            f"{self.base_url}/blocks/{block_id}/children",
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def get_database(self, database_id: str) -> Dict:
        """
        Get a database
        
        Args:
            database_id: The ID of the database
            
        Returns:
            The database
        """
        response = await self.client.get(
            f"{self.base_url}/databases/{database_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    async def search(self, query: str, filter_params: Optional[Dict] = None, 
                   sorts: Optional[List[Dict]] = None, page_size: int = 100) -> Dict:
        """
        Search Notion
        
        Args:
            query: The search query
            filter_params: Filters to apply to the search
            sorts: Sort orders to apply to the search
            page_size: Number of results per page
            
        Returns:
            The search response
        """
        search_params = {
            "query": query,
            "page_size": page_size
        }
        
        if filter_params:
            search_params["filter"] = filter_params
            
        if sorts:
            search_params["sort"] = sorts
        
        response = await self.client.post(
            f"{self.base_url}/search",
            headers=self.headers,
            json=search_params
        )
        response.raise_for_status()
        return response.json()

# MCP Server handlers
@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available read-only Notion tools"""
    return [
        Tool(
            name="list_databases",
            description="List available Notion databases",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_database_schema",
            description="Get the schema of a Notion database",
            inputSchema={
                "type": "object",
                "properties": {
                    "database_id": {
                        "type": "string",
                        "description": "The ID of the database"
                    }
                },
                "required": ["database_id"]
            }
        ),
        Tool(
            name="query_database",
            description="Query items from a Notion database",
            inputSchema={
                "type": "object",
                "properties": {
                    "database_id": {
                        "type": "string",
                        "description": "The ID of the database"
                    },
                    "filter": {
                        "type": "object",
                        "description": "Filters to apply to the query (optional)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of items to return (default: 100)"
                    }
                },
                "required": ["database_id"]
            }
        ),
        Tool(
            name="get_page_content",
            description="Get the content of a Notion page",
            inputSchema={
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "The ID of the page"
                    }
                },
                "required": ["page_id"]
            }
        ),
        Tool(
            name="search_notion",
            description="Search Notion for pages and databases",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "filter": {
                        "type": "object",
                        "description": "Filter by object type: 'page' or 'database' (optional)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of results to return (default: 100)"
                    }
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | EmbeddedResource]:
    """Handle tool calls for read-only Notion interaction"""
    reader = NotionReader()
    
    try:
        if name == "list_databases":
            # Search for databases
            result = await reader.search("", filter_params={"property": "object", "value": "database"})
            databases = result.get("results", [])
            
            formatted_dbs = []
            for db in databases:
                formatted_dbs.append({
                    "id": db["id"],
                    "title": db["title"][0]["plain_text"] if db["title"] else "Untitled",
                    "created_time": db["created_time"],
                    "last_edited_time": db["last_edited_time"]
                })
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps(formatted_dbs, indent=2)
                )
            ]
        
        elif name == "get_database_schema":
            database_id = arguments.get("database_id")
            if not database_id:
                raise ValueError("Database ID is required")
                
            database = await reader.get_database(database_id)
            
            schema = {
                "id": database["id"],
                "title": database["title"][0]["plain_text"] if database["title"] else "Untitled",
                "properties": database["properties"]
            }
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps(schema, indent=2)
                )
            ]
        
        elif name == "query_database":
            database_id = arguments.get("database_id")
            filter_params = arguments.get("filter")
            limit = int(arguments.get("limit", 100))
            
            if not database_id:
                raise ValueError("Database ID is required")
            
            # For pagination
            all_results = []
            next_cursor = None
            
            while len(all_results) < limit:
                page_size = min(100, limit - len(all_results))
                response = await reader.query_database(
                    database_id, 
                    filter_params=filter_params, 
                    page_size=page_size,
                    start_cursor=next_cursor
                )
                
                results = response.get("results", [])
                all_results.extend(results)
                
                next_cursor = response.get("next_cursor")
                if not next_cursor or not results:
                    break
            
            # Format the results more nicely
            formatted_results = []
            for item in all_results:
                properties = {}
                for prop_name, prop_value in item.get("properties", {}).items():
                    # Extract the actual value based on property type
                    if prop_value["type"] == "title":
                        properties[prop_name] = prop_value["title"][0]["plain_text"] if prop_value["title"] else ""
                    elif prop_value["type"] == "rich_text":
                        properties[prop_name] = prop_value["rich_text"][0]["plain_text"] if prop_value["rich_text"] else ""
                    elif prop_value["type"] == "number":
                        properties[prop_name] = prop_value["number"]
                    elif prop_value["type"] == "select":
                        properties[prop_name] = prop_value["select"]["name"] if prop_value["select"] else None
                    elif prop_value["type"] == "multi_select":
                        properties[prop_name] = [option["name"] for option in prop_value["multi_select"]]
                    elif prop_value["type"] == "date":
                        properties[prop_name] = prop_value["date"]["start"] if prop_value["date"] else None
                    elif prop_value["type"] == "checkbox":
                        properties[prop_name] = prop_value["checkbox"]
                    else:
                        # For other property types, just include the raw value
                        properties[prop_name] = prop_value
                
                formatted_results.append({
                    "id": item["id"],
                    "created_time": item["created_time"],
                    "last_edited_time": item["last_edited_time"],
                    "properties": properties
                })
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps(formatted_results, indent=2)
                )
            ]
        
        elif name == "get_page_content":
            page_id = arguments.get("page_id")
            if not page_id:
                raise ValueError("Page ID is required")
            
            # Get page metadata
            page = await reader.get_page(page_id)
            
            # Get page content (blocks)
            blocks_response = await reader.get_block_children(page_id)
            blocks = blocks_response.get("results", [])
            
            # Format page properties
            properties = {}
            for prop_name, prop_value in page.get("properties", {}).items():
                # Extract the actual value based on property type (similar to query_database)
                if prop_value["type"] == "title":
                    properties[prop_name] = prop_value["title"][0]["plain_text"] if prop_value["title"] else ""
                elif prop_value["type"] == "rich_text":
                    properties[prop_name] = prop_value["rich_text"][0]["plain_text"] if prop_value["rich_text"] else ""
                elif prop_value["type"] == "number":
                    properties[prop_name] = prop_value["number"]
                elif prop_value["type"] == "select":
                    properties[prop_name] = prop_value["select"]["name"] if prop_value["select"] else None
                elif prop_value["type"] == "multi_select":
                    properties[prop_name] = [option["name"] for option in prop_value["multi_select"]]
                elif prop_value["type"] == "date":
                    properties[prop_name] = prop_value["date"]["start"] if prop_value["date"] else None
                elif prop_value["type"] == "checkbox":
                    properties[prop_name] = prop_value["checkbox"]
                else:
                    # For other property types, just include the raw value
                    properties[prop_name] = prop_value
            
            # Format blocks (simplify block content)
            formatted_blocks = []
            for block in blocks:
                block_type = block["type"]
                block_content = {
                    "id": block["id"],
                    "type": block_type,
                    "created_time": block["created_time"],
                    "last_edited_time": block["last_edited_time"],
                }
                
                # Extract content based on block type
                if block_type == "paragraph":
                    block_content["text"] = "".join([text["plain_text"] for text in block["paragraph"]["rich_text"]])
                elif block_type == "heading_1":
                    block_content["text"] = "".join([text["plain_text"] for text in block["heading_1"]["rich_text"]])
                elif block_type == "heading_2":
                    block_content["text"] = "".join([text["plain_text"] for text in block["heading_2"]["rich_text"]])
                elif block_type == "heading_3":
                    block_content["text"] = "".join([text["plain_text"] for text in block["heading_3"]["rich_text"]])
                elif block_type == "bulleted_list_item":
                    block_content["text"] = "".join([text["plain_text"] for text in block["bulleted_list_item"]["rich_text"]])
                elif block_type == "numbered_list_item":
                    block_content["text"] = "".join([text["plain_text"] for text in block["numbered_list_item"]["rich_text"]])
                elif block_type == "to_do":
                    block_content["text"] = "".join([text["plain_text"] for text in block["to_do"]["rich_text"]])
                    block_content["checked"] = block["to_do"]["checked"]
                elif block_type == "toggle":
                    block_content["text"] = "".join([text["plain_text"] for text in block["toggle"]["rich_text"]])
                elif block_type == "code":
                    block_content["text"] = "".join([text["plain_text"] for text in block["code"]["rich_text"]])
                    block_content["language"] = block["code"]["language"]
                elif block_type == "image":
                    if "external" in block["image"]:
                        block_content["url"] = block["image"]["external"]["url"]
                    elif "file" in block["image"]:
                        block_content["url"] = block["image"]["file"]["url"]
                    else:
                        block_content["url"] = None
                # Add more block types as needed
                
                formatted_blocks.append(block_content)
            
            result = {
                "id": page["id"],
                "created_time": page["created_time"],
                "last_edited_time": page["last_edited_time"],
                "properties": properties,
                "content": formatted_blocks
            }
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )
            ]
        
        elif name == "search_notion":
            query = arguments.get("query", "")
            filter_params = arguments.get("filter")
            limit = int(arguments.get("limit", 100))
            
            result = await reader.search(query, filter_params=filter_params, page_size=limit)
            
            # Format search results
            formatted_results = []
            for item in result.get("results", []):
                obj_type = item["object"]
                
                if obj_type == "page":
                    title = ""
                    for prop in item.get("properties", {}).values():
                        if prop["type"] == "title" and prop["title"]:
                            title = "".join([text["plain_text"] for text in prop["title"]])
                            break
                    
                    formatted_results.append({
                        "id": item["id"],
                        "type": "page",
                        "title": title,
                        "created_time": item["created_time"],
                        "last_edited_time": item["last_edited_time"]
                    })
                elif obj_type == "database":
                    title = "".join([text["plain_text"] for text in item["title"]]) if item["title"] else "Untitled"
                    
                    formatted_results.append({
                        "id": item["id"],
                        "type": "database",
                        "title": title,
                        "created_time": item["created_time"],
                        "last_edited_time": item["last_edited_time"]
                    })
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps(formatted_results, indent=2)
                )
            ]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        logger.error(f"Error in call_tool: {str(e)}")
        return [
            TextContent(
                type="text",
                text=f"Error: {str(e)}\nPlease make sure your Notion integration is properly set up and has appropriate permissions."
            )
        ]
    finally:
        # Close the client connection
        await reader.close()

async def main():
    """Main entry point for the server"""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())