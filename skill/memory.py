import os
import httpx
from typing import Optional

MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:4801")

class MemoryClient:
    """Client for the GraphRAG Memory Server"""
    
    def __init__(self, base_url: str = MEMORY_SERVER_URL):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        await self.client.aclose()
    
    async def health(self) -> dict:
        """Check server health"""
        resp = await self.client.get(f"{self.base_url}/health")
        return resp.json()
    
    async def add_memory(
        self,
        content: str,
        tags: Optional[list] = None,
        metadata: Optional[dict] = None
    ) -> dict:
        """Store a new memory"""
        payload = {
            "content": content,
            "tags": tags or [],
            "metadata": metadata or {}
        }
        resp = await self.client.post(f"{self.base_url}/memories", json=payload)
        resp.raise_for_status()
        return resp.json()
    
    async def list_memories(self, tag: Optional[str] = None, limit: int = 20) -> list:
        """List memories"""
        params = {"limit": limit}
        if tag:
            params["tag"] = tag
        resp = await self.client.get(f"{self.base_url}/memories", params=params)
        resp.raise_for_status()
        return resp.json()
    
    async def get_memory(self, memory_id: str) -> dict:
        """Get a specific memory"""
        resp = await self.client.get(f"{self.base_url}/memories/{memory_id}")
        resp.raise_for_status()
        return resp.json()
    
    async def query(self, query: str, limit: int = 5, tags: Optional[list] = None) -> dict:
        """Query memories"""
        payload = {
            "query": query,
            "limit": limit,
            "tags": tags
        }
        resp = await self.client.post(f"{self.base_url}/query", json=payload)
        resp.raise_for_status()
        return resp.json()
    
    async def get_graph(self) -> dict:
        """Get knowledge graph"""
        resp = await self.client.get(f"{self.base_url}/graph")
        resp.raise_for_status()
        return resp.json()
    
    async def explore_entity(self, entity_name: str) -> dict:
        """Explore an entity's connections"""
        resp = await self.client.get(f"{self.base_url}/graph/explore/{entity_name}")
        resp.raise_for_status()
        return resp.json()
    
    async def delete_memory(self, memory_id: str) -> dict:
        """Delete a memory"""
        resp = await self.client.delete(f"{self.base_url}/memories/{memory_id}")
        resp.raise_for_status()
        return resp.json()
    
    async def clear_memories(self) -> dict:
        """Clear all memories"""
        resp = await self.client.delete(f"{self.base_url}/memories")
        resp.raise_for_status()
        return resp.json()


# Global client instance
_memory_client: Optional[MemoryClient] = None

def get_memory_client() -> MemoryClient:
    global _memory_client
    if _memory_client is None:
        _memory_client = MemoryClient()
    return _memory_client

async def remember(
    content: str,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None
) -> str:
    """
    Store a new memory with entity extraction.
    
    Usage:
        await remember("Peilin's birthday is December 15")
    """
    client = get_memory_client()
    memory = await client.add_memory(content, tags, metadata)
    return f"Memory stored: {memory['id']}"

async def recall(query: str, limit: int = 5) -> list:
    """
    Query memories semantically.
    
    Usage:
        results = await recall("Peilin's birthday")
    """
    client = get_memory_client()
    result = await client.query(query, limit)
    return result.get("results", [])

async def memories(tag: Optional[str] = None, limit: int = 20) -> list:
    """
    List all memories, optionally filtered by tag.
    
    Usage:
        all_memories = await memories()
        birthday_memories = await memories(tag="birthday")
    """
    client = get_memory_client()
    return await client.list_memories(tag=tag, limit=limit)

async def knowledge_graph() -> dict:
    """
    Get the knowledge graph.
    
    Usage:
        graph = await knowledge_graph()
    """
    client = get_memory_client()
    return await client.get_graph()

async def explore(entity_name: str) -> dict:
    """
    Explore an entity's connections in the graph.
    
    Usage:
        info = await explore("Peilin")
    """
    client = get_memory_client()
    return await client.explore_entity(entity_name)
