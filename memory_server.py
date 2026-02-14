"""
GraphRAG-based Memory Server for OpenClaw

This server provides memory storage and retrieval using knowledge graph concepts:
- Extracts entities and relationships from memories using LLM
- Stores memories as a graph (nodes + edges)
- Supports semantic queries over the memory graph

Run: uvicorn memory_server:app --reload --port 4801
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
from datetime import datetime
from pathlib import Path

# Try to import graph components - fall back to simple version if not available
try:
    from graphrag import (
        initialize_storage,
        load_graph,
        insert_graph,
        query_graph,
        Graph
    )
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False
    print("GraphRAG not available, using simple memory storage")

app = FastAPI(title="OpenClaw Memory", version="0.1.0")

# Configuration
MEMORY_DIR = Path(os.getenv("MEMORY_DIR", "/tmp/openclaw-memory"))
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
MEMORIES_FILE = MEMORY_DIR / "memories.json"
GRAPH_FILE = MEMORY_DIR / "graph.json"

# LLM configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")  # Same as OpenClaw
LLM_MODEL = os.getenv("LLM_MODEL", "openrouter/anthropic/claude-3.5-sonnet")
USE_OPENROUTER = bool(OPENROUTER_API_KEY)

# --- Data Models ---

class MemoryInput(BaseModel):
    """Input for creating a new memory"""
    content: str
    tags: Optional[List[str]] = []
    metadata: Optional[Dict[str, Any]] = {}
    
class Memory(BaseModel):
    """A stored memory with extracted entities"""
    id: str
    content: str
    timestamp: str
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    entities: List[Dict[str, str]] = []  # Extracted entities
    relationships: List[Dict[str, str]] = []  # Extracted relationships

class MemoryQuery(BaseModel):
    """Query for searching memories"""
    query: str
    limit: Optional[int] = 5
    tags: Optional[List[str]] = None

class EntityRelationship(BaseModel):
    """Entity or relationship in the knowledge graph"""
    type: str  # "entity" or "relationship"
    name: str
    entity_type: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None
    target: Optional[str] = None

# --- Simple Memory Storage (fallback) ---

def load_memories() -> List[Memory]:
    """Load memories from disk"""
    if MEMORIES_FILE.exists():
        with open(MEMORIES_FILE, "r") as f:
            data = json.load(f)
            return [Memory(**m) for m in data]
    return []

def save_memories(memories: List[Memory]) -> None:
    """Save memories to disk"""
    with open(MEMORIES_FILE, "w") as f:
        json.dump([m.model_dump() for m in memories], f, indent=2)

def load_graph_data() -> Dict[str, Any]:
    """Load graph data from disk"""
    if GRAPH_FILE.exists():
        with open(GRAPH_FILE, "r") as f:
            return json.load(f)
    return {"entities": [], "relationships": []}

def save_graph_data(graph_data: Dict[str, Any]) -> None:
    """Save graph data to disk"""
    with open(GRAPH_FILE, "w") as f:
        json.dump(graph_data, f, indent=2)

# --- LLM-based Entity Extraction ---

async def extract_entities_llm(content: str) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Use LLM to extract entities and relationships from content"""
    if not OPENAI_API_KEY and not OPENROUTER_API_KEY:
        return [], []
    
    try:
        from langchain_openai import ChatOpenAI
        
        # Use OpenRouter if API key is available
        if USE_OPENROUTER:
            llm = ChatOpenAI(
                model=LLM_MODEL,
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base="https://openrouter.ai/api/v1"
            )
        else:
            llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)
        
        # Prompt for entity extraction
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""Extract entities and relationships from the following content.

Content: {content}

Return a JSON object with:
- "entities": list of {{
    "name": entity name,
    "type": entity type (person, project, concept, tool, location, organization, or other)
  }}
- "relationships": list of {{
    "source": source entity name,
    "target": target entity name,
    "type": relationship type (knows, works_on, uses, located_in, created, etc.)
  }}

Only extract entities that are clearly mentioned. Be concise."""
        )
        
        response = llm.invoke(prompt.format(content=content))
        content_json = response.content
        
        # Try to parse JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', content_json)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("entities", []), data.get("relationships", [])
    except Exception as e:
        print(f"Entity extraction error: {e}")
    
    return [], []

# --- API Routes ---

@app.get("/")
async def root():
    return {
        "name": "OpenClaw Memory",
        "version": "0.1.0",
        "graphrag": GRAPHRAG_AVAILABLE,
        "llm": {
            "provider": "openrouter" if USE_OPENROUTER else "openai" if OPENAI_API_KEY else "none",
            "model": LLM_MODEL
        },
        "endpoints": {
            "POST /memories": "Store a new memory",
            "GET /memories": "List all memories",
            "GET /memories/{id}": "Get a specific memory",
            "POST /query": "Query memories semantically",
            "GET /graph": "Get knowledge graph",
            "DELETE /memories": "Clear all memories"
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok", "graphrag": GRAPHRAG_AVAILABLE}

# --- Memory Operations ---

@app.post("/memories", response_model=Memory)
async def add_memory(memory_input: MemoryInput):
    """Store a new memory with entity extraction"""
    import uuid
    
    memory_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()
    
    # Extract entities if LLM is available
    entities = []
    relationships = []
    if OPENAI_API_KEY or OPENROUTER_API_KEY:
        entities, relationships = await extract_entities_llm(memory_input.content)
    
    memory = Memory(
        id=memory_id,
        content=memory_input.content,
        timestamp=timestamp,
        tags=memory_input.tags or [],
        metadata=memory_input.metadata or {},
        entities=entities,
        relationships=relationships
    )
    
    # Save memory
    memories = load_memories()
    memories.append(memory)
    save_memories(memories)
    
    # Update graph
    graph_data = load_graph_data()
    
    # Add new entities
    for entity in entities:
        entity_exists = any(
            e.get("name") == entity.get("name") 
            for e in graph_data.get("entities", [])
        )
        if not entity_exists:
            graph_data.setdefault("entities", []).append(entity)
    
    # Add new relationships
    for rel in relationships:
        rel_exists = any(
            r.get("source") == rel.get("source") and 
            r.get("target") == rel.get("target")
            for r in graph_data.get("relationships", [])
        )
        if not rel_exists:
            graph_data.setdefault("relationships", []).append({
                **rel,
                "memory_id": memory_id
            })
    
    save_graph_data(graph_data)
    
    return memory

@app.get("/memories", response_model=List[Memory])
async def list_memories(tag: Optional[str] = None, limit: int = 20):
    """List all memories, optionally filtered by tag"""
    memories = load_memories()
    
    if tag:
        memories = [m for m in memories if tag in m.tags]
    
    # Sort by timestamp, newest first
    memories.sort(key=lambda m: m.timestamp, reverse=True)
    
    return memories[:limit]

@app.get("/memories/{memory_id}", response_model=Memory)
async def get_memory(memory_id: str):
    """Get a specific memory by ID"""
    memories = load_memories()
    for memory in memories:
        if memory.id == memory_id:
            return memory
    raise HTTPException(status_code=404, detail="Memory not found")

@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory by ID"""
    memories = load_memories()
    original_count = len(memories)
    memories = [m for m in memories if m.id != memory_id]
    
    if len(memories) == original_count:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    save_memories(memories)
    
    # Update graph - remove relationships from this memory
    graph_data = load_graph_data()
    graph_data["relationships"] = [
        r for r in graph_data.get("relationships", [])
        if r.get("memory_id") != memory_id
    ]
    save_graph_data(graph_data)
    
    return {"deleted": memory_id}

@app.delete("/memories")
async def clear_memories():
    """Clear all memories"""
    save_memories([])
    save_graph_data({"entities": [], "relationships": []})
    return {"cleared": True}

# --- Query Operations ---

@app.post("/query")
async def query_memories(query: MemoryQuery):
    """Query memories using semantic search or graph traversal"""
    memories = load_memories()
    
    if not memories:
        return {"results": [], "message": "No memories stored"}
    
    # If LLM available, use semantic search
    if (OPENAI_API_KEY or OPENROUTER_API_KEY) and GRAPHRAG_AVAILABLE:
        try:
            from langchain_openai import ChatOpenAI
            from langchain.schema import HumanMessage
            
            # Use OpenRouter if available
            if USE_OPENROUTER:
                llm = ChatOpenAI(
                    model=LLM_MODEL,
                    openai_api_key=OPENROUTER_API_KEY,
                    openai_api_base="https://openrouter.ai/api/v1"
                )
            else:
                llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)
            
            # Get relevant context from graph
            graph_data = load_graph_data()
            
            # Build context from recent memories
            recent_memories = memories[-10:]
            context = "\n".join([
                f"- {m.content}" + (f" [tags: {', '.join(m.tags)}]" if m.tags else "")
                for m in recent_memories
            ])
            
            # Ask LLM to find relevant memories
            prompt = f"""Given the query: "{query.query}"

And these memories:
{context}

Which memories are most relevant to the query? Return a JSON list of memory IDs that are relevant, with a brief reason.
"""
            response = llm.invoke([HumanMessage(content=prompt)])
            
            import re
            json_match = re.search(r'\[[\s\S]*\]', response.content)
            if json_match:
                relevant_ids = json.loads(json_match.group())
                results = [m for m in memories if m.id in relevant_ids]
                return {"results": results, "type": "semantic"}
        except Exception as e:
            print(f"Query error: {e}")
    
    # Fallback: keyword search
    query_lower = query.query.lower()
    results = []
    
    for memory in memories:
        # Score by relevance
        score = 0
        if query_lower in memory.content.lower():
            score += 10
        if memory.tags:
            for tag in memory.tags:
                if query_lower in tag.lower():
                    score += 5
        if query.tags:
            for tag in query.tags:
                if tag in memory.tags:
                    score += 3
        
        if score > 0:
            results.append((score, memory))
    
    # Sort by score, then by timestamp
    results.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)
    
    return {
        "results": [r[1] for r in results[:query.limit]],
        "type": "keyword"
    }

# --- Graph Operations ---

@app.get("/graph")
async def get_graph():
    """Get the knowledge graph"""
    return load_graph_data()

@app.get("/graph/entities")
async def get_entities(entity_type: Optional[str] = None):
    """Get entities, optionally filtered by type"""
    graph_data = load_graph_data()
    entities = graph_data.get("entities", [])
    
    if entity_type:
        entities = [e for e in entities if e.get("type") == entity_type]
    
    return {"entities": entities}

@app.get("/graph/relationships")
async def get_relationships(entity: Optional[str] = None):
    """Get relationships, optionally filtered by entity"""
    graph_data = load_graph_data()
    relationships = graph_data.get("relationships", [])
    
    if entity:
        relationships = [
            r for r in relationships
            if r.get("source") == entity or r.get("target") == entity
        ]
    
    return {"relationships": relationships}

@app.get("/graph/explore/{entity_name}")
async def explore_entity(entity_name: str):
    """Explore an entity and its connections"""
    graph_data = load_graph_data()
    
    # Find the entity
    entity = None
    for e in graph_data.get("entities", []):
        if e.get("name", "").lower() == entity_name.lower():
            entity = e
            break
    
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    # Find related relationships
    relationships = [
        r for r in graph_data.get("relationships", [])
        if r.get("source") == entity_name or r.get("target") == entity_name
    ]
    
    # Find related entities
    related_entities = []
    for r in relationships:
        if r.get("source") == entity_name:
            related_entities.append(r.get("target"))
        if r.get("target") == entity_name:
            related_entities.append(r.get("source"))
    
    return {
        "entity": entity,
        "relationships": relationships,
        "connected_entities": related_entities
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4801)
