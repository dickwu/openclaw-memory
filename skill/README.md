# Memory Skill for OpenClaw

This skill provides graph-based memory storage and retrieval using GraphRAG concepts.

## Setup

1. Install dependencies:
```bash
cd ~/opensource/openclaw-memory
pip install -e .
```

2. Set environment variables:
```bash
export OPENAI_API_KEY="your-api-key"  # Optional: for semantic search
export MEMORY_DIR="/path/to/memories"  # Optional: custom storage path
```

3. Start the memory server:
```bash
uvicorn memory_server:app --reload --port 4801
```

## Usage

### Store a memory
```
Remember that Peilin's birthday is December 15th
```

The skill will:
- Store the memory with a unique ID
- Extract entities (person: Peilin, date: December 15th)
- Extract relationships (has_birthday)

### Query memories
```
What do you remember about Peilin?
```

The skill will:
- Search memories semantically or by keyword
- Return relevant memories with context

### List all memories
```
List all memories
```

### Query the knowledge graph
```
Show me the knowledge graph
```

## Configuration

The skill connects to `http://localhost:4801` by default. 
Override with `MEMORY_SERVER_URL` env var.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/memories` | POST | Store a new memory |
| `/memories` | GET | List memories |
| `/memories/{id}` | GET | Get specific memory |
| `/query` | POST | Semantic query |
| `/graph` | GET | Knowledge graph |
| `/graph/explore/{entity}` | GET | Explore entity connections |
