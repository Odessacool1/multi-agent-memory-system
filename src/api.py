"""
FastAPI REST API Service for the Multi-Agent System.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.memory import SemanticMemoryStore
from src.mcp_tools import registry
from src.agent_system import AutonomousAgentSystem

app = FastAPI(
    title="Production Multi-Agent Memory System",
    description="Autonomous Agent Orchestration with FastEmbed Vector Memory & Model Context Protocol (MCP)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global instances
memory_store = SemanticMemoryStore()
agent_system = AutonomousAgentSystem(memory_store=memory_store, tool_registry=registry)


class WorkflowRequest(BaseModel):
    query: str = Field(..., example="Calculate cluster memory requirement: 1024 * 32")


class MemoryInsertRequest(BaseModel):
    id: str = Field(..., example="policy_infra_01")
    content: str = Field(..., example="All GPU clusters are pinned to cluster_alpha node pools.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "total_memories": len(memory_store.entries),
        "registered_tools": len(registry.get_manifest())
    }


@app.get("/tools")
async def list_tools():
    """Returns available MCP tool definitions and schemas."""
    return {"tools": registry.get_manifest()}


@app.post("/workflow/execute")
async def execute_agent_workflow(req: WorkflowRequest):
    """Dispatches autonomous agent workflow for a given query."""
    try:
        result = await agent_system.execute_workflow(req.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/add")
async def add_memory_record(req: MemoryInsertRequest):
    """Embeds and commits a knowledge record into the persistent vector memory."""
    try:
        entry = memory_store.add_memory(req.id, req.content, req.metadata)
        return {"status": "success", "id": entry.id, "total_memories": len(memory_store.entries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/search")
async def search_memory(q: str, top_k: int = 3):
    """Queries vector memory by semantic cosine similarity."""
    results = memory_store.search(q, top_k=top_k)
    return {"query": q, "results": results}
