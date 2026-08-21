"""Unit tests for AutonomousAgentSystem and MCP Tools."""
import pytest
import asyncio
from src.memory import SemanticMemoryStore
from src.mcp_tools import registry
from src.agent_system import AutonomousAgentSystem


@pytest.mark.asyncio
async def test_agent_workflow_execution():
    store = SemanticMemoryStore()
    store.add_memory("rule_1", "All memory allocations are checked against cluster constraints.")
    
    agent = AutonomousAgentSystem(memory_store=store, tool_registry=registry)
    result = await agent.execute_workflow("Calculate memory allocation: 512 * 8")

    assert result["status"] == "SUCCESS"
    assert len(result["execution_steps"]) > 0
    assert result["execution_steps"][0]["tool"] == "calculator"
    assert result["execution_steps"][0]["output"]["result"] == "4096"
