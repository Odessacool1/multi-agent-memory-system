"""Unit tests for AutonomousAgentSystem and MCP Tools."""
import asyncio
from src.memory import SemanticMemoryStore
from src.mcp_tools import registry
from src.agent_system import AutonomousAgentSystem


def test_agent_workflow_execution():
    store = SemanticMemoryStore()
    store.add_memory("rule_1", "All memory allocations are checked against cluster constraints.")
    
    agent = AutonomousAgentSystem(memory_store=store, tool_registry=registry)
    result = asyncio.run(agent.execute_workflow("Calculate memory allocation: 512 * 8"))

    assert result["status"] == "SUCCESS"
    assert len(result["execution_steps"]) > 0
    assert result["execution_steps"][0]["tool"] == "calculator"
    assert result["execution_steps"][0]["output"]["result"] == "4096"
