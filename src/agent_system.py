"""
Autonomous Multi-Agent Orchestrator (Planner, Executor, and Synthesizer).
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from src.memory import SemanticMemoryStore
from src.mcp_tools import MCPToolRegistry, registry


class AutonomousAgentSystem:
    """Coordinates specialized agent roles with semantic memory and MCP tool execution."""

    def __init__(self, memory_store: SemanticMemoryStore, tool_registry: Optional[MCPToolRegistry] = None):
        self.memory = memory_store
        self.tools = tool_registry or registry

    async def _plan_task(self, user_query: str, relevant_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deconstructs user query into deterministic execution steps."""
        await asyncio.sleep(0.02)
        plan = []
        
        lower_q = user_query.lower()
        if "calculate" in lower_q or any(c in user_query for c in "+-*/"):
            expr = user_query.split(":")[-1].strip() if ":" in user_query else "1024 * 16"
            plan.append({
                "step": 1,
                "agent": "ToolExecutor",
                "tool": "calculator",
                "args": {"expression": expr}
            })
            
        if any(k in lower_q for k in ["status", "metric", "cluster", "health", "vram"]):
            plan.append({
                "step": len(plan) + 1,
                "agent": "ToolExecutor",
                "tool": "system_metrics",
                "args": {"subsystem": "production_cluster_gpu"}
            })

        if any(k in lower_q for k in ["index", "sync", "dataset", "documents"]):
            plan.append({
                "step": len(plan) + 1,
                "agent": "ToolExecutor",
                "tool": "vector_indexer",
                "args": {"dataset_name": "enterprise_kb_v1"}
            })

        if not plan:
            plan.append({
                "step": 1,
                "agent": "KnowledgeSynthesizer",
                "action": "direct_synthesis",
                "context": relevant_context
            })
            
        return plan

    async def execute_workflow(self, user_query: str) -> Dict[str, Any]:
        """Main asynchronous agent loop: Retrieval -> Planning -> Execution -> Long-Term Commit."""
        start_time = time.perf_counter()
        
        # 1. Semantic Memory Retrieval
        memories = self.memory.search(user_query, top_k=2)

        # 2. Dynamic Planning
        plan = await self._plan_task(user_query, memories)

        # 3. Tool Execution Phase
        execution_results = []
        for step in plan:
            if step.get("agent") == "ToolExecutor":
                tool_name = step["tool"]
                args = step["args"]
                output = await self.tools.execute(tool_name, args)
                execution_results.append({
                    "step": step["step"],
                    "tool": tool_name,
                    "output": output
                })

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # 4. Result Synthesis
        synthesis = {
            "query": user_query,
            "relevant_memories": memories,
            "execution_steps": execution_results,
            "execution_time_ms": elapsed_ms,
            "status": "SUCCESS",
            "summary": f"Workflow resolved in {elapsed_ms}ms with {len(execution_results)} tool actions."
        }

        # 5. Commit to Persistent Memory
        self.memory.add_memory(
            memory_id=f"trace_{int(time.time() * 1000)}",
            text=f"Processed query: {user_query} | Steps: {len(execution_results)}",
            metadata={"latency_ms": elapsed_ms}
        )

        return synthesis
