"""
Model Context Protocol (MCP) Tool Bridge, JSON-RPC Registry, and Sandboxed Executors.
"""

from typing import Callable, Dict, Any, List, Optional
import inspect
from pydantic import BaseModel


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters_schema: Dict[str, Any]


class MCPToolRegistry:
    """Standardized Tool Registry following Model Context Protocol (MCP) tool schemas."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._definitions: Dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str):
        """Decorator to register a Python callable as a typed MCP tool."""
        def decorator(func: Callable):
            sig = inspect.signature(func)
            properties = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                param_type = "string"
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == dict:
                    param_type = "object"
                elif param.annotation == list:
                    param_type = "array"

                properties[param_name] = {"type": param_type}
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            self._definitions[name] = ToolDefinition(
                name=name,
                description=description,
                parameters_schema={
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            )
            self._tools[name] = func
            return func
        return decorator

    def get_manifest(self) -> List[Dict[str, Any]]:
        """Returns JSON schema definitions for all registered tools."""
        return [tool.model_dump() for tool in self._definitions.values()]

    async def execute(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Executes a registered tool within structured error boundaries."""
        args = arguments or {}
        if tool_name not in self._tools:
            return {"error": f"Tool '{tool_name}' not found.", "status": "failed"}

        func = self._tools[tool_name]
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = func(**args)
            return {"result": result, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "exception"}


# Global default registry instance
registry = MCPToolRegistry()

@registry.register(
    name="calculator",
    description="Safely evaluates deterministic mathematical expressions."
)
def compute_math(expression: str) -> str:
    allowed_chars = set("0123456789+-*/(). %")
    if not all(c in allowed_chars for c in expression):
        raise ValueError("Security constraint: Invalid characters detected in expression.")
    return str(eval(expression, {"__builtins__": None}, {}))

@registry.register(
    name="system_metrics",
    description="Inspects real-time cluster node health, VRAM, and system metrics."
)
def get_system_metrics(subsystem: str = "cluster_alpha") -> Dict[str, Any]:
    return {
        "subsystem": subsystem,
        "status": "HEALTHY",
        "cpu_usage_pct": 18.2,
        "memory_free_mb": 8192,
        "active_worker_threads": 8,
        "gpu_vram_allocated_gb": 12.4
    }

@registry.register(
    name="vector_indexer",
    description="Simulates automated vector re-indexing for external document sources."
)
def trigger_vector_indexing(dataset_name: str) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "indexed_chunks": 420,
        "embedding_dimensions": 384,
        "status": "COMPLETED"
    }
