"""
Interactive CLI and Demonstration Entrypoint for the Multi-Agent System.
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.memory import SemanticMemoryStore
from src.agent_system import AutonomousAgentSystem

console = Console()


async def run_demo():
    console.print(Panel.fit(
        "[bold cyan]⚡ PRODUCTION-READY MULTI-AGENT SYSTEM[/bold cyan]\n"
        "[dim]FastEmbed • Model Context Protocol (MCP) • Persistent Semantic Memory[/dim]\n"
        "[green]Author: Odessacool1[/green]",
        border_style="cyan"
    ))

    # 1. Initialize Memory Store
    memory = SemanticMemoryStore()
    memory.add_memory(
        "policy_01",
        "Production server cluster node is allocated with 64GB RAM and 8 vCPUs on Vultr GPU cloud."
    )
    memory.add_memory(
        "policy_02",
        "All database synchronization events must enforce idempotent vector insertion."
    )

    # 2. Initialize Agent System
    agent = AutonomousAgentSystem(memory_store=memory)

    queries = [
        "Calculate total cluster memory capacity: 64 * 4",
        "Inspect real-time GPU VRAM and cluster health metrics",
        "Trigger vector index update for enterprise document repository"
    ]

    for q in queries:
        console.print(f"\n[bold yellow]❯ Running Query:[/bold yellow] [bold white]{q}[/bold white]")
        res = await agent.execute_workflow(q)
        
        table = Table(title="Workflow Execution Summary", border_style="blue")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Status", res["status"])
        table.add_row("Latency", f"{res['execution_time_ms']} ms")
        table.add_row("Steps Executed", str(len(res["execution_steps"])))
        table.add_row("Summary", res["summary"])
        console.print(table)


if __name__ == "__main__":
    asyncio.run(run_demo())
