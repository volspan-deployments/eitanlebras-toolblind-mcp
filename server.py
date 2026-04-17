from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
import uvicorn
import threading
from fastmcp import FastMCP
import httpx
import os
from typing import Optional

mcp = FastMCP("ToolBlind Test API")

BASE_URL = "http://localhost:8000"


@mcp.tool()
async def get_api_info() -> dict:
    """Get an overview of the ToolBlind API and check its health/status. Use this first to confirm the API is running and to understand available capabilities and version information."""
    _track("get_api_info")
    async with httpx.AsyncClient() as client:
        root_response = await client.get(f"{BASE_URL}/")
        root_data = root_response.json()

        health_response = await client.get(f"{BASE_URL}/health")
        health_data = health_response.json()

        return {
            "overview": root_data,
            "health": health_data
        }


@mcp.tool()
async def get_dataset_stats() -> dict:
    """Retrieve aggregate statistics about the ToolBlind benchmark dataset, including total task count, breakdown by tier (1/2/3), domain distribution, and outcome categories. Use this to understand the dataset scope before exploring or running tasks."""
    _track("get_dataset_stats")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/stats")
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def list_tasks(
    tier: Optional[str] = None,
    domain: Optional[str] = None,
    outcome: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> dict:
    """List and filter benchmark tasks from the ToolBlind dataset. Use this to browse available tasks, narrow down by difficulty tier (1=easy, 2=medium, 3=hard), domain (e.g. web, code, math), or ground truth outcome. Supports pagination via limit and offset."""
    _track("list_tasks")
    params = {}
    if tier is not None:
        params["tier"] = tier
    if domain is not None:
        params["domain"] = domain
    if outcome is not None:
        params["outcome"] = outcome
    if limit is not None:
        params["limit"] = limit
    if offset is not None:
        params["offset"] = offset

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/tasks", params=params)
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def get_task(task_id: str) -> dict:
    """Retrieve full details for a specific ToolBlind task by ID, including its description, required tools, trajectory steps, ground truth outcome, and metadata. Use this to inspect a task before running it or to analyze its structure."""
    _track("get_task")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/tasks/{task_id}")
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def run_task(
    task_id: str,
    strategy: Optional[str] = "smart"
) -> dict:
    """Execute a single ToolBlind benchmark task using a stub agent strategy and return the agent's trajectory, decision, and evaluation result. Use this to test how an agent handles tool absence on a specific task. Strategies: 'smart' (attempts correct reasoning), 'always_halt' (always stops when tool is missing), 'always_confabulate' (always fabricates a path forward)."""
    _track("run_task")
    params = {}
    if strategy is not None:
        params["strategy"] = strategy

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(f"{BASE_URL}/run/{task_id}", params=params)
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def run_batch(
    strategy: Optional[str] = "smart",
    tier: Optional[str] = None,
    domain: Optional[str] = None,
    sample: Optional[int] = None,
    seed: Optional[int] = None
) -> dict:
    """Run multiple ToolBlind benchmark tasks in batch and receive aggregate evaluation metrics including ToolBlind Score, accuracy by tier/domain, and strategy performance. Use this to evaluate agent behavior at scale or benchmark a specific strategy across a filtered subset of tasks."""
    _track("run_batch")
    params = {}
    if strategy is not None:
        params["strategy"] = strategy
    if tier is not None:
        params["tier"] = tier
    if domain is not None:
        params["domain"] = domain
    if sample is not None:
        params["sample"] = sample
    if seed is not None:
        params["seed"] = seed

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(f"{BASE_URL}/run/batch", params=params)
        response.raise_for_status()
        return response.json()




_SERVER_SLUG = "eitanlebras-toolblind"
_REQUIRES_AUTH = False

def _get_api_key() -> str:
    """Get API key from environment. Clients pass keys via MCP config headers."""
    return os.environ.get("API_KEY", "")

def _auth_headers() -> dict:
    """Build authorization headers for upstream API calls."""
    key = _get_api_key()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}", "X-API-Key": key}

def _track(tool_name: str, ua: str = ""):
    import threading
    def _send():
        try:
            import urllib.request, json as _json
            data = _json.dumps({"slug": _SERVER_SLUG, "event": "tool_call", "tool": tool_name, "user_agent": ua}).encode()
            req = urllib.request.Request("https://www.volspan.dev/api/analytics/event", data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()

async def health(request):
    return JSONResponse({"status": "ok", "server": mcp.name})

async def tools(request):
    registered = await mcp.list_tools()
    tool_list = [{"name": t.name, "description": t.description or ""} for t in registered]
    return JSONResponse({"tools": tool_list, "count": len(tool_list)})

sse_app = mcp.http_app(transport="sse")

app = Starlette(
    routes=[
        Route("/health", health),
        Route("/tools", tools),
        Mount("/", sse_app),
    ],
    lifespan=sse_app.lifespan,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
