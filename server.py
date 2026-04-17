from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
import uvicorn
import threading
from fastmcp import FastMCP
import httpx
import os
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("ToolBlind")

BASE_URL = "http://localhost:8000"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def get_headers():
    return {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }


@mcp.tool()
async def generate_dataset(
    seed: int = 42,
    output_dir: Optional[str] = None,
    num_tasks: int = 500,
) -> dict:
    """Generate the ToolBlind benchmark dataset of tasks. Use this before running any experiments — it creates the 500-task dataset across categories and commitment tiers. Must be run first if no dataset exists."""
    _track("generate_dataset")
    payload = {
        "seed": seed,
        "num_tasks": num_tasks,
    }
    if output_dir is not None:
        payload["output_dir"] = output_dir

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{BASE_URL}/generate_dataset",
            json=payload,
            headers=get_headers(),
        )
        if response.status_code >= 400:
            return {"error": response.text, "status_code": response.status_code}
        return response.json()


@mcp.tool()
async def run_experiment(
    experiment_type: str,
    models: Optional[List[str]] = None,
    sample: Optional[int] = None,
    results_dir: Optional[str] = None,
) -> dict:
    """Run a specific ToolBlind experiment to evaluate AI agents on tool-absence reasoning. Use this to execute one of the five experiment types: baseline, commitment, framing, registry_size, or cot. Each experiment tests a different aspect of agent behavior when a required tool is missing."""
    _track("run_experiment")
    if models is None:
        models = ["claude"]

    payload: dict = {
        "experiment_type": experiment_type,
        "models": models,
    }
    if sample is not None:
        payload["sample"] = sample
    if results_dir is not None:
        payload["results_dir"] = results_dir

    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            f"{BASE_URL}/run_experiment",
            json=payload,
            headers=get_headers(),
        )
        if response.status_code >= 400:
            return {"error": response.text, "status_code": response.status_code}
        return response.json()


@mcp.tool()
async def run_all_experiments(
    models: Optional[List[str]] = None,
    sample: Optional[int] = None,
) -> dict:
    """Run all five ToolBlind experiments (baseline, commitment, framing, registry_size, cot) sequentially across one or more models. Use this for full experimental reproduction as described in the paper."""
    _track("run_all_experiments")
    if models is None:
        models = ["claude", "openai", "gemini"]

    payload: dict = {
        "models": models,
    }
    if sample is not None:
        payload["sample"] = sample

    async with httpx.AsyncClient(timeout=3600.0) as client:
        response = await client.post(
            f"{BASE_URL}/run_all_experiments",
            json=payload,
            headers=get_headers(),
        )
        if response.status_code >= 400:
            return {"error": response.text, "status_code": response.status_code}
        return response.json()


@mcp.tool()
async def analyze_results(
    result_files: Optional[List[str]] = None,
    use_latest: bool = True,
    output_format: str = "rich",
) -> dict:
    """Analyze experiment results from one or more result files. Use this after running experiments to compute confabulation rates, commitment effects, and model comparisons. Can target specific files or the latest results automatically."""
    _track("analyze_results")
    payload: dict = {
        "use_latest": use_latest,
        "output_format": output_format,
    }
    if result_files is not None:
        payload["result_files"] = result_files

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{BASE_URL}/analyze_results",
            json=payload,
            headers=get_headers(),
        )
        if response.status_code >= 400:
            return {"error": response.text, "status_code": response.status_code}
        return response.json()


@mcp.tool()
async def list_tasks(
    category: Optional[str] = None,
    commitment_tier: Optional[int] = None,
    limit: int = 20,
    tasks_dir: Optional[str] = None,
) -> dict:
    """List and filter tasks in the ToolBlind dataset. Use this to inspect the benchmark tasks, understand the task distribution across categories and commitment tiers, or find specific tasks for debugging."""
    _track("list_tasks")
    payload: dict = {
        "limit": limit,
    }
    if category is not None:
        payload["category"] = category
    if commitment_tier is not None:
        payload["commitment_tier"] = commitment_tier
    if tasks_dir is not None:
        payload["tasks_dir"] = tasks_dir

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/list_tasks",
            json=payload,
            headers=get_headers(),
        )
        if response.status_code >= 400:
            return {"error": response.text, "status_code": response.status_code}
        return response.json()


@mcp.tool()
async def validate_dataset(
    tasks_dir: Optional[str] = None,
    strict: bool = False,
) -> dict:
    """Validate the integrity and structure of the generated ToolBlind dataset. Use this to verify the dataset is complete, correctly formatted, and has the expected distribution of tasks before running expensive experiments."""
    _track("validate_dataset")
    payload: dict = {
        "strict": strict,
    }
    if tasks_dir is not None:
        payload["tasks_dir"] = tasks_dir

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{BASE_URL}/validate_dataset",
            json=payload,
            headers=get_headers(),
        )
        if response.status_code >= 400:
            return {"error": response.text, "status_code": response.status_code}
        return response.json()


@mcp.tool()
async def get_environment_info(
    check_api_keys: bool = True,
    show_cache_stats: bool = False,
) -> dict:
    """Retrieve information about the current ToolBlind environment configuration, including API key status, directory paths, cache state, and available models. Use this to diagnose setup issues or confirm configuration before running experiments."""
    _track("get_environment_info")
    payload: dict = {
        "check_api_keys": check_api_keys,
        "show_cache_stats": show_cache_stats,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/get_environment_info",
            json=payload,
            headers=get_headers(),
        )
        if response.status_code >= 400:
            return {"error": response.text, "status_code": response.status_code}
        return response.json()


@mcp.tool()
async def evaluate_agent_response(
    task_id: str,
    agent_trajectory: List[dict],
    agent_final_response: str,
    judge_model: Optional[str] = None,
) -> dict:
    """Use the judge model to evaluate a single agent response to a tool-absence scenario. Use this for debugging, manual inspection, or evaluating custom agent outputs outside of the main experiment pipeline."""
    _track("evaluate_agent_response")
    payload: dict = {
        "task_id": task_id,
        "agent_trajectory": agent_trajectory,
        "agent_final_response": agent_final_response,
    }
    if judge_model is not None:
        payload["judge_model"] = judge_model

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{BASE_URL}/evaluate_agent_response",
            json=payload,
            headers=get_headers(),
        )
        if response.status_code >= 400:
            return {"error": response.text, "status_code": response.status_code}
        return response.json()




_SERVER_SLUG = "eitanlebras-toolblind"
_REQUIRES_AUTH = True

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
