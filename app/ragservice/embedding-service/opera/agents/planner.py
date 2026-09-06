"""中心化 Planner Agent。"""
from ..llm import ResponsesClient
from ..schemas import PlanResult
def plan(client: ResponsesClient, prompt: str, question: str) -> PlanResult:
    """根据原始问题生成受限多跳计划。"""
    return client.complete(prompt, f"Question:\n{question}", PlanResult, "opera_plan")
