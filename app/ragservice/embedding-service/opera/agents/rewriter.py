"""Rewrite Agent。"""
import json
from ..llm import ResponsesClient
from ..schemas import RewriteResult
def rewrite(client: ResponsesClient, prompt: str, subgoal: str, failure_reason: str | None) -> RewriteResult:
    """仅根据当前失败原因生成新的检索 query。"""
    return client.complete(prompt, json.dumps({"subgoal":subgoal,"failure_reason":failure_reason}, ensure_ascii=False), RewriteResult, "opera_rewrite")
