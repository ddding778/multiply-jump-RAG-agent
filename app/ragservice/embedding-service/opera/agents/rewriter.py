"""Rewrite Agent。"""
import json
from ..llm import ResponsesClient
from ..schemas import RewriteResult


def rewrite(
    client: ResponsesClient,
    prompt: str,
    subgoal: str,
    current_query: str,
    accepted_dependencies: dict[str, str],
    failure_reason: str | None,
    evidence: list[dict[str, object]],
) -> RewriteResult:
    """根据当前子目标、失败原因和本轮候选 paragraph 生成新的检索 query。

    参数 client 为 Responses 调用客户端，prompt 为已解析的 system prompt，subgoal 为 Planner 的原子子目标，
    current_query 为本轮实际检索 query，accepted_dependencies 为已验证的前序答案，
    failure_reason 为 Analysis-Answer 的不足原因，evidence 为本轮不可信候选 paragraph；
    返回通过 RewriteResult 校验的改写查询。
    """

    body = {
        "subgoal": subgoal,
        "current_query": current_query,
        "accepted_dependencies": accepted_dependencies,
        "failure_reason": failure_reason,
        "untrusted_evidence": evidence,
    }
    return client.complete(prompt, json.dumps(body, ensure_ascii=False), RewriteResult, "opera_rewrite")
