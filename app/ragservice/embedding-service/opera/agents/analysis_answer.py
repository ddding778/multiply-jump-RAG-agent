"""Analysis-Answer Agent。"""
import json
from ..llm import ResponsesClient
from ..schemas import AnalysisResult
def analyze(client: ResponsesClient, prompt: str, subgoal: str, dependencies: dict[str, str], evidence: list[dict[str, object]]) -> AnalysisResult:
    """基于不可信 evidence 与已验证前序结果生成结构化子答案。"""
    body={"subgoal":subgoal,"accepted_dependencies":dependencies,"untrusted_evidence":evidence}
    return client.complete(prompt, json.dumps(body, ensure_ascii=False), AnalysisResult, "opera_analysis")
