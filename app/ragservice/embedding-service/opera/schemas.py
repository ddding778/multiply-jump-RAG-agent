"""OPERA HTTP、Agent 返回和运行状态的严格 Pydantic Schema。"""
from __future__ import annotations
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, model_validator

class StrictSchema(BaseModel):
    """拒绝未声明字段的 Schema 基类。"""
    model_config = ConfigDict(extra="forbid")

class RetrievalScope(StrEnum):
    """表示 HotpotQA 检索范围。"""
    CASE = "case"
    ALL = "all"

class OperaAskRequest(StrictSchema):
    """定义 OPERA HTTP 请求。"""
    question: str = Field(min_length=1, max_length=1000)
    retrieval_scope: RetrievalScope
    case_id: str | None = Field(default=None, min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=10)
    @model_validator(mode="after")
    def validate_scope(self) -> "OperaAskRequest":
        """校验 case/all 与 case_id 的组合。"""
        if self.retrieval_scope == RetrievalScope.CASE and not self.case_id: raise ValueError("case_id is required when retrieval_scope=case")
        if self.retrieval_scope == RetrievalScope.ALL and self.case_id is not None: raise ValueError("case_id must be omitted when retrieval_scope=all")
        return self

class PlanStep(StrictSchema):
    """表示 Planner 的一个可执行子目标。"""
    step_id: str = Field(pattern=r"^step_[1-4]$")
    subgoal: str = Field(min_length=1, max_length=500)
    depends_on: list[str] = Field(default_factory=list, max_length=3)
    is_final: bool

class PlanResult(StrictSchema):
    """表示经规划模型返回的多跳计划。"""
    steps: list[PlanStep] = Field(min_length=1, max_length=4)
    @model_validator(mode="after")
    def validate_plan(self) -> "PlanResult":
        """校验连续步骤编号、唯一最终步骤和顺序依赖。"""
        ids=[step.step_id for step in self.steps]
        if len(ids)!=len(set(ids)) or sum(step.is_final for step in self.steps)!=1 or not self.steps[-1].is_final: raise ValueError("plan requires unique steps and one final last step")
        for index, step in enumerate(self.steps):
            if step.step_id != f"step_{index + 1}": raise ValueError("plan step_id must be consecutive from step_1")
            if not set(step.depends_on).issubset(set(ids[:index])): raise ValueError("dependencies must reference previous steps")
        return self

class EvidenceRef(StrictSchema):
    """表示可回查的 paragraph 句子证据。"""
    chunk_id: str = Field(min_length=1)
    sentence_index: int = Field(ge=0)

class AnalysisStatus(StrEnum):
    """表示证据是否支持当前子目标。"""
    SUPPORTED="supported"; INSUFFICIENT="insufficient"

class AnalysisResult(StrictSchema):
    """表示 Analysis-Answer Agent 的受限输出。"""
    status: AnalysisStatus
    answer: str | None = Field(..., max_length=2000)
    evidence: list[EvidenceRef] = Field(..., max_length=6)
    needs_rewrite: bool
    failure_reason: str | None = Field(..., max_length=500)
    @model_validator(mode="after")
    def validate_result(self) -> "AnalysisResult":
        """校验支持与证据不足两种互斥状态。"""
        if self.status==AnalysisStatus.SUPPORTED and (not self.answer or not self.evidence or self.needs_rewrite): raise ValueError("supported result requires answer and evidence")
        if self.status==AnalysisStatus.INSUFFICIENT and (self.answer is not None or self.evidence or not self.needs_rewrite): raise ValueError("insufficient result must request rewrite")
        return self

class RewriteResult(StrictSchema):
    """表示 Rewrite Agent 仅可返回的查询改写。"""
    rewritten_query: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=100)

class StepResult(StrictSchema):
    """表示 Executor 写入 ExecutionState 的已验证步骤结果。"""
    step_id: str; status: str; answer: str | None = None; evidence: list[EvidenceRef] = Field(default_factory=list); retrieval_query: str; rewrite_count: int = Field(ge=0)

class OperaAskResponse(StrictSchema):
    """定义 OPERA HTTP 响应。"""
    run_id: str; status: str; answer: str | None; final_evidence: list[EvidenceRef]; completed_step_count: int
