import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ask, makeRequest, type StreamEvent } from "./stream";
import "./style.css";

type Data = Record<string, unknown>;
type Status = "idle" | "running" | "completed" | "insufficient" | "error";
type Tab = "input" | "output" | "prompt";
interface Node {
  id: string;
  title: string;
  step: string;
  kind: "agent" | "retrieval";
  input: unknown;
  prompt?: string;
  output?: unknown;
  status: string;
  elapsed: number;
  duration?: number;
  attempt?: number;
}
const example =
  "What screenwriter with credits for Evolution co-wrote a film starring Nicolas Cage and Tea Leoni?";
const agentNames: Record<string, string> = {
  opera_plan: "Planner",
  opera_analysis: "Analysis-Answer",
  opera_rewrite: "Rewrite",
};
const statusNames: Record<Status, string> = {
  idle: "等待提问",
  running: "正在执行",
  completed: "回答完成",
  insufficient: "证据不足",
  error: "运行中断",
};

/** 安全读取字符串；参数 value 是接口值，返回字符串或空值。 */
function text(value: unknown): string {
  return typeof value === "string" ? value : "";
}
/** 安全读取对象；参数 value 是接口值，返回字典或空对象。 */
function record(value: unknown): Data {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Data)
    : {};
}
/** 安全读取数组；参数 value 是接口值，返回数组或空列表。 */
function list(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}
/** 格式化真实内容；参数 value 为输入或输出，返回缩进 JSON 或原始文本。 */
function pretty(value: unknown): string {
  if (typeof value === "string") {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      return value;
    }
  }
  return JSON.stringify(value, null, 2) ?? "";
}
/** 构建按真实事件顺序排列的节点；参数 events 为本次事件，返回可选择的调用卡片。 */
function buildNodes(events: StreamEvent[]): Node[] {
  const nodes: Node[] = [];
  for (const event of events) {
    const d = event.data;
    const id = text(d.call_id) || text(d.retrieval_id);
    if (event.type === "agent_started")
      nodes.push({
        id,
        title: agentNames[text(d.agent)] ?? "Agent",
        kind: "agent",
        step: text(d.step_id),
        input: d.input,
        prompt: text(d.instructions),
        status: "调用中",
        elapsed: event.elapsed_ms,
        attempt: Number(d.attempt),
      });
    if (event.type === "retrieval_started")
      nodes.push({
        id,
        title: "Hybrid Retriever",
        kind: "retrieval",
        step: text(d.step_id),
        input: { query: d.query, top_k: d.top_k },
        status: "检索中",
        elapsed: event.elapsed_ms,
      });
    const node = nodes.find((item) => item.id === id);
    if (node && event.type === "agent_output") {
      node.output = d.output;
      node.status = "等待校验";
      node.duration = Number(d.duration_ms);
    }
    if (node && event.type === "agent_validated")
      node.status =
        d.validation === "passed"
          ? "Schema 通过"
          : d.will_retry
            ? "校验失败 · 将修复"
            : "校验失败";
    if (node && event.type === "agent_failed") {
      node.status = "调用失败";
      node.output = {
        error_type: d.error_type,
        provider_metadata: d.provider_metadata,
      };
    }
    if (node && event.type === "retrieval_completed") {
      node.status = "检索完成";
      node.output = d.paragraphs;
      node.duration = event.elapsed_ms - node.elapsed;
    }
    if (event.type === "run_failed")
      for (const active of nodes) {
        if (["调用中", "检索中", "等待校验"].includes(active.status))
          active.status = "执行中断";
      }
  }
  return nodes;
}

/** 绘制界面图标；参数 name、size 指定图形与尺寸，返回无外部依赖的 SVG。 */
function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const paths: Record<string, string> = {
    arrow: "M5 12h14m-6-6 6 6-6 6",
    layers: "m12 3 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 16l9 5 9-5",
    spark: "m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z",
    copy: "M9 9h11v11H9zM15 9V4H4v11h5",
    search: "M10 17a7 7 0 1 0 0-14 7 7 0 0 0 0 14Zm5-2 6 6",
    code: "m8 6-6 6 6 6m8-12 6 6-6 6",
    check: "m5 12 4 4L19 6",
    book: "M4 4h7a3 3 0 0 1 3 3v14a4 4 0 0 0-4-2H4V4Zm10 3a3 3 0 0 1 3-3h4v15h-3a4 4 0 0 0-4 2",
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.65"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={paths[name] ?? paths.layers} />
    </svg>
  );
}

/** 展示证据正文；参数 paragraphs 为真实检索段落，返回带原始句子索引的列表。 */
function Paragraphs({ paragraphs }: { paragraphs: unknown[] }) {
  return (
    <div className="paragraphs">
      {paragraphs.map((item, i) => {
        const p = record(item);
        return (
          <article className="paragraph" key={`${text(p.chunk_id)}-${i}`}>
            <div className="paragraph-title">
              <Icon name="book" />
              <strong>{text(p.title)}</strong>
            </div>
            <code>{text(p.chunk_id)}</code>
            <ol start={0}>
              {list(p.sentences).map((sentence, j) => (
                <li key={j} value={j}>
                  {text(sentence)}
                </li>
              ))}
            </ol>
          </article>
        );
      })}
    </div>
  );
}

/** 渲染演示工作台；无参数，返回包含独立请求状态、事件时间线和输入输出面板的页面。 */
function App() {
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [error, setError] = useState("");
  const [phase, setPhase] = useState("");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [tab, setTab] = useState<Tab>("input");
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const busy = useRef(false);
  const controller = useRef<AbortController | null>(null);
  const startTime = useRef(0);
  const nodes = buildNodes(events);
  const active = nodes.find((node) => node.id === selected) ?? nodes.at(-1);
  const final = events.find((event) => event.type === "run_completed")?.data;
  const runId = events[0]?.run_id ?? "";
  const plan = list(
    record(events.find((event) => event.type === "plan_ready")?.data.plan)
      .steps,
  );
  const completedSteps = events.filter(
    (event) => event.type === "step_completed",
  ).length;
  const running = status === "running";
  const evidence = list(final?.final_evidence);
  const allParagraphs = events
    .filter((event) => event.type === "retrieval_completed")
    .flatMap((event) => list(event.data.paragraphs));

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(
      () => setElapsed((Date.now() - startTime.current) / 1000),
      250,
    );
    return () => window.clearInterval(timer);
  }, [running]);
  useEffect(() => () => controller.current?.abort(), []);

  /** 提交一次问题；无参数和返回值，校验后消费真实事件，用同步锁防止连续点击。 */
  async function submit(): Promise<void> {
    if (busy.current) return;
    try {
      makeRequest(question);
    } catch (e) {
      setError((e as Error).message);
      return;
    }
    busy.current = true;
    controller.current = new AbortController();
    setEvents([]);
    setSelected("");
    setTab("input");
    setError("");
    setPhase("正在连接 OPERA 后端…");
    setWarnings([]);
    setCopied(false);
    setCopyError("");
    setElapsed(0);
    startTime.current = Date.now();
    setStatus("running");
    console.info("opera_demo_started");
    try {
      await ask(
        question,
        (event) => {
          setEvents((previous) => [...previous, event]);
          if (event.type === "initialization")
            setPhase(text(event.data.message));
          if (event.type === "dependency_warning")
            setWarnings((previous) => [...previous, text(event.data.message)]);
          if (event.type === "run_completed") {
            if (
              !["completed", "insufficient"].includes(text(event.data.status))
            )
              throw new Error("服务返回了未知的完成状态。");
            if (
              event.data.status === "completed" &&
              !text(event.data.answer).trim()
            )
              throw new Error("服务未返回有效答案。");
            setStatus(event.data.status as Status);
            console.info("opera_demo_completed", {
              run_id: event.run_id,
              status: event.data.status,
            });
          }
          if (event.type === "run_failed") {
            setStatus("error");
            setError(text(event.data.message) || "运行失败，请手动重试。");
          }
        },
        controller.current.signal,
      );
    } catch (e) {
      setStatus("error");
      setError(e instanceof Error ? e.message : "连接中断，请检查服务。");
      console.warn("opera_demo_failed");
    } finally {
      busy.current = false;
      setElapsed((Date.now() - startTime.current) / 1000);
    }
  }

  /** 复制本次运行 ID；无参数和返回值，权限失败时提示用户手动复制。 */
  async function copyRun(): Promise<void> {
    try {
      await navigator.clipboard.writeText(runId);
      setCopied(true);
      setCopyError("");
    } catch {
      setCopyError("复制失败，请选中运行 ID 手动复制。");
    }
  }

  return (
    <div className="app-shell">
      <aside className="rail">
        <a className="brand-symbol" href="#" aria-label="OPERA 首页">
          O<span />
        </a>
        <div className="rail-active" title="推理工作台">
          <Icon name="layers" size={22} />
        </div>
        <a href="#architecture" title="架构说明">
          <Icon name="code" size={22} />
        </a>
        <span className="rail-bottom">LOCAL</span>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <div className="brand">
            OPERA
            <span className="brand-divider" />
            <span>多跳推理工作台</span>
            <span className="edition">INTERVIEW DEMO</span>
          </div>
          <div className="local-badge">
            <span />
            本地演示环境
          </div>
        </header>
        <main>
          <section className="intro">
            <div>
              <div className="eyebrow">MULTI-AGENT · MULTI-HOP RAG</div>
              <h1>
                看见答案，<span>也看见依据。</span>
              </h1>
              <p>从问题拆解到证据推导，逐步查看每一次真实的 Agent 调用。</p>
            </div>
            <div className="corpus-stamp">
              <Icon name="book" size={22} />
              <div>
                <strong>HotpotQA</strong>
                <span>全库检索 · 独立请求</span>
              </div>
            </div>
          </section>
          <div className="top-grid">
            <section className="panel question-panel">
              <div className="section-heading">
                <h2>
                  <span className="section-number">01</span>提出一个多跳问题
                </h2>
                <span className="muted small">QUESTION</span>
              </div>
              <label className="sr-only" htmlFor="question">
                多跳问题
              </label>
              <textarea
                id="question"
                value={question}
                disabled={running}
                placeholder="试着问一个需要关联多条事实的问题…"
                onChange={(e) => {
                  setQuestion(e.target.value);
                  setError("");
                }}
                onKeyDown={(e) => {
                  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                    e.preventDefault();
                    void submit();
                  }
                }}
              />
              <div className="input-meta">
                <span>支持中英文 · 推荐使用 HotpotQA 英文问题</span>
                <span
                  className={
                    [...question.trim()].length > 1000 ? "over-limit" : ""
                  }
                >
                  {[...question.trim()].length} / 1000
                </span>
              </div>
              <div className="question-actions">
                <button
                  className="example-button"
                  disabled={running}
                  onClick={() => {
                    setQuestion(example);
                    setError("");
                  }}
                >
                  <Icon name="spark" size={16} />
                  填入示例问题
                </button>
                <button
                  className="primary-button"
                  disabled={running}
                  onClick={() => void submit()}
                >
                  {running ? (
                    <span className="spinner" />
                  ) : (
                    <Icon name="arrow" />
                  )}
                  <span>{running ? "正在推理" : "开始推理"}</span>
                  <kbd>Ctrl ↵</kbd>
                </button>
              </div>
              <p className="scope-note">
                语料仅限
                HotpotQA；不联网搜索，也不检索项目源码。每次运行独立计费，不自动重发。
              </p>
            </section>
            <section className="panel architecture-panel" id="architecture">
              <div className="section-heading">
                <h2>三个 Agent，一条证据链</h2>
                <span className="outline-badge">架构示意</span>
              </div>
              <div className="architecture-flow">
                <span>Planner</span>
                <b>→</b>
                <span>Retriever</span>
                <b>→</b>
                <span>Analysis</span>
              </div>
              <div className="rewrite-path">
                ↳ 证据不足 → Rewrite → 再次检索 ↵
              </div>
              <p>
                规划有依赖的子目标，以检索证据驱动回答。最终答案来自计划中的最后一个子目标。
              </p>
              <div className="architecture-tags">
                <span>Dense + BM25</span>
                <span>RRF + MMR</span>
                <span>Schema 校验</span>
              </div>
            </section>
          </div>
          <section className="run-bar" aria-live="polite">
            <div className={`status-pill ${status}`}>
              <span className={running ? "pulse-dot" : ""} />
              {statusNames[status]}
            </div>
            <div className="run-metric">
              <strong>
                {elapsed.toFixed(1)}
                <small>s</small>
              </strong>
              <span>运行耗时</span>
            </div>
            <div className="run-metric">
              <strong>
                {completedSteps}
                <small>{plan.length ? ` / ${plan.length}` : ""}</small>
              </strong>
              <span>已完成子目标</span>
            </div>
            <div className="run-metric">
              <strong>
                {nodes.filter((node) => node.kind === "agent").length}
              </strong>
              <span>模型调用</span>
            </div>
            <div className="run-id">
              <span>RUN ID</span>
              <code>{runId || "开始运行后生成"}</code>
              {runId && (
                <button onClick={() => void copyRun()} aria-label="复制运行 ID">
                  <Icon name={copied ? "check" : "copy"} size={16} />
                </button>
              )}
            </div>
          </section>
          {(error || copyError) && (
            <div className="error-banner" role="alert">
              {error || copyError}
            </div>
          )}
          {running && nodes.length === 0 && (
            <div className="phase-banner" role="status">
              {phase}
            </div>
          )}
          {warnings.length > 0 && (
            <div className="warning-banner" role="status">
              {warnings.map((message, index) => (
                <p key={index}>{message}</p>
              ))}
            </div>
          )}
          <div className="execution-grid">
            <section className="panel timeline-panel">
              <div className="section-heading">
                <h2>
                  <span className="section-number">02</span>实时执行
                </h2>
                <span className="count-badge">{nodes.length}</span>
              </div>
              <p className="section-description">
                节点开始时显示输入，返回后显示输出。
              </p>
              {!nodes.length ? (
                <div className="timeline-empty">
                  <div className="empty-orbit">
                    <Icon name="layers" size={25} />
                  </div>
                  <h3>
                    {running
                      ? "正在准备执行器"
                      : status === "error"
                        ? "准备或执行失败"
                        : "等待第一条证据链"}
                  </h3>
                  <p>
                    {running
                      ? phase
                      : status === "error"
                        ? "请查看上方错误提示，处理后手动重新提交。"
                        : "发送问题后，真实调用将按执行顺序出现在这里。"}
                  </p>
                </div>
              ) : (
                <div className="timeline">
                  {nodes.map((node, i) => (
                    <button
                      key={node.id}
                      className={`timeline-node ${active?.id === node.id ? "selected" : ""}`}
                      onClick={() => {
                        setSelected(node.id);
                        setTab(node.output === undefined ? "input" : "output");
                      }}
                    >
                      <div className={`node-icon ${node.kind}`}>
                        <Icon
                          name={
                            node.kind === "retrieval"
                              ? "search"
                              : node.title === "Planner"
                                ? "layers"
                                : "spark"
                          }
                          size={17}
                        />
                      </div>
                      <div className="node-body">
                        <div>
                          <strong>{node.title}</strong>
                          <span>{String(i + 1).padStart(2, "0")}</span>
                        </div>
                        <p>
                          {node.step || "问题规划"}
                          {node.attempt ? ` · 第 ${node.attempt} 次尝试` : ""}
                        </p>
                        <span
                          className={`node-status ${node.status.includes("失败") ? "failed" : ""}`}
                        >
                          {node.status}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
              {plan.length > 0 && (
                <details className="plan-details">
                  <summary>查看规划的 {plan.length} 个子目标</summary>
                  {plan.map((item, i) => {
                    const step = record(item);
                    return (
                      <p key={i}>
                        <b>{text(step.step_id)}</b> {text(step.subgoal)}
                        <small>
                          依赖：{list(step.depends_on).join(", ") || "无"}
                          {step.is_final ? " · 最终子目标" : ""}
                        </small>
                      </p>
                    );
                  })}
                </details>
              )}
            </section>
            <section className="panel inspector">
              <div className="section-heading">
                <h2>
                  {active ? active.title : "调用详情"}
                  {active?.step && (
                    <span className="step-tag">{active.step}</span>
                  )}
                </h2>
                <span className="muted small">
                  {active?.duration !== undefined
                    ? `${(active.duration / 1000).toFixed(2)} s`
                    : "LIVE INSPECTOR"}
                </span>
              </div>
              <div className="tabs" role="tablist" aria-label="调用内容">
                {(["input", "output", "prompt"] as Tab[]).map((value) => (
                  <button
                    role="tab"
                    aria-selected={tab === value}
                    key={value}
                    onClick={() => setTab(value)}
                    disabled={
                      value === "prompt" && active?.kind === "retrieval"
                    }
                    className={tab === value ? "active" : ""}
                  >
                    {value === "input"
                      ? "实际输入"
                      : value === "output"
                        ? "调用输出"
                        : "系统提示词"}
                    {value === "output" && active?.output !== undefined && (
                      <span className="tab-dot" />
                    )}
                  </button>
                ))}
              </div>
              {!active ? (
                <div className="inspector-empty">
                  <div className="code-illustration">
                    <span />
                    <span />
                    <span />
                    <span />
                  </div>
                  <h3>每一步，都有据可查</h3>
                  <p>
                    选择左侧节点，查看发送给 Agent 的实际输入、
                    <br />
                    系统提示词与返回结果。
                  </p>
                  <span className="outline-badge">
                    真实事件驱动 · 不模拟执行进度
                  </span>
                </div>
              ) : (
                <div className="inspector-content">
                  <div className="content-caption">
                    <span>
                      {tab === "prompt"
                        ? "本次实际使用的系统提示词"
                        : tab === "input"
                          ? "本次实际调用输入"
                          : active.status}
                    </span>
                    <span>
                      {tab === "output"
                        ? "OUTPUT"
                        : tab === "input"
                          ? "INPUT"
                          : "INSTRUCTIONS"}
                    </span>
                  </div>
                  {tab === "output" && active.output === undefined ? (
                    <div className="waiting-output">
                      {running ? <span className="spinner" /> : null}
                      {running
                        ? "正在等待此节点返回…"
                        : "此节点未返回可用结果。"}
                    </div>
                  ) : tab === "output" && active.kind === "retrieval" ? (
                    <Paragraphs paragraphs={list(active.output)} />
                  ) : (
                    <pre className="code-view">
                      {pretty(
                        tab === "input"
                          ? active.input
                          : tab === "prompt"
                            ? active.prompt
                            : active.output,
                      )}
                    </pre>
                  )}
                  {tab === "output" && active.kind === "agent" && (
                    <p className="validation-note">
                      Schema 通过表示结构有效；证据引用和步骤依赖仍由 Executor
                      校验，只有接受后的子目标才计入完成数。
                    </p>
                  )}
                </div>
              )}
            </section>
          </div>
          <section
            className={`panel answer-panel ${status === "completed" ? "answer-ready" : ""}`}
          >
            <div className="section-heading">
              <h2>
                <span className="section-number">03</span>最终答案与证据
              </h2>
              <span className="outline-badge">EVIDENCE-BASED ANSWER</span>
            </div>
            {status === "completed" && final ? (
              <>
                <p className="final-answer">{text(final.answer)}</p>
                <div className="evidence-grid">
                  {evidence.map((item, i) => {
                    const ref = record(item);
                    const p = record(
                      allParagraphs.find(
                        (item) => record(item).chunk_id === ref.chunk_id,
                      ),
                    );
                    const sentence = list(p.sentences)[
                      Number(ref.sentence_index)
                    ];
                    return (
                      <article className="evidence-card" key={i}>
                        <span className="evidence-label">证据 {i + 1}</span>
                        <strong>{text(p.title) || "证据引用"}</strong>
                        {typeof sentence === "string" && (
                          <blockquote>{sentence}</blockquote>
                        )}
                        <code>{text(ref.chunk_id)}</code>
                        <small>
                          sentence_index: {String(ref.sentence_index)}
                        </small>
                      </article>
                    );
                  })}
                </div>
              </>
            ) : (
              <div className="answer-placeholder">
                <Icon
                  name={status === "insufficient" ? "search" : "book"}
                  size={23}
                />
                <div>
                  <strong>
                    {status === "insufficient"
                      ? "现有 HotpotQA 证据不足，系统未生成答案。"
                      : status === "error"
                        ? "本次运行未完成，已收到的调用过程保留在上方。"
                        : "最终子目标完成后，答案将在这里呈现。"}
                  </strong>
                  <p>
                    {status === "insufficient"
                      ? "可以查看上方检索结果与改写过程，了解证据缺口。"
                      : "证据正文仅来自本次真实检索，不补写未返回的内容。"}
                  </p>
                </div>
              </div>
            )}
          </section>
          <footer>
            <span>
              <span className="footer-dot" />
              OPERA-style Multi-Agent RAG
            </span>
            <span>
              每次提问独立运行 · 页面刷新后清空 ·
              关闭页面不保证已发出的模型调用停止
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
