export interface StreamEvent {
  version: 1;
  run_id: string;
  seq: number;
  type: string;
  elapsed_ms: number;
  data: Record<string, unknown>;
}

/** 校验问题并生成独立请求；参数 question 为原始输入，返回 HTTP 请求体或抛出中文错误。 */
export function makeRequest(question: string) {
  const value = question.trim();
  if (!value) throw new Error("请输入一个问题。");
  if ([...value].length > 1000) throw new Error("问题不能超过 1000 个字符。");
  return { question: value, retrieval_scope: "all", top_k: 3 };
}

/** 将 HTTP 状态映射为中文；参数 status 为状态码，返回可展示文案。 */
export function httpMessage(status: number): string {
  return (
    (
      {
        400: "本次 Agent 输出不可用，请稍后手动重试。",
        403: "此演示接口仅允许本机指定来源访问。",
        404: "实时接口未开启，请使用 README 中的演示启动命令。",
        409: "上一条演示请求仍在执行，请等待结束后再提交。",
        422: "请求参数无效，请检查问题内容。",
        503: "RAG 服务暂不可用，请检查服务与依赖。",
      } as Record<number, string>
    )[status] ?? "服务响应异常，请检查后端后手动重试。"
  );
}

/** 增量解析 SSE；参数 notify 接收已校验事件，返回可跨网络分块重复调用的解析函数。 */
export function createParser(notify: (event: StreamEvent) => void) {
  let buffer = "";
  let lastSequence = 0;
  let runId = "";
  return function push(chunk: string): void {
    buffer += chunk;
    if (buffer.length > 2_000_000) throw new Error("事件数据超过展示上限。");
    let match: RegExpExecArray | null;
    while ((match = /\r?\n\r?\n/.exec(buffer)) !== null) {
      const block = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);
      const json = block
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (!json) continue;
      let event: StreamEvent;
      try {
        event = JSON.parse(json);
      } catch {
        throw new Error("服务返回了无法解析的事件。");
      }
      if (
        !event ||
        event.version !== 1 ||
        typeof event.run_id !== "string" ||
        !event.run_id ||
        !Number.isInteger(event.seq) ||
        event.seq !== lastSequence + 1 ||
        typeof event.type !== "string" ||
        !event.data ||
        typeof event.data !== "object" ||
        Array.isArray(event.data) ||
        !Number.isFinite(event.elapsed_ms) ||
        (runId && runId !== event.run_id)
      )
        throw new Error("服务返回的事件顺序或格式不正确。");
      runId = event.run_id;
      lastSequence = event.seq;
      notify(event);
    }
  };
}

/** 发出单次 POST 并消费流；参数为问题、回调与取消信号，无返回值，设置连接和业务进展期限。 */
export async function ask(
  question: string,
  notify: (event: StreamEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const body = JSON.stringify(makeRequest(question));
  const local = new AbortController();
  const combined = AbortSignal.any([signal, local.signal]);
  let timeoutMessage = "";
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  let receivedResponse = false;
  let terminal = false;
  /** 结束超时等待；参数 message 为中文原因，无返回值，不重发请求。 */
  function expire(message: string): void {
    timeoutMessage = message;
    local.abort();
  }
  let progressTimer = setTimeout(
    () => expire("后端连接超过 8 秒未响应，请检查 8082 服务和本地代理。"),
    8000,
  );
  const totalTimer = setTimeout(
    () =>
      expire("本次运行超过 5 分钟，已停止等待；已发出的模型调用可能仍在执行。"),
    300000,
  );
  /** 更新业务进展期限；无参数和返回值，心跳不会触发此函数。 */
  function touch(): void {
    clearTimeout(progressTimer);
    progressTimer = setTimeout(
      () =>
        expire(
          "90 秒内未收到新的执行进展，已停止等待。请检查当前节点、后端日志和网络；在途调用可能仍在执行。",
        ),
      90000,
    );
  }
  try {
    const response = await fetch("/api/opera/ask/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body,
      signal: combined,
    });
    receivedResponse = true;
    touch();
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      if (detail?.code === "backend_unavailable")
        throw new Error(
          "无法连接 OPERA 后端（8082）。请先启动 Python 服务，再手动重试。",
        );
      throw new Error(httpMessage(response.status));
    }
    if (
      !response.body ||
      !response.headers.get("content-type")?.includes("text/event-stream")
    )
      throw new Error("服务没有返回实时事件流，请检查代理配置。");
    const push = createParser((event) => {
      if (terminal) throw new Error("运行结束后收到额外事件。");
      touch();
      terminal = event.type === "run_completed" || event.type === "run_failed";
      notify(event);
    });
    reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (!terminal) {
      const { value, done } = await reader.read();
      if (done) {
        push(decoder.decode());
        break;
      }
      push(decoder.decode(value, { stream: true }));
    }
    if (!terminal)
      throw new Error(
        "连接提前中断，已收到的过程已保留；后端当前调用可能仍在执行。",
      );
  } catch (error) {
    if (timeoutMessage) throw new Error(timeoutMessage);
    if (!receivedResponse)
      throw new Error("无法连接 OPERA 服务，请检查前端代理和后端是否启动。");
    if (
      error instanceof TypeError ||
      (error instanceof DOMException && error.name === "AbortError")
    )
      throw new Error(
        "与后端的连接已中断，已收到的过程已保留。请检查服务状态后手动重试。",
      );
    throw error;
  } finally {
    clearTimeout(progressTimer);
    clearTimeout(totalTimer);
    if (reader) {
      await reader.cancel().catch(() => undefined);
      reader.releaseLock();
    }
  }
}
