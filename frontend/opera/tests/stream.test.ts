import { test } from "node:test";
import assert from "node:assert/strict";
import { makeRequest, createParser, httpMessage, ask } from "../src/stream.ts";

test("代理明确报告后端未启动，不重试", async () => {
  const original = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    return new Response('{"code":"backend_unavailable"}', { status: 503 });
  };
  try {
    await assert.rejects(
      ask("q", () => {}, new AbortController().signal),
      /后端（8082）/,
    );
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = original;
  }
});

test("连接超过 8 秒后解除等待", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const original = globalThis.fetch;
  globalThis.fetch = async (_url, options) =>
    new Promise((_resolve, reject) => {
      options?.signal?.addEventListener("abort", () =>
        reject(new DOMException("aborted", "AbortError")),
      );
    });
  try {
    const pending = assert.rejects(
      ask("q", () => {}, new AbortController().signal),
      /超过 8 秒/,
    );
    t.mock.timers.tick(8001);
    await pending;
  } finally {
    globalThis.fetch = original;
    t.mock.timers.reset();
  }
});

test("只有心跳没有业务进展时也会超时", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const original = globalThis.fetch;
  globalThis.fetch = async (_url, options) =>
    new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(": keep-alive\n\n"));
          options?.signal?.addEventListener("abort", () =>
            controller.error(new DOMException("aborted", "AbortError")),
          );
        },
      }),
      { headers: { "content-type": "text/event-stream" } },
    );
  try {
    const pending = assert.rejects(
      ask("q", () => {}, new AbortController().signal),
      /90 秒/,
    );
    await new Promise((resolve) => setImmediate(resolve));
    t.mock.timers.tick(90001);
    await pending;
  } finally {
    globalThis.fetch = original;
    t.mock.timers.reset();
  }
});

test("问题边界按 Unicode code point 校验，all 不包含 case_id", () => {
  assert.throws(() => makeRequest("  "));
  assert.throws(() => makeRequest("a".repeat(1001)));
  assert.equal(makeRequest("😀".repeat(1000)).question.length, 2000);
  assert.deepEqual(makeRequest(" q "), {
    question: "q",
    retrieval_scope: "all",
    top_k: 3,
  });
});

test("SSE 处理任意分块、CRLF、心跳和多事件", () => {
  const received: unknown[] = [];
  const push = createParser((event) => received.push(event));
  const e1 = {
    version: 1,
    run_id: "r",
    seq: 1,
    type: "run_started",
    elapsed_ms: 0,
    data: {},
  };
  const e2 = {
    ...e1,
    seq: 2,
    type: "agent_output",
    data: { output: "中文\n<script>bad()</script>" },
  };
  const wire = `: keep-alive\r\n\r\ndata: ${JSON.stringify(e1)}\r\n\r\nevent: agent_output\ndata: ${JSON.stringify(e2)}\n\n`;
  for (const char of wire) push(char);
  assert.deepEqual(received, [e1, e2]);
});

test("拒绝缺失事件、串 run 和无效 JSON", () => {
  const push = createParser(() => {});
  const event = {
    version: 1,
    run_id: "r",
    seq: 1,
    type: "run_started",
    elapsed_ms: 0,
    data: {},
  };
  push(`data: ${JSON.stringify(event)}\n\n`);
  assert.throws(() =>
    push(`data: ${JSON.stringify({ ...event, seq: 3 })}\n\n`),
  );
  assert.throws(() => createParser(() => {})("data: {oops}\n\n"));
});

test("HTTP 错误文案按类别区分", () => {
  const messages = [400, 403, 404, 409, 422, 503].map(httpMessage);
  assert.equal(new Set(messages).size, 6);
});

test("没有终态的断流报错且不重试", async () => {
  const originalFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    return new Response(
      'data: {"version":1,"run_id":"r","seq":1,"type":"run_started","elapsed_ms":0,"data":{}}\n\n',
      { headers: { "content-type": "text/event-stream" } },
    );
  };
  try {
    await assert.rejects(
      ask("q", () => {}, new AbortController().signal),
      /连接提前中断/,
    );
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
