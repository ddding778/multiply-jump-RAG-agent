import { defineConfig, type ProxyOptions } from "vite";

/** 去掉前端代理前缀；参数 path 为请求路径，返回后端路径。 */
function rewriteApi(path: string): string {
  return path.replace(/^\/api/, "");
}

const proxy: Record<string, ProxyOptions> = {
  "^/api/(opera/ask/stream|health)$": {
    target: "http://127.0.0.1:8082",
    rewrite: rewriteApi,
    timeout: 0,
    proxyTimeout: 0,
    /** 将后端连接故障转换为稳定 JSON；参数 proxy 为代理实例，无返回值。 */
    configure(proxy) {
      proxy.on("error", (_error, _request, response) => {
        if ("writeHead" in response && !response.headersSent) {
          response.writeHead(503, {
            "Content-Type": "application/json; charset=utf-8",
          });
          response.end(JSON.stringify({ code: "backend_unavailable" }));
        }
      });
    },
  },
};

export default defineConfig({
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy,
    fs: { strict: true, allow: ["."] },
  },
  preview: { host: "127.0.0.1", port: 5173, strictPort: true, proxy },
  build: { outDir: "../../out/opera-frontend/dist", emptyOutDir: false },
});
