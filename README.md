# 第一个学习go-zero的项目
- 参考了采用了looklook的整体架构，开发文档请看docs目录下的开发文档.md。
- 前端页面还没调试，只是ai做的初版前端

## AI RPC 本地启动

### 输入

- MySQL：本地 `3306` 端口，连接配置位于未提交的 `app/ai/rpc/etc/ai.yaml`。
- Redis：本地 `6379` 端口。
- etcd：本地 `2379` 端口。AI RPC 会在启动时注册服务，未启动 etcd 时 RPC 无法监听。
- DeepSeek 密钥：项目根目录 `.env` 中的 `DEEPSEEK_API_KEY`，该文件已被 Git 忽略。

`.env` 示例：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
```

### 启动命令

Go 服务不会自动读取 `.env`。在 PowerShell 中先将 key 注入当前进程环境，再启动 AI RPC：

```powershell
$deepSeekLine = Get-Content .env | Where-Object { $_ -match '^\s*DEEPSEEK_API_KEY\s*=' } | Select-Object -First 1
if ($null -eq $deepSeekLine) { throw 'DEEPSEEK_API_KEY missing from .env' }
$env:DEEPSEEK_API_KEY = ($deepSeekLine -split '=', 2)[1].Trim().Trim('"').Trim("'")
go run .\app\ai\rpc\ai.go -f .\app\ai\rpc\etc\ai.yaml
```

运行环境变量中的 `DEEPSEEK_API_KEY` 会覆盖 `ai.yaml` 的同名配置。命令不会输出密钥；启动成功后，AI RPC 默认监听 `8080`。
