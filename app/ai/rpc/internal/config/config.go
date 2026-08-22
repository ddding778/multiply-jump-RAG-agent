package config

import (
	"os"

	"github.com/zeromicro/go-zero/core/stores/cache"
	"github.com/zeromicro/go-zero/core/stores/redis"
	"github.com/zeromicro/go-zero/zrpc"
)

type Config struct {
	zrpc.RpcServerConf
	DeepSeek   DeepSeek
	DataSource string
	CacheRedis cache.CacheConf
	RedisConf  redis.RedisConf
	Prompt     Prompt
	RAG        RAG
}

type DeepSeek struct {
	BaseUrl  string
	Model    string
	ChatPath string
	ApiKey   string
}

type Prompt struct {
	System           string
	MaxHistoryRounds int32
}

type RAG struct {
	EmbeddingUrl string
}

// ApplyEnvironment 使用运行环境中的敏感配置覆盖文件配置，避免将密钥固化到 YAML。
// 参数 c 为已加载的服务配置；无返回值，环境变量为空时保持原配置不变。
func ApplyEnvironment(c *Config) {
	if apiKey := os.Getenv("DEEPSEEK_API_KEY"); apiKey != "" {
		c.DeepSeek.ApiKey = apiKey
	}
}
