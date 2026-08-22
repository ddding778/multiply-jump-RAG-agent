package config

import "testing"

// TestApplyEnvironment 验证运行环境中的密钥会覆盖文件中的旧配置。
func TestApplyEnvironment(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "environment-key")
	conf := Config{DeepSeek: DeepSeek{ApiKey: "file-key"}}
	ApplyEnvironment(&conf)
	if conf.DeepSeek.ApiKey != "environment-key" {
		t.Fatalf("ApiKey = %q, want environment override", conf.DeepSeek.ApiKey)
	}
}
