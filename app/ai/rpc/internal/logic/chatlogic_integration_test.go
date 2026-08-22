//go:build integration

package logic

import (
	"context"
	"path/filepath"
	"testing"

	"aiChat/app/ai/rpc/internal/config"
	"aiChat/app/ai/rpc/internal/svc"
	"aiChat/app/ai/rpc/pb"

	"github.com/zeromicro/go-zero/core/conf"
)

// TestChatIntegration 使用真实 MySQL、Redis 和模型服务验证 Chat 主链路。
// 运行前需要设置 DEEPSEEK_API_KEY，并确保 MySQL 与 Redis 可连接。
func TestChatIntegration(t *testing.T) {
	var serviceConfig config.Config
	configPath := filepath.Join("..", "..", "etc", "ai.yaml")
	conf.MustLoad(configPath, &serviceConfig)
	config.ApplyEnvironment(&serviceConfig)
	serviceConfig.RAG.EmbeddingUrl = "http://127.0.0.1:1"

	serviceContext := svc.NewServiceContext(serviceConfig)
	chatLogic := NewChatLogic(context.Background(), serviceContext)

	first, err := chatLogic.Chat(&pb.ChatRequest{
		UserId:  9001,
		Message: "请仅回复 CHAT_INTEGRATION_OK，并说明 JSON 的花括号不会被输入过滤拦截：{\"ok\": true}。",
	})
	if err != nil {
		t.Fatalf("first Chat() error = %v", err)
	}
	if first.SessionId == "" || first.Reply == "" {
		t.Fatal("first Chat() returned empty session or reply")
	}
	t.Logf("first_chat=ok session_id=%s reply_bytes=%d", first.SessionId, len(first.Reply))

	second, err := chatLogic.Chat(&pb.ChatRequest{
		UserId:    9001,
		SessionId: first.SessionId,
		Message:   "继续本次对话，仅回复 CHAT_HISTORY_OK。",
	})
	if err != nil {
		t.Fatalf("second Chat() error = %v", err)
	}
	if second.SessionId != first.SessionId || second.Reply == "" {
		t.Fatal("second Chat() did not preserve session or reply")
	}
	t.Logf("second_chat=ok session_id=%s reply_bytes=%d", second.SessionId, len(second.Reply))

	_, err = chatLogic.Chat(&pb.ChatRequest{
		UserId:    9002,
		SessionId: first.SessionId,
		Message:   "跨用户读取测试",
	})
	if err == nil {
		t.Fatal("cross-user session access unexpectedly succeeded")
	}
	t.Log("cross_user_session=rejected")
}
