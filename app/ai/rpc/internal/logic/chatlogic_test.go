package logic

import (
	"strings"
	"testing"
)

// TestValidateMessage 验证代码和 JSON 等正常技术输入不会被关键词过滤误拦截。
func TestValidateMessage(t *testing.T) {
	tests := []struct {
		name    string
		message string
		wantErr bool
	}{
		{
			name:    "允许包含 system、花括号和代码的技术问题",
			message: "system prompt 是什么？请解释 map[string]int{\"a\": 1}",
		},
		{
			name:    "拒绝空消息",
			message: " \n\t ",
			wantErr: true,
		},
		{
			name:    "拒绝控制字符",
			message: "hello\x00world",
			wantErr: true,
		},
		{
			name:    "拒绝超长消息",
			message: strings.Repeat("a", maxMessageBytes+1),
			wantErr: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateMessage(test.message)
			if (err != nil) != test.wantErr {
				t.Fatalf("validateMessage() error = %v, wantErr %v", err, test.wantErr)
			}
		})
	}
}

// TestBuildUserContent 验证检索资料和用户问题会被明确分隔。
func TestBuildUserContent(t *testing.T) {
	content := buildUserContent("如何排查死锁？", []string{"资料一", "资料二"})
	if !strings.Contains(content, "【不可信参考资料】") {
		t.Fatal("缺少不可信参考资料边界")
	}
	if !strings.Contains(content, "【用户问题】\n如何排查死锁？") {
		t.Fatal("缺少用户问题边界")
	}
}

// TestTrimHistory 验证历史裁剪只保留最近消息。
func TestTrimHistory(t *testing.T) {
	history := []map[string]string{
		{"content": "first"},
		{"content": "second"},
		{"content": "third"},
	}
	trimmed := trimHistory(history, 2)
	if len(trimmed) != 2 || trimmed[0]["content"] != "second" || trimmed[1]["content"] != "third" {
		t.Fatalf("trimHistory() = %#v, want the last two messages", trimmed)
	}
}

// TestGenerateSessionID 验证新会话标识为 UUID 风格的随机标识。
func TestGenerateSessionID(t *testing.T) {
	sessionID, err := generateSessionID()
	if err != nil {
		t.Fatalf("generateSessionID() error = %v", err)
	}
	if len(sessionID) != 36 || strings.Count(sessionID, "-") != 4 {
		t.Fatalf("generateSessionID() = %q, want UUID-style identifier", sessionID)
	}
}
