package model

import (
	"context"
	"fmt"

	"github.com/zeromicro/go-zero/core/stores/cache"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

var _ MessagesModel = (*customMessagesModel)(nil)

type (
	// MessagesModel is an interface to be customized, add more methods here,
	// and implement the added methods in customMessagesModel.
	MessagesModel interface {
		messagesModel
		FindRecentByConversationId(ctx context.Context, conversationId int64, limit int) ([]Messages, error)
		InsertExchange(ctx context.Context, conversationId int64, userContent string, assistantContent string) error
	}

	customMessagesModel struct {
		*defaultMessagesModel
	}
)

// NewMessagesModel returns a model for the database table.
func NewMessagesModel(conn sqlx.SqlConn, c cache.CacheConf, opts ...cache.Option) MessagesModel {
	return &customMessagesModel{
		defaultMessagesModel: newMessagesModel(conn, c, opts...),
	}
}

// FindRecentByConversationId 查询会话最近消息，结果按消息 ID 倒序返回。
// 参数 conversationId 为会话主键，limit 为最大消息数；返回最近消息列表。
func (m *customMessagesModel) FindRecentByConversationId(ctx context.Context, conversationId int64, limit int) ([]Messages, error) {
	if limit <= 0 {
		return []Messages{}, nil
	}

	var messages []Messages
	query := fmt.Sprintf("select %s from %s where `conversation_id` = ? order by `id` desc limit ?", messagesRows, m.table)
	if err := m.QueryRowsNoCacheCtx(ctx, &messages, query, conversationId, limit); err != nil {
		return nil, err
	}
	return messages, nil
}

// InsertExchange 在同一事务中保存一轮用户消息和助手回复，避免只写入其中一条。
// 参数 conversationId 为会话主键，userContent 和 assistantContent 分别为本轮内容；返回事务执行错误。
func (m *customMessagesModel) InsertExchange(ctx context.Context, conversationId int64, userContent string, assistantContent string) error {
	query := fmt.Sprintf("insert into %s (`conversation_id`, `role`, `content`) values (?, ?, ?)", m.table)
	return m.TransactCtx(ctx, func(ctx context.Context, session sqlx.Session) error {
		if _, err := session.ExecCtx(ctx, query, conversationId, 0, userContent); err != nil {
			return err
		}
		if _, err := session.ExecCtx(ctx, query, conversationId, 1, assistantContent); err != nil {
			return err
		}
		return nil
	})
}
