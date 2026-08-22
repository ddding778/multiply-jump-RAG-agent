package model

import (
	"context"
	"fmt"

	"github.com/zeromicro/go-zero/core/stores/cache"
	"github.com/zeromicro/go-zero/core/stores/sqlc"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

var _ ConversationsModel = (*customConversationsModel)(nil)

type (
	// ConversationsModel is an interface to be customized, add more methods here,
	// and implement the added methods in customConversationsModel.
	ConversationsModel interface {
		conversationsModel
		FindOneByUserIdAndSessionId(ctx context.Context, userId int64, sessionId string) (*Conversations, error)
	}

	customConversationsModel struct {
		*defaultConversationsModel
	}
)

// NewConversationsModel returns a model for the database table.
func NewConversationsModel(conn sqlx.SqlConn, c cache.CacheConf, opts ...cache.Option) ConversationsModel {
	return &customConversationsModel{
		defaultConversationsModel: newConversationsModel(conn, c, opts...),
	}
}

// FindOneByUserIdAndSessionId 按用户和会话标识查询会话，避免跨用户读取会话。
// 参数 userId 为当前认证用户，sessionId 为客户端传入会话标识；返回匹配会话或未找到错误。
func (m *customConversationsModel) FindOneByUserIdAndSessionId(ctx context.Context, userId int64, sessionId string) (*Conversations, error) {
	var resp Conversations
	query := fmt.Sprintf("select %s from %s where `user_id` = ? and `session_id` = ? limit 1", conversationsRows, m.table)
	err := m.QueryRowNoCacheCtx(ctx, &resp, query, userId, sessionId)
	switch err {
	case nil:
		return &resp, nil
	case sqlc.ErrNotFound:
		return nil, ErrNotFound
	default:
		return nil, err
	}
}
