// Code scaffolded by goctl. Safe to edit.
// goctl 1.9.2

package user

import (
	"aiChat/app/ai/rpc/pb"
	"aiChat/pkg/ctxdata"
	"context"

	"aiChat/app/gateway/api/internal/svc"
	"aiChat/app/gateway/api/internal/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type ChatLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewChatLogic(ctx context.Context, svcCtx *svc.ServiceContext) *ChatLogic {
	return &ChatLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *ChatLogic) Chat(req *types.ChatReq) (resp *types.ChatResp, err error) {

	// 从上下文中获取请求信息
	userId := ctxdata.GetUidFromCtx(l.ctx)
	requestId := ctxdata.GetRequestID(l.ctx)

	logx.Infof("event=chat_request_start request_id=%s user_id=%d session_id=%s message_bytes=%d",
		requestId, userId, req.SessionId, len(req.Message))

	rpcResp, err := l.svcCtx.AiRpc.Chat(l.ctx, &pb.ChatRequest{
		SessionId: req.SessionId,
		Message:   req.Message,
		UserId:    userId,
	})
	if err != nil {
		logx.Errorf("event=chat_request_error request_id=%s user_id=%d session_id=%s err=%v",
			requestId, userId, req.SessionId, err)
		return nil, err
	}

	logx.Infof("event=chat_request_success request_id=%s user_id=%d session_id=%s",
		requestId, userId, req.SessionId)

	return &types.ChatResp{
		Reply:     rpcResp.Reply,
		SessionId: rpcResp.SessionId,
	}, nil
}
