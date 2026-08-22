// Code scaffolded by goctl. Safe to edit.
// goctl 1.9.2

package user

import (
	"context"

	"aiChat/app/gateway/api/internal/svc"
	"aiChat/app/gateway/api/internal/types"
	"aiChat/app/user/rpc/pb/user"

	"github.com/zeromicro/go-zero/core/logx"
)

type RegisterLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

// register
func NewRegisterLogic(ctx context.Context, svcCtx *svc.ServiceContext) *RegisterLogic {
	return &RegisterLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *RegisterLogic) Register(req *types.RegisterReq) (resp *types.RegisterResp, err error) {
	rpcResp, err := l.svcCtx.UserRpc.Register(l.ctx, &user.RegisterRequest{
		Mobile:   req.Mobile,
		Password: req.Password,
	})
	if err != nil {
		logx.Errorf("Register err: %v", err)
		return nil, err
	}

	return &types.RegisterResp{
		AccessToken:  rpcResp.AccessToken,
		AccessExpire: rpcResp.AccessExpire,
		RefreshAfter: rpcResp.RefreshAfter,
	}, nil
}
