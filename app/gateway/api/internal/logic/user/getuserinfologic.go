// Code scaffolded by goctl. Safe to edit.
// goctl 1.9.2

package user

import (
	"context"

	"aiChat/app/gateway/api/internal/svc"
	"aiChat/app/gateway/api/internal/types"
	"aiChat/app/user/rpc/pb/user"
	"aiChat/pkg/ctxdata"

	"github.com/zeromicro/go-zero/core/logx"
)

type GetUserInfoLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

// get user info
func NewGetUserInfoLogic(ctx context.Context, svcCtx *svc.ServiceContext) *GetUserInfoLogic {
	return &GetUserInfoLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *GetUserInfoLogic) GetUserInfo(req *types.UserInfoReq) (resp *types.UserInfoResp, err error) {
	userId := ctxdata.GetUidFromCtx(l.ctx)
	rpcResp, err := l.svcCtx.UserRpc.GetUserInfo(l.ctx, &user.UserInfoRequest{
		UserId: userId,
	})
	if err != nil {
		return nil, err

	}
	return &types.UserInfoResp{
		UserId:   userId,
		Mobile:   rpcResp.Mobile,
		Nickname: rpcResp.Nickname,
		Sex:      rpcResp.Sex,
	}, nil
}
