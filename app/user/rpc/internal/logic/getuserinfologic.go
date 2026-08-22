package logic

import (
	"context"

	"aiChat/app/user/rpc/internal/svc"
	"aiChat/app/user/rpc/pb/user"

	"github.com/pkg/errors"
	"github.com/zeromicro/go-zero/core/logx"
)

type GetUserInfoLogic struct {
	ctx    context.Context
	svcCtx *svc.ServiceContext
	logx.Logger
}

func NewGetUserInfoLogic(ctx context.Context, svcCtx *svc.ServiceContext) *GetUserInfoLogic {
	return &GetUserInfoLogic{
		ctx:    ctx,
		svcCtx: svcCtx,
		Logger: logx.WithContext(ctx),
	}
}

func (l *GetUserInfoLogic) GetUserInfo(in *user.UserInfoRequest) (*user.UserInfoResponse, error) {
	userRecord, err := l.svcCtx.UserModel.FindOne(l.ctx, uint64(in.UserId))
	if err != nil {
		return nil, errors.New("用户不存在")
	}

	return &user.UserInfoResponse{
		Id:       int64(userRecord.Id),
		Mobile:   userRecord.Mobile,
		Nickname: userRecord.Nickname,
		Sex:      int64(userRecord.Sex),
	}, nil
}
