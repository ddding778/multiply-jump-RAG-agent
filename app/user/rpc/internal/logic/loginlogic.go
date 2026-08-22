package logic

import (
	"context"

	"aiChat/app/user/rpc/internal/svc"
	"aiChat/app/user/rpc/pb/user"
	"aiChat/pkg/jwt"

	"github.com/pkg/errors"
	"github.com/zeromicro/go-zero/core/logx"
	"golang.org/x/crypto/bcrypt"
)

type LoginLogic struct {
	ctx    context.Context
	svcCtx *svc.ServiceContext
	logx.Logger
}

func NewLoginLogic(ctx context.Context, svcCtx *svc.ServiceContext) *LoginLogic {
	return &LoginLogic{
		ctx:    ctx,
		svcCtx: svcCtx,
		Logger: logx.WithContext(ctx),
	}
}

func (l *LoginLogic) Login(in *user.LoginRequest) (*user.LoginResponse, error) {
	// 1. 根据手机号查询用户
	userRecord, err := l.svcCtx.UserModel.FindOneByMobile(l.ctx, in.Mobile)
	if err != nil {
		return nil, errors.New("手机号或密码错误")
	}

	// 2. 密码校验
	err = bcrypt.CompareHashAndPassword([]byte(userRecord.Password), []byte(in.Password))
	if err != nil {
		return nil, errors.New("手机号或密码错误")
	}

	// 3. 登录成功
	token, err := jwt.GenerateToken(l.svcCtx.Config.JwtAuth, int64(userRecord.Id))
	if err != nil {
		return nil, err
	}

	return &user.LoginResponse{
		UserId:       int64(userRecord.Id),
		AccessToken:  token.AccessToken,
		AccessExpire: token.AccessExpire,
		RefreshAfter: token.RefreshAfter,
	}, nil
}
