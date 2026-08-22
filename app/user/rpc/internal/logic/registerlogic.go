package logic

import (
	"context"

	"aiChat/app/user/rpc/internal/model"
	"aiChat/app/user/rpc/internal/svc"
	"aiChat/app/user/rpc/pb/user"
	"aiChat/pkg/jwt"

	"github.com/pkg/errors"
	"github.com/zeromicro/go-zero/core/logx"
	"golang.org/x/crypto/bcrypt"
)

type RegisterLogic struct {
	ctx    context.Context
	svcCtx *svc.ServiceContext
	logx.Logger
}

func NewRegisterLogic(ctx context.Context, svcCtx *svc.ServiceContext) *RegisterLogic {
	return &RegisterLogic{
		ctx:    ctx,
		svcCtx: svcCtx,
		Logger: logx.WithContext(ctx),
	}
}

func (l *RegisterLogic) Register(in *user.RegisterRequest) (*user.RegisterResponse, error) {

	// 1. 校验手机号是否已注册
	user_exam, err := l.svcCtx.UserModel.FindOneByMobile(l.ctx, in.Mobile)
	if err != nil && err != model.ErrNotFound {
		logx.Errorf("ErrNotFound, mobile:%s,err:%v", in.Mobile, err)
		return nil, err
	}
	if user_exam != nil {
		logx.Errorf("Register user exists mobile:%s,err:%v", in.Mobile, err)
		return nil, errors.New("user has been registered")
	}

	// 2. 密码加密
	hashedPwd, err := bcrypt.GenerateFromPassword([]byte(in.Password), bcrypt.DefaultCost)
	if err != nil {
		logx.Errorf("hash password err, password:%s,err:%v", in.Password, err)
		return nil, err
	}

	// 3. 注册用户
	newUser := &model.Users{
		Mobile:   in.Mobile,
		Password: string(hashedPwd),
		Nickname: in.Mobile,
		Sex:      0,
		Status:   1,
	}
	result, err := l.svcCtx.UserModel.Insert(l.ctx, newUser)
	if err != nil {
		return nil, err
	}
	userId, _ := result.LastInsertId()

	// 4. 生成token
	jwtCfg := jwt.Config{
		AccessSecret: l.svcCtx.Config.JwtAuth.AccessSecret,
		AccessExpire: l.svcCtx.Config.JwtAuth.AccessExpire,
	}

	token, err := jwt.GenerateToken(jwtCfg, userId)
	if err != nil {
		logx.Errorf("GenerateToken, userId:%d,err:%v", userId, err)
		return nil, err
	}

	return &user.RegisterResponse{
		UserId:       userId,
		AccessToken:  token.AccessToken,
		AccessExpire: token.AccessExpire,
		RefreshAfter: token.RefreshAfter,
	}, nil
}
