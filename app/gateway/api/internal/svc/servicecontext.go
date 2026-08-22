// Code scaffolded by goctl. Safe to edit.
// goctl 1.9.2

package svc

import (
	aipb "aiChat/app/ai/rpc/pb"
	"aiChat/app/gateway/api/internal/config"
	"aiChat/app/user/rpc/pb/user"

	"github.com/zeromicro/go-zero/zrpc"
)

type ServiceContext struct {
	Config  config.Config
	AiRpc   aipb.AiRpcClient
	UserRpc user.UserRpcClient
}

func NewServiceContext(c config.Config) *ServiceContext {
	return &ServiceContext{
		Config:  c,
		AiRpc:   aipb.NewAiRpcClient(zrpc.MustNewClient(c.AiRpc).Conn()),
		UserRpc: user.NewUserRpcClient(zrpc.MustNewClient(c.UserRpc).Conn()),
	}
}
