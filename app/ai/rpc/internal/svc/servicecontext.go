package svc

import (
	"aiChat/app/ai/rpc/internal/config"
	"aiChat/app/ai/rpc/internal/model"
	rag "aiChat/app/ai/rpc/internal/ragservice"

	"github.com/zeromicro/go-zero/core/logx"
	"github.com/zeromicro/go-zero/core/stores/cache"
	"github.com/zeromicro/go-zero/core/stores/redis"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
	"github.com/zeromicro/go-zero/core/syncx"
)

type ServiceContext struct {
	Config             config.Config
	ConversationsModel model.ConversationsModel // 对话历史模型
	MessagesModel      model.MessagesModel      // 消息模型
	Cache              cache.Cache
	RedisClient        *redis.Redis
	RagClient          *rag.Client
}

func NewServiceContext(c config.Config) *ServiceContext {
	logx.SetUp(c.Log)
	conn := sqlx.NewMysql(c.DataSource)
	cacher := cache.New(c.CacheRedis, syncx.NewSingleFlight(), &cache.Stat{}, nil)
	redisClient := redis.MustNewRedis(c.RedisConf)

	return &ServiceContext{
		Config:             c,
		ConversationsModel: model.NewConversationsModel(conn, c.CacheRedis),
		MessagesModel:      model.NewMessagesModel(conn, c.CacheRedis),
		Cache:              cacher,
		RedisClient:        redisClient,
		RagClient:          rag.NewClient(c.RAG.EmbeddingUrl),
	}
}
