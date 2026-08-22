package ctxdata

import (
	"context"
	"encoding/json"

	"github.com/google/uuid"
	"github.com/zeromicro/go-zero/core/logx"
)

// CtxKeyJwtUserId get uid from ctx
var (
	CtxKeyJwtUserId = "userId"
	CtxKeyRequestID = "request_id"
)

func GetUidFromCtx(ctx context.Context) int64 {
	var uid int64
	if jsonUid, ok := ctx.Value(CtxKeyJwtUserId).(json.Number); ok {
		if int64Uid, err := jsonUid.Int64(); err == nil {
			uid = int64Uid
		} else {
			logx.WithContext(ctx).Errorf("GetUidFromCtx err : %+v", err)
		}
	}
	return uid
}

func WithRequestID(ctx context.Context, requestId string) context.Context {
	return context.WithValue(ctx, CtxKeyRequestID, requestId)
}

func GetRequestID(ctx context.Context) string {
	if v, ok := ctx.Value(CtxKeyRequestID).(string); ok && v != "" {
		return v
	}
	return uuid.NewString()
}
