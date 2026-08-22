// Code scaffolded by goctl. Safe to edit.
// goctl 1.9.2

package user

import (
	"net/http"

	"aiChat/app/gateway/api/internal/logic/user"
	"aiChat/app/gateway/api/internal/svc"
	"aiChat/app/gateway/api/internal/types"
	"aiChat/pkg/ctxdata"

	"github.com/google/uuid"
	"github.com/zeromicro/go-zero/rest/httpx"
)

// chat
func ChatHandler(svcCtx *svc.ServiceContext) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req types.ChatReq
		if err := httpx.Parse(r, &req); err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
			return
		}

		requestId := r.Header.Get("X-Request-Id")
		if requestId == "" {
			requestId = uuid.NewString()
		}
		ctx := ctxdata.WithRequestID(r.Context(), requestId)

		l := user.NewChatLogic(ctx, svcCtx)
		resp, err := l.Chat(&req)
		if err != nil {
			httpx.ErrorCtx(ctx, w, err)
		} else {
			w.Header().Set("X-Request-Id", requestId)
			httpx.OkJsonCtx(ctx, w, resp)
		}
	}
}
