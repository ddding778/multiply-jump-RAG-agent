FROM golang:1.21-alpine AS builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o user-rpc ./app/user/rpc/user.go

FROM alpine:latest
RUN apk --no-cache add ca-certificates tzdata
WORKDIR /app
COPY --from=builder /build/user-rpc .
COPY app/user/rpc/etc /app/etc
ENTRYPOINT ["./user-rpc", "-f", "etc/user.yaml"]