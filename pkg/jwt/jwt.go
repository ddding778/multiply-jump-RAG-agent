package jwt

import (
	"errors"
	"time"

	"github.com/golang-jwt/jwt/v4"
)

type Config struct {
	AccessSecret string
	AccessExpire int64 // 单位：秒
}

type TokenResult struct {
	AccessToken  string
	AccessExpire int64 // 过期时间戳（秒）
	RefreshAfter int64 // 建议刷新时间戳（秒）
}

func GenerateToken(cfg Config, userId int64) (*TokenResult, error) {
	if cfg.AccessSecret == "" || cfg.AccessExpire <= 0 {
		return nil, errors.New("invalid jwt config")
	}
	now := time.Now().Unix()
	claims := jwt.MapClaims{
		"userId": userId,
		"exp":    now + cfg.AccessExpire,
		"iat":    now,
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenString, err := token.SignedString([]byte(cfg.AccessSecret))
	if err != nil {
		return nil, err
	}
	return &TokenResult{
		AccessToken:  tokenString,
		AccessExpire: now + cfg.AccessExpire,
		RefreshAfter: now + cfg.AccessExpire/2,
	}, nil
}
