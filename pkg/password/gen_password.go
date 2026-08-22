package genpassword

import (
	"crypto/rand"
	"math/big"
)

const (
	lowerChars  = "abcdefghijklmnopqrstuvwxyz"
	upperChars  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	digitChars  = "0123456789"
	symbolChars = "!@#$%^&*()-_=+[]{}<>?"
	allChars    = lowerChars + upperChars + digitChars + symbolChars
)

// GenerateRandomPassword 生成指定长度的随机密码（默认16位）
func GenerateRandomPassword(length int) (string, error) {
	if length < 8 {
		length = 8 // 最小长度保护
	}
	result := make([]byte, length)
	for i := 0; i < length; i++ {
		// 从 allChars 中随机取一个字符
		n, err := rand.Int(rand.Reader, big.NewInt(int64(len(allChars))))
		if err != nil {
			return "", err
		}
		result[i] = allChars[n.Int64()]
	}
	// 可选：确保至少包含一位数字、一位大写、一位小写、一位符号（此处略，概率足够高）
	return string(result), nil
}
