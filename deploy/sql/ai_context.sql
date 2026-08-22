-- 警告：本脚本会删除 aichat_chat_db 中现有的 Chat 会话和消息数据。
-- 仅在确认不需要保留历史数据的环境中手动执行。
CREATE DATABASE IF NOT EXISTS aichat_chat_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE aichat_chat_db;

DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS conversations;

CREATE TABLE conversations (
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id     BIGINT NOT NULL COMMENT '用户ID',
    session_id  VARCHAR(64) NOT NULL COMMENT '客户端会话ID',
    title       VARCHAR(255) DEFAULT '新对话' COMMENT '会话标题',
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    UNIQUE KEY uk_user_session (user_id, session_id),
    INDEX idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户会话表';

CREATE TABLE messages (
    id              BIGINT PRIMARY KEY AUTO_INCREMENT,
    conversation_id BIGINT NOT NULL COMMENT '会话ID',
    role            TINYINT NOT NULL COMMENT '角色: 0-用户, 1-AI助手',
    content         TEXT NOT NULL COMMENT '消息内容',
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_conversation_id (conversation_id, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话消息表';
