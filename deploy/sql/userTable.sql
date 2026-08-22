-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS `aichat_user_db` 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

-- 使用该数据库
USE `aichat_user_db`;

-- 创建用户表
CREATE TABLE `users` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '用户ID',
    `mobile` VARCHAR(20) NOT NULL COMMENT '手机号',
    `password` VARCHAR(255) NOT NULL COMMENT '加密密码',
    `nickname` VARCHAR(64) NOT NULL DEFAULT '' COMMENT '昵称',
    `sex` TINYINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '性别:0-未知,1-男,2-女',
    `status` TINYINT UNSIGNED NOT NULL DEFAULT 1 COMMENT '状态:1-正常,0-禁用',
    `last_login_at` DATETIME DEFAULT NULL COMMENT '最后登录时间',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uniq_mobile` (`mobile`),
    KEY `idx_status` (`status`),
    KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户基本信息表';