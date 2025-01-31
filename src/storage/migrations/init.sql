-- ストレージ層の初期化スクリプト

-- 拡張機能の有効化
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 音声ファイルメタデータテーブル
CREATE TABLE IF NOT EXISTS audio_files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    duration INTEGER,
    sample_rate INTEGER,
    channels SMALLINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    storage_path VARCHAR(512) NOT NULL,
    status VARCHAR(50) NOT NULL,
    metadata JSONB
);

-- 解析結果テーブル
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    audio_file_id UUID REFERENCES audio_files(id),
    analysis_type VARCHAR(50) NOT NULL,
    result_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(100),
    processing_time FLOAT
);

-- バックアップ履歴テーブル
CREATE TABLE IF NOT EXISTS backups (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    backup_date TIMESTAMP WITH TIME ZONE NOT NULL,
    backup_size BIGINT NOT NULL,
    file_count INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL,
    storage_path VARCHAR(512) NOT NULL,
    metadata JSONB
);

-- アクセス統計テーブル
CREATE TABLE IF NOT EXISTS access_stats (
    id SERIAL PRIMARY KEY,
    file_id UUID REFERENCES audio_files(id),
    access_type VARCHAR(50) NOT NULL,
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    client_info JSONB
);

-- 操作ログテーブル
CREATE TABLE IF NOT EXISTS operation_logs (
    id SERIAL PRIMARY KEY,
    operation_type VARCHAR(50) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(50) NOT NULL,
    details JSONB
);

-- エラーログテーブル
CREATE TABLE IF NOT EXISTS error_logs (
    id SERIAL PRIMARY KEY,
    error_type VARCHAR(50) NOT NULL,
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    details JSONB
);

-- インデックスの作成
CREATE INDEX IF NOT EXISTS idx_audio_files_created_at ON audio_files(created_at);
CREATE INDEX IF NOT EXISTS idx_audio_files_status ON audio_files(status);
CREATE INDEX IF NOT EXISTS idx_audio_files_updated_at ON audio_files(updated_at);
CREATE INDEX IF NOT EXISTS idx_analysis_results_audio_file_id ON analysis_results(audio_file_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_type ON analysis_results(analysis_type);
CREATE INDEX IF NOT EXISTS idx_access_stats_file_id ON access_stats(file_id);
CREATE INDEX IF NOT EXISTS idx_access_stats_accessed_at ON access_stats(accessed_at);
CREATE INDEX IF NOT EXISTS idx_operation_logs_start_time ON operation_logs(start_time);
CREATE INDEX IF NOT EXISTS idx_error_logs_timestamp ON error_logs(timestamp);

-- パーティショニングの設定
-- アクセス統計の月次パーティション
CREATE TABLE IF NOT EXISTS access_stats_partitioned (
    LIKE access_stats INCLUDING ALL
) PARTITION BY RANGE (accessed_at);

-- 初期パーティションの作成（例：2024年1月から12月）
DO $$
BEGIN
    FOR month IN 1..12 LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS access_stats_y2024m%s PARTITION OF access_stats_partitioned
            FOR VALUES FROM (%L) TO (%L)',
            LPAD(month::text, 2, '0'),
            format('2024-%s-01', LPAD(month::text, 2, '0')),
            format('2024-%s-01', LPAD((month + 1)::text, 2, '0'))
        );
    END LOOP;
END $$;

-- 更新トリガーの作成
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_audio_files_updated_at
    BEFORE UPDATE ON audio_files
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- メトリクス集計用ビューの作成
CREATE OR REPLACE VIEW storage_metrics AS
SELECT
    COUNT(*) as total_files,
    SUM(file_size) as total_size,
    COUNT(CASE WHEN status = 'active' THEN 1 END) as active_files,
    COUNT(CASE WHEN status = 'archived' THEN 1 END) as archived_files,
    COUNT(CASE WHEN status = 'deleted' THEN 1 END) as deleted_files
FROM audio_files;

CREATE OR REPLACE VIEW daily_access_stats AS
SELECT
    DATE(accessed_at) as access_date,
    access_type,
    COUNT(*) as access_count
FROM access_stats
GROUP BY DATE(accessed_at), access_type;

-- 権限の設定
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO current_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO current_user;