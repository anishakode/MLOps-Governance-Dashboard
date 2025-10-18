-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas for organization
CREATE SCHEMA IF NOT EXISTS governance;
CREATE SCHEMA IF NOT EXISTS audit;

-- Grant permissions
GRANT ALL ON SCHEMA governance TO mlops_user;
GRANT ALL ON SCHEMA audit TO mlops_user;

-- Create audit log table
CREATE TABLE audit.audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(255),
    operation VARCHAR(10),
    old_values JSONB,
    new_values JSONB,
    changed_by VARCHAR(255),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_audit_log_table_name ON audit.audit_log(table_name);
CREATE INDEX idx_audit_log_changed_at ON audit.audit_log(changed_at);