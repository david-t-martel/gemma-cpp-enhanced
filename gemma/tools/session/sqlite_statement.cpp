// Implementation of RAII sqlite3 statement wrapper
#include "sqlite_statement.h"
#include <sqlite3.h>

namespace gemma { namespace session {

SqliteStatement::SqliteStatement(sqlite3* db, const char* sql) noexcept : db_(db) {
    if (db_ && sql) {
        rc_ = sqlite3_prepare_v2(db_, sql, -1, &stmt_, nullptr);
    }
}

SqliteStatement::~SqliteStatement() { finalize(); }

void SqliteStatement::finalize() noexcept {
    if (stmt_) {
        sqlite3_finalize(stmt_);
        stmt_ = nullptr;
    }
}

void SqliteStatement::move_from(SqliteStatement& other) noexcept {
    db_ = other.db_;
    stmt_ = other.stmt_;
    rc_ = other.rc_;
    other.db_ = nullptr;
    other.stmt_ = nullptr;
    other.rc_ = 0;
}

int SqliteStatement::bind_text(int index, const std::string& value) noexcept {
    if (!stmt_) return SQLITE_MISUSE;
    return sqlite3_bind_text(stmt_, index, value.c_str(), -1, SQLITE_TRANSIENT);
}

int SqliteStatement::bind_int64(int index, long long value) noexcept {
    if (!stmt_) return SQLITE_MISUSE;
    return sqlite3_bind_int64(stmt_, index, value);
}

int SqliteStatement::bind_int(int index, int value) noexcept {
    if (!stmt_) return SQLITE_MISUSE;
    return sqlite3_bind_int(stmt_, index, value);
}

int SqliteStatement::step() noexcept {
    if (!stmt_) return SQLITE_MISUSE;
    return sqlite3_step(stmt_);
}

const unsigned char* SqliteStatement::column_text(int col) const noexcept {
    if (!stmt_) return nullptr;
    return sqlite3_column_text(stmt_, col);
}

long long SqliteStatement::column_int64(int col) const noexcept {
    if (!stmt_) return 0;
    return sqlite3_column_int64(stmt_, col);
}

} } // namespace gemma::session
