// RAII wrapper for sqlite3 prepared statements (incremental refactor step)
#pragma once

struct sqlite3;
struct sqlite3_stmt;

#include <string>

namespace gemma { namespace session {

class SqliteStatement {
public:
    SqliteStatement(sqlite3* db, const char* sql) noexcept;
    ~SqliteStatement();

    SqliteStatement(const SqliteStatement&) = delete;
    SqliteStatement& operator=(const SqliteStatement&) = delete;

    SqliteStatement(SqliteStatement&& other) noexcept { move_from(other); }
    SqliteStatement& operator=(SqliteStatement&& other) noexcept {
        if (this != &other) { finalize(); move_from(other); }
        return *this; }

    bool valid() const noexcept { return stmt_ != nullptr; }
    int rc() const noexcept { return rc_; }
    sqlite3_stmt* get() const noexcept { return stmt_; }

    int bind_text(int index, const std::string& value) noexcept;
    int bind_int64(int index, long long value) noexcept;
    int bind_int(int index, int value) noexcept;

    int step() noexcept; // wraps sqlite3_step

    const unsigned char* column_text(int col) const noexcept;
    long long column_int64(int col) const noexcept;

private:
    sqlite3* db_ = nullptr;
    sqlite3_stmt* stmt_ = nullptr;
    int rc_ = 0;

    void finalize() noexcept;
    void move_from(SqliteStatement& other) noexcept;
};

} } // namespace gemma::session
