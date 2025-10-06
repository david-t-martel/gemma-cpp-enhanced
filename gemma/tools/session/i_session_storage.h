// Abstract storage interface (Phase 1 refactor step) decoupling SessionManager
// from concrete SessionStorage implementation.
#pragma once

#include <memory>
#include <string>
#include <vector>

// No JSON header dependency here—use serialized string payloads at the boundary.

namespace gemma {
namespace session {

class Session; // forward declaration

class ISessionStorage {
public:
    virtual ~ISessionStorage() = default;

    // Lifecycle
    virtual bool initialize() = 0;
    virtual void close() = 0;

    // CRUD
    virtual bool save_session(std::shared_ptr<Session> session) = 0;
    virtual std::shared_ptr<Session> load_session(const std::string& session_id) = 0;
    virtual bool delete_session(const std::string& session_id) = 0;
    virtual bool session_exists(const std::string& session_id) = 0;

    // Listing / export
    virtual std::vector<std::string> list_sessions() = 0;
    virtual std::vector<std::string> list_sessions(size_t limit, size_t offset,
                                                   const std::string& sort_by,
                                                   bool ascending) = 0;
    virtual bool export_to_json(const std::string& file_path) = 0;
    virtual bool import_from_json(const std::string& file_path, bool overwrite_existing) = 0;

    // Stats
    virtual std::string get_statistics() = 0; // serialized JSON

    // Maintenance
    virtual size_t cleanup_expired_sessions() = 0;
};

} // namespace session
} // namespace gemma
