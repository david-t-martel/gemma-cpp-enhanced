// Interface for session caching layer (Phase 1 refactor)
#pragma once

#include <memory>
#include <string>

namespace gemma { namespace session {

class Session;

class ISessionCache {
public:
    virtual ~ISessionCache() = default;
    virtual std::shared_ptr<Session> get(const std::string& session_id) = 0;
    virtual void put(const std::string& session_id, std::shared_ptr<Session> session) = 0;
    virtual void remove(const std::string& session_id) = 0;
    virtual void clear() = 0;
    virtual size_t size() const = 0;
    virtual size_t capacity() const = 0;
};

} } // namespace gemma::session
