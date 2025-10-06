#pragma once

#include "i_session_cache.h"
#include <list>
#include <unordered_map>
#include <mutex>

namespace gemma { namespace session {

class LRUCache : public ISessionCache {
public:
    explicit LRUCache(size_t capacity);
    ~LRUCache() override = default;

    std::shared_ptr<Session> get(const std::string& session_id) override;
    void put(const std::string& session_id, std::shared_ptr<Session> session) override;
    void remove(const std::string& session_id) override;
    void clear() override;
    size_t size() const override;
    size_t capacity() const override { return capacity_; }

private:
    struct Node { std::string session_id; std::shared_ptr<Session> session; };
    size_t capacity_;
    std::list<Node> list_;
    std::unordered_map<std::string, std::list<Node>::iterator> map_;
    mutable std::mutex mutex_;
    void evict();
};

} } // namespace gemma::session
