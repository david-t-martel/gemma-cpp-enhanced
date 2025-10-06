#include "lru_cache.h"
#include "Session.h"
#include <stdexcept>

namespace gemma { namespace session {

LRUCache::LRUCache(size_t capacity) : capacity_(capacity) {
    if (capacity_ == 0) throw std::invalid_argument("LRUCache capacity must be > 0");
}

std::shared_ptr<Session> LRUCache::get(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(session_id);
    if (it == map_.end()) return nullptr;
    list_.splice(list_.begin(), list_, it->second);
    return it->second->session;
}

void LRUCache::put(const std::string& session_id, std::shared_ptr<Session> session) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(session_id);
    if (it != map_.end()) {
        it->second->session = std::move(session);
        list_.splice(list_.begin(), list_, it->second);
        return;
    }
    if (list_.size() >= capacity_) evict();
    list_.emplace_front(Node{session_id, std::move(session)});
    map_[session_id] = list_.begin();
}

void LRUCache::remove(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(session_id);
    if (it != map_.end()) { list_.erase(it->second); map_.erase(it); }
}

void LRUCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    list_.clear();
    map_.clear();
}

size_t LRUCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return list_.size();
}

void LRUCache::evict() {
    if (list_.empty()) return;
    auto &back = list_.back();
    map_.erase(back.session_id);
    list_.pop_back();
}

} } // namespace gemma::session
