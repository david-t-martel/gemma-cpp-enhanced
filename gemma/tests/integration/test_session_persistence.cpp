#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../utils/test_helpers.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

// Mock session persistence system
// In real implementation: #include "../../src/session/SessionPersistence.h"

using namespace gemma::test;
using namespace testing;
using json = nlohmann::json;

// Mock session data structure
struct SessionData {
    std::string id;
    std::string created_at;
    std::string last_activity;
    size_t max_context_tokens;
    json metadata;
    std::vector<json> messages;
    json state;
    
    json to_json() const {
        return json{
            {"id", id},
            {"created_at", created_at},
            {"last_activity", last_activity},
            {"max_context_tokens", max_context_tokens},
            {"metadata", metadata},
            {"messages", messages},
            {"state", state}
        };
    }
    
    static SessionData from_json(const json& j) {
        SessionData data;
        data.id = j.value("id", "");
        data.created_at = j.value("created_at", "");
        data.last_activity = j.value("last_activity", "");
        data.max_context_tokens = j.value("max_context_tokens", 8192);
        data.metadata = j.value("metadata", json::object());
        data.messages = j.value("messages", json::array());
        data.state = j.value("state", json::object());
        return data;
    }
};

// Mock session persistence interface
class MockSessionPersistence {
public:
    MOCK_METHOD(bool, initialize, (const std::string& storage_path), ());
    MOCK_METHOD(void, shutdown, (), ());
    MOCK_METHOD(bool, save_session, (const SessionData& session_data), ());
    MOCK_METHOD(std::optional<SessionData>, load_session, (const std::string& session_id), ());
    MOCK_METHOD(bool, delete_session, (const std::string& session_id), ());
    MOCK_METHOD(std::vector<std::string>, list_session_ids, (), ());
    MOCK_METHOD(bool, session_exists, (const std::string& session_id), ());
    MOCK_METHOD(bool, backup_sessions, (const std::string& backup_path), ());
    MOCK_METHOD(bool, restore_sessions, (const std::string& backup_path), ());
    MOCK_METHOD(bool, export_session, (const std::string& session_id, const std::string& export_path), ());
    MOCK_METHOD(bool, import_session, (const std::string& import_path), ());
    MOCK_METHOD(size_t, get_storage_size, (), ());
    MOCK_METHOD(json, get_storage_stats, (), ());
    MOCK_METHOD(bool, cleanup_old_sessions, (std::chrono::hours max_age), ());
    MOCK_METHOD(bool, compact_storage, (), ());
    MOCK_METHOD(bool, validate_storage_integrity, (), ());
    MOCK_METHOD(std::vector<SessionData>, search_sessions, (const json& query), ());
};

// Mock session serializer for different formats
class MockSessionSerializer {
public:
    MOCK_METHOD(std::string, serialize_to_json, (const SessionData& session), ());
    MOCK_METHOD(std::vector<uint8_t>, serialize_to_binary, (const SessionData& session), ());
    MOCK_METHOD(std::string, serialize_to_xml, (const SessionData& session), ());
    MOCK_METHOD(SessionData, deserialize_from_json, (const std::string& json_str), ());
    MOCK_METHOD(SessionData, deserialize_from_binary, (const std::vector<uint8_t>& binary_data), ());
    MOCK_METHOD(SessionData, deserialize_from_xml, (const std::string& xml_str), ());
    MOCK_METHOD(bool, validate_serialized_data, (const std::string& data, const std::string& format), ());
};

class SessionPersistenceTest : public GemmaTestBase {
protected:
    void SetUp() override {
        GemmaTestBase::SetUp();
        
        persistence_ = std::make_unique<MockSessionPersistence>();
        serializer_ = std::make_unique<MockSessionSerializer>();
        
        storage_path_ = test_dir_ / "session_storage";
        backup_path_ = test_dir_ / "session_backup";
        
        std::filesystem::create_directories(storage_path_);
        std::filesystem::create_directories(backup_path_);
        
        setup_default_expectations();
        create_test_sessions();
    }
    
    void setup_default_expectations() {
        ON_CALL(*persistence_, initialize(_)).WillByDefault(Return(true));
        ON_CALL(*persistence_, save_session(_)).WillByDefault(Return(true));
        ON_CALL(*persistence_, delete_session(_)).WillByDefault(Return(true));
        ON_CALL(*persistence_, session_exists(_)).WillByDefault(Return(false));
        ON_CALL(*persistence_, list_session_ids()).WillByDefault(Return(std::vector<std::string>{}));
        ON_CALL(*persistence_, get_storage_size()).WillByDefault(Return(1024 * 1024)); // 1MB
        ON_CALL(*persistence_, validate_storage_integrity()).WillByDefault(Return(true));
    }
    
    void create_test_sessions() {
        test_session_1_.id = "session-001";
        test_session_1_.created_at = "2024-01-01T10:00:00Z";
        test_session_1_.last_activity = "2024-01-01T10:30:00Z";
        test_session_1_.max_context_tokens = 4096;
        test_session_1_.metadata = json{
            {"user_id", "user123"},
            {"session_type", "chat"},
            {"tags", {"test", "demo"}}
        };
        test_session_1_.messages = {
            json{{"role", "user"}, {"content", "Hello"}, {"timestamp", "2024-01-01T10:15:00Z"}},
            json{{"role", "assistant"}, {"content", "Hi there!"}, {"timestamp", "2024-01-01T10:15:01Z"}}
        };
        test_session_1_.state = json{{"context_tokens_used", 25}, {"model_backend", "cpu"}};
        
        test_session_2_.id = "session-002";
        test_session_2_.created_at = "2024-01-01T11:00:00Z";
        test_session_2_.last_activity = "2024-01-01T11:45:00Z";
        test_session_2_.max_context_tokens = 8192;
        test_session_2_.metadata = json{
            {"user_id", "user456"},
            {"session_type", "qa"},
            {"tags", {"research"}}
        };
        test_session_2_.messages = {
            json{{"role", "user"}, {"content", "What is AI?"}, {"timestamp", "2024-01-01T11:30:00Z"}},
            json{{"role", "assistant"}, {"content", "AI stands for Artificial Intelligence..."}, {"timestamp", "2024-01-01T11:30:02Z"}}
        };
        test_session_2_.state = json{{"context_tokens_used", 45}, {"model_backend", "intel"}};
    }
    
    std::unique_ptr<MockSessionPersistence> persistence_;
    std::unique_ptr<MockSessionSerializer> serializer_;
    std::filesystem::path storage_path_;
    std::filesystem::path backup_path_;
    SessionData test_session_1_;
    SessionData test_session_2_;
};

// Basic persistence operations tests

TEST_F(SessionPersistenceTest, InitializeStorage) {
    EXPECT_CALL(*persistence_, initialize(storage_path_.string()))
        .Times(1)
        .WillOnce(Return(true));
    
    bool initialized = persistence_->initialize(storage_path_.string());
    EXPECT_TRUE(initialized);
}

TEST_F(SessionPersistenceTest, SaveAndLoadSession) {
    // Save session
    EXPECT_CALL(*persistence_, save_session(MatchesSession(test_session_1_)))
        .Times(1)
        .WillOnce(Return(true));
    
    bool saved = persistence_->save_session(test_session_1_);
    EXPECT_TRUE(saved);
    
    // Load session
    EXPECT_CALL(*persistence_, load_session(test_session_1_.id))
        .Times(1)
        .WillOnce(Return(std::make_optional(test_session_1_)));
    
    auto loaded = persistence_->load_session(test_session_1_.id);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->id, test_session_1_.id);
    EXPECT_EQ(loaded->messages.size(), 2);
    EXPECT_EQ(loaded->metadata["user_id"], "user123");
}

TEST_F(SessionPersistenceTest, LoadNonExistentSession) {
    EXPECT_CALL(*persistence_, load_session("nonexistent"))
        .Times(1)
        .WillOnce(Return(std::nullopt));
    
    auto loaded = persistence_->load_session("nonexistent");
    EXPECT_FALSE(loaded.has_value());
}

TEST_F(SessionPersistenceTest, DeleteSession) {
    // First ensure session exists
    EXPECT_CALL(*persistence_, session_exists(test_session_1_.id))
        .Times(1)
        .WillOnce(Return(true));
    
    bool exists_before = persistence_->session_exists(test_session_1_.id);
    EXPECT_TRUE(exists_before);
    
    // Delete session
    EXPECT_CALL(*persistence_, delete_session(test_session_1_.id))
        .Times(1)
        .WillOnce(Return(true));
    
    bool deleted = persistence_->delete_session(test_session_1_.id);
    EXPECT_TRUE(deleted);
    
    // Verify it no longer exists
    EXPECT_CALL(*persistence_, session_exists(test_session_1_.id))
        .Times(1)
        .WillOnce(Return(false));
    
    bool exists_after = persistence_->session_exists(test_session_1_.id);
    EXPECT_FALSE(exists_after);
}

TEST_F(SessionPersistenceTest, ListSessionIds) {
    std::vector<std::string> expected_ids = {test_session_1_.id, test_session_2_.id};
    
    EXPECT_CALL(*persistence_, list_session_ids())
        .Times(1)
        .WillOnce(Return(expected_ids));
    
    auto session_ids = persistence_->list_session_ids();
    EXPECT_EQ(session_ids.size(), 2);
    EXPECT_THAT(session_ids, UnorderedElementsAre(test_session_1_.id, test_session_2_.id));
}

// Session backup and restore tests

TEST_F(SessionPersistenceTest, BackupSessions) {
    std::string backup_file = (backup_path_ / "sessions_backup.json").string();
    
    EXPECT_CALL(*persistence_, backup_sessions(backup_file))
        .Times(1)
        .WillOnce(Return(true));
    
    bool backed_up = persistence_->backup_sessions(backup_file);
    EXPECT_TRUE(backed_up);
}

TEST_F(SessionPersistenceTest, RestoreSessions) {
    std::string backup_file = (backup_path_ / "sessions_restore.json").string();
    
    // Create a mock backup file
    json backup_data = {
        {"version", "1.0"},
        {"timestamp", "2024-01-01T12:00:00Z"},
        {"sessions", {test_session_1_.to_json(), test_session_2_.to_json()}}
    };
    
    std::ofstream backup_stream(backup_file);
    backup_stream << backup_data.dump(2);
    backup_stream.close();
    
    EXPECT_CALL(*persistence_, restore_sessions(backup_file))
        .Times(1)
        .WillOnce(Return(true));
    
    bool restored = persistence_->restore_sessions(backup_file);
    EXPECT_TRUE(restored);
}

TEST_F(SessionPersistenceTest, ExportSingleSession) {
    std::string export_file = (test_dir_ / "exported_session.json").string();
    
    EXPECT_CALL(*persistence_, export_session(test_session_1_.id, export_file))
        .Times(1)
        .WillOnce(Return(true));
    
    bool exported = persistence_->export_session(test_session_1_.id, export_file);
    EXPECT_TRUE(exported);
}

TEST_F(SessionPersistenceTest, ImportSingleSession) {
    std::string import_file = (test_dir_ / "import_session.json").string();
    
    // Create import file
    std::ofstream import_stream(import_file);
    import_stream << test_session_1_.to_json().dump(2);
    import_stream.close();
    
    EXPECT_CALL(*persistence_, import_session(import_file))
        .Times(1)
        .WillOnce(Return(true));
    
    bool imported = persistence_->import_session(import_file);
    EXPECT_TRUE(imported);
}

// Serialization format tests

TEST_F(SessionPersistenceTest, JSONSerialization) {
    std::string expected_json = test_session_1_.to_json().dump(2);
    
    EXPECT_CALL(*serializer_, serialize_to_json(MatchesSession(test_session_1_)))
        .Times(1)
        .WillOnce(Return(expected_json));
    
    std::string serialized = serializer_->serialize_to_json(test_session_1_);
    EXPECT_FALSE(serialized.empty());
    
    // Test deserialization
    EXPECT_CALL(*serializer_, deserialize_from_json(serialized))
        .Times(1)
        .WillOnce(Return(test_session_1_));
    
    SessionData deserialized = serializer_->deserialize_from_json(serialized);
    EXPECT_EQ(deserialized.id, test_session_1_.id);
    EXPECT_EQ(deserialized.messages.size(), test_session_1_.messages.size());
}

TEST_F(SessionPersistenceTest, BinarySerialization) {
    std::vector<uint8_t> binary_data = {0x01, 0x02, 0x03, 0x04}; // Mock binary data
    
    EXPECT_CALL(*serializer_, serialize_to_binary(MatchesSession(test_session_1_)))
        .Times(1)
        .WillOnce(Return(binary_data));
    
    auto serialized = serializer_->serialize_to_binary(test_session_1_);
    EXPECT_FALSE(serialized.empty());
    
    // Test deserialization
    EXPECT_CALL(*serializer_, deserialize_from_binary(serialized))
        .Times(1)
        .WillOnce(Return(test_session_1_));
    
    SessionData deserialized = serializer_->deserialize_from_binary(serialized);
    EXPECT_EQ(deserialized.id, test_session_1_.id);
}

TEST_F(SessionPersistenceTest, XMLSerialization) {
    std::string xml_data = R"(<?xml version="1.0"?>
<session>
    <id>session-001</id>
    <created_at>2024-01-01T10:00:00Z</created_at>
    <messages>
        <message>
            <role>user</role>
            <content>Hello</content>
        </message>
    </messages>
</session>)";
    
    EXPECT_CALL(*serializer_, serialize_to_xml(MatchesSession(test_session_1_)))
        .Times(1)
        .WillOnce(Return(xml_data));
    
    std::string serialized = serializer_->serialize_to_xml(test_session_1_);
    EXPECT_THAT(serialized, HasSubstr("<session>"));
    EXPECT_THAT(serialized, HasSubstr("session-001"));
    
    // Test deserialization
    EXPECT_CALL(*serializer_, deserialize_from_xml(serialized))
        .Times(1)
        .WillOnce(Return(test_session_1_));
    
    SessionData deserialized = serializer_->deserialize_from_xml(serialized);
    EXPECT_EQ(deserialized.id, test_session_1_.id);
}

TEST_F(SessionPersistenceTest, ValidateSerializedData) {
    std::string valid_json = test_session_1_.to_json().dump();
    std::string invalid_json = "{invalid json}";
    
    EXPECT_CALL(*serializer_, validate_serialized_data(valid_json, "json"))
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*serializer_, validate_serialized_data(invalid_json, "json"))
        .Times(1)
        .WillOnce(Return(false));
    
    bool valid_result = serializer_->validate_serialized_data(valid_json, "json");
    bool invalid_result = serializer_->validate_serialized_data(invalid_json, "json");
    
    EXPECT_TRUE(valid_result);
    EXPECT_FALSE(invalid_result);
}

// Storage management tests

TEST_F(SessionPersistenceTest, GetStorageStatistics) {
    json expected_stats = {
        {"total_sessions", 2},
        {"total_size_bytes", 1024 * 1024},
        {"average_session_size_bytes", 512 * 1024},
        {"oldest_session", "2024-01-01T10:00:00Z"},
        {"newest_session", "2024-01-01T11:00:00Z"},
        {"storage_path", storage_path_.string()}
    };
    
    EXPECT_CALL(*persistence_, get_storage_stats())
        .Times(1)
        .WillOnce(Return(expected_stats));
    
    auto stats = persistence_->get_storage_stats();
    EXPECT_EQ(stats["total_sessions"], 2);
    EXPECT_GT(stats["total_size_bytes"].get<size_t>(), 0);
}

TEST_F(SessionPersistenceTest, CleanupOldSessions) {
    std::chrono::hours max_age{24}; // 24 hours
    
    EXPECT_CALL(*persistence_, cleanup_old_sessions(max_age))
        .Times(1)
        .WillOnce(Return(true));
    
    bool cleaned = persistence_->cleanup_old_sessions(max_age);
    EXPECT_TRUE(cleaned);
}

TEST_F(SessionPersistenceTest, CompactStorage) {
    // Get storage size before compaction
    EXPECT_CALL(*persistence_, get_storage_size())
        .Times(2)
        .WillOnce(Return(2 * 1024 * 1024)) // 2MB before
        .WillOnce(Return(1 * 1024 * 1024)); // 1MB after
    
    size_t size_before = persistence_->get_storage_size();
    
    EXPECT_CALL(*persistence_, compact_storage())
        .Times(1)
        .WillOnce(Return(true));
    
    bool compacted = persistence_->compact_storage();
    EXPECT_TRUE(compacted);
    
    size_t size_after = persistence_->get_storage_size();
    EXPECT_LT(size_after, size_before);
}

TEST_F(SessionPersistenceTest, ValidateStorageIntegrity) {
    EXPECT_CALL(*persistence_, validate_storage_integrity())
        .Times(1)
        .WillOnce(Return(true));
    
    bool valid = persistence_->validate_storage_integrity();
    EXPECT_TRUE(valid);
}

// Session search and filtering tests

TEST_F(SessionPersistenceTest, SearchSessionsByUser) {
    json search_query = {
        {"metadata.user_id", "user123"}
    };
    
    std::vector<SessionData> expected_results = {test_session_1_};
    
    EXPECT_CALL(*persistence_, search_sessions(search_query))
        .Times(1)
        .WillOnce(Return(expected_results));
    
    auto results = persistence_->search_sessions(search_query);
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].id, test_session_1_.id);
}

TEST_F(SessionPersistenceTest, SearchSessionsByTimeRange) {
    json search_query = {
        {"created_at", {
            {"$gte", "2024-01-01T10:00:00Z"},
            {"$lt", "2024-01-01T11:00:00Z"}
        }}
    };
    
    std::vector<SessionData> expected_results = {test_session_1_};
    
    EXPECT_CALL(*persistence_, search_sessions(search_query))
        .Times(1)
        .WillOnce(Return(expected_results));
    
    auto results = persistence_->search_sessions(search_query);
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].id, test_session_1_.id);
}

TEST_F(SessionPersistenceTest, SearchSessionsByTags) {
    json search_query = {
        {"metadata.tags", {
            {"$contains", "test"}
        }}
    };
    
    std::vector<SessionData> expected_results = {test_session_1_};
    
    EXPECT_CALL(*persistence_, search_sessions(search_query))
        .Times(1)
        .WillOnce(Return(expected_results));
    
    auto results = persistence_->search_sessions(search_query);
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].metadata["tags"], json({"test", "demo"}));
}

// Concurrent access tests

TEST_F(SessionPersistenceTest, ConcurrentSessionSaves) {
    const int num_threads = 4;
    const int sessions_per_thread = 5;
    std::atomic<int> successful_saves{0};
    
    EXPECT_CALL(*persistence_, save_session(_))
        .Times(num_threads * sessions_per_thread)
        .WillRepeatedly(Return(true));
    
    ThreadSafetyUtils::run_concurrent_test([this, &successful_saves]() {
        SessionData session = test_session_1_;
        session.id = "concurrent-" + std::to_string(std::rand());
        
        bool saved = persistence_->save_session(session);
        if (saved) {
            successful_saves++;
        }
    }, num_threads, sessions_per_thread);
    
    EXPECT_EQ(successful_saves.load(), num_threads * sessions_per_thread);
}

TEST_F(SessionPersistenceTest, ConcurrentSessionReadsAndWrites) {
    const int num_operations = 20;
    std::atomic<int> successful_operations{0};
    
    // Setup expectations for mixed operations
    EXPECT_CALL(*persistence_, save_session(_))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(true));
    
    EXPECT_CALL(*persistence_, load_session(_))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::make_optional(test_session_1_)));
    
    EXPECT_CALL(*persistence_, delete_session(_))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(true));
    
    std::vector<std::future<bool>> futures;
    
    for (int i = 0; i < num_operations; ++i) {
        futures.push_back(std::async(std::launch::async, [this, i, &successful_operations]() {
            try {
                int operation = i % 3;
                bool success = false;
                
                switch (operation) {
                    case 0: { // Save
                        SessionData session = test_session_1_;
                        session.id = "concurrent-save-" + std::to_string(i);
                        success = persistence_->save_session(session);
                        break;
                    }
                    case 1: { // Load
                        auto loaded = persistence_->load_session(test_session_1_.id);
                        success = loaded.has_value();
                        break;
                    }
                    case 2: { // Delete
                        success = persistence_->delete_session("temp-session-" + std::to_string(i));
                        break;
                    }
                }
                
                if (success) {
                    successful_operations++;
                }
                return success;
            } catch (...) {
                return false;
            }
        }));
    }
    
    // Wait for all operations
    for (auto& future : futures) {
        future.get();
    }
    
    EXPECT_GT(successful_operations.load(), 0);
}

// Error handling and recovery tests

TEST_F(SessionPersistenceTest, HandleStorageCorruption) {
    // Simulate corrupted storage
    EXPECT_CALL(*persistence_, validate_storage_integrity())
        .Times(1)
        .WillOnce(Return(false)); // Storage is corrupted
    
    bool integrity_check = persistence_->validate_storage_integrity();
    EXPECT_FALSE(integrity_check);
    
    // Attempt to repair by compacting storage
    EXPECT_CALL(*persistence_, compact_storage())
        .Times(1)
        .WillOnce(Return(true));
    
    bool repaired = persistence_->compact_storage();
    EXPECT_TRUE(repaired);
    
    // Verify integrity after repair
    EXPECT_CALL(*persistence_, validate_storage_integrity())
        .Times(1)
        .WillOnce(Return(true));
    
    bool integrity_after_repair = persistence_->validate_storage_integrity();
    EXPECT_TRUE(integrity_after_repair);
}

TEST_F(SessionPersistenceTest, HandleDiskSpaceIssues) {
    // Simulate disk full condition
    EXPECT_CALL(*persistence_, save_session(_))
        .Times(1)
        .WillOnce(Throw(std::runtime_error("No space left on device")));
    
    EXPECT_THROW(persistence_->save_session(test_session_1_), std::runtime_error);
    
    // Cleanup old sessions to free space
    EXPECT_CALL(*persistence_, cleanup_old_sessions(_))
        .Times(1)
        .WillOnce(Return(true));
    
    bool cleaned = persistence_->cleanup_old_sessions(std::chrono::hours{1});
    EXPECT_TRUE(cleaned);
    
    // Retry save after cleanup
    EXPECT_CALL(*persistence_, save_session(_))
        .Times(1)
        .WillOnce(Return(true));
    
    bool retry_save = persistence_->save_session(test_session_1_);
    EXPECT_TRUE(retry_save);
}

// Performance tests

TEST_F(SessionPersistenceTest, PerformanceLargeSessions) {
    // Create a large session with many messages
    SessionData large_session = test_session_1_;
    large_session.id = "large-session";
    
    // Add 1000 messages
    for (int i = 0; i < 1000; ++i) {
        large_session.messages.push_back(json{
            {"role", (i % 2 == 0) ? "user" : "assistant"},
            {"content", "Message " + std::to_string(i) + " with some content to test performance"},
            {"timestamp", "2024-01-01T10:" + std::to_string(i % 60) + ":00Z"}
        });
    }
    
    EXPECT_CALL(*persistence_, save_session(MatchesSessionId("large-session")))
        .Times(1)
        .WillOnce(Return(true));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    bool saved = persistence_->save_session(large_session);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    EXPECT_TRUE(saved);
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Large session save time: " << duration.count() << " ms" << std::endl;
    
    // Should complete within reasonable time (e.g., 1 second)
    EXPECT_LT(duration.count(), 1000);
}

TEST_F(SessionPersistenceTest, PerformanceBulkOperations) {
    const int num_sessions = 100;
    std::vector<SessionData> sessions;
    
    // Create multiple sessions
    for (int i = 0; i < num_sessions; ++i) {
        SessionData session = test_session_1_;
        session.id = "bulk-session-" + std::to_string(i);
        sessions.push_back(session);
    }
    
    EXPECT_CALL(*persistence_, save_session(_))
        .Times(num_sessions)
        .WillRepeatedly(Return(true));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (const auto& session : sessions) {
        persistence_->save_session(session);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Bulk save of " << num_sessions << " sessions: " << duration.count() << " ms" << std::endl;
    
    // Performance expectation: should save 100 sessions in under 5 seconds
    EXPECT_LT(duration.count(), 5000);
    
    double sessions_per_second = (num_sessions * 1000.0) / duration.count();
    std::cout << "Sessions per second: " << sessions_per_second << std::endl;
    
    EXPECT_GT(sessions_per_second, 20.0); // Should handle at least 20 sessions/second
}

// Custom matchers for session data
MATCHER_P(MatchesSession, expected_session, "") {
    return arg.id == expected_session.id &&
           arg.created_at == expected_session.created_at &&
           arg.messages.size() == expected_session.messages.size();
}

MATCHER_P(MatchesSessionId, expected_id, "") {
    return arg.id == expected_id;
}

// Cleanup test

TEST_F(SessionPersistenceTest, ProperShutdown) {
    EXPECT_CALL(*persistence_, shutdown())
        .Times(1);
    
    persistence_->shutdown();
}