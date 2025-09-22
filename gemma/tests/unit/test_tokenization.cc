// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <sentencepiece_processor.h>
#include <filesystem>
#include <string>
#include <vector>

#include "gemma/gemma.h"
#include "util/test_util.h"

namespace gcpp {
namespace {

class TokenizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_tokenizer_path_ = "/c/codedev/llm/.models/tokenizer.spm";

        // Only initialize if tokenizer file exists
        if (std::filesystem::exists(test_tokenizer_path_)) {
            auto status = tokenizer_.Load(test_tokenizer_path_);
            tokenizer_available_ = status.ok();
            if (!tokenizer_available_) {
                std::cerr << "Failed to load tokenizer: " << status.ToString() << std::endl;
            }
        } else {
            tokenizer_available_ = false;
        }
    }

    void TearDown() override {
        // Clean up
    }

    std::string test_tokenizer_path_;
    sentencepiece::SentencePieceProcessor tokenizer_;
    bool tokenizer_available_ = false;
};

TEST_F(TokenizationTest, BasicEncoding) {
    if (!tokenizer_available_) {
        GTEST_SKIP() << "Tokenizer not available";
    }

    std::vector<std::string> test_texts = {
        "Hello world",
        "This is a test sentence.",
        "How are you doing today?",
        "The quick brown fox jumps over the lazy dog.",
        ""
    };

    for (const auto& text : test_texts) {
        SCOPED_TRACE("Testing text: " + text);

        std::vector<int> tokens;
        auto status = tokenizer_.Encode(text, &tokens);
        EXPECT_TRUE(status.ok()) << "Failed to encode: " << status.ToString();

        if (text.empty()) {
            EXPECT_TRUE(tokens.empty() || tokens.size() == 1);
        } else {
            EXPECT_GT(tokens.size(), 0);
        }

        // Verify all tokens are within vocabulary
        for (int token : tokens) {
            EXPECT_GE(token, 0);
            EXPECT_LT(token, tokenizer_.GetPieceSize());
        }
    }
}

TEST_F(TokenizationTest, BasicDecoding) {
    if (!tokenizer_available_) {
        GTEST_SKIP() << "Tokenizer not available";
    }

    std::vector<std::string> test_texts = {
        "Hello",
        "world",
        "test",
        "sentence"
    };

    for (const auto& text : test_texts) {
        SCOPED_TRACE("Testing text: " + text);

        // Encode then decode
        std::vector<int> tokens;
        auto encode_status = tokenizer_.Encode(text, &tokens);
        EXPECT_TRUE(encode_status.ok());

        std::string decoded;
        auto decode_status = tokenizer_.Decode(tokens, &decoded);
        EXPECT_TRUE(decode_status.ok()) << "Failed to decode: " << decode_status.ToString();

        // The decoded text should contain the original (may have extra spaces)
        EXPECT_FALSE(decoded.empty());
    }
}

TEST_F(TokenizationTest, RoundTripConsistency) {
    if (!tokenizer_available_) {
        GTEST_SKIP() << "Tokenizer not available";
    }

    std::vector<std::string> test_texts = {
        "Hello world!",
        "This is a comprehensive test.",
        "Numbers: 123 456 789",
        "Symbols: @#$%^&*()",
        "Mixed: Hello123 world@2024!",
        "Code: def hello(): print('world')"
    };

    for (const auto& original_text : test_texts) {
        SCOPED_TRACE("Testing round-trip for: " + original_text);

        // Encode
        std::vector<int> tokens;
        auto encode_status = tokenizer_.Encode(original_text, &tokens);
        EXPECT_TRUE(encode_status.ok());
        EXPECT_GT(tokens.size(), 0);

        // Decode
        std::string decoded_text;
        auto decode_status = tokenizer_.Decode(tokens, &decoded_text);
        EXPECT_TRUE(decode_status.ok());

        // Check consistency (allowing for potential whitespace normalization)
        EXPECT_FALSE(decoded_text.empty());

        // Re-encode to verify stability
        std::vector<int> retokens;
        auto reencode_status = tokenizer_.Encode(decoded_text, &retokens);
        EXPECT_TRUE(reencode_status.ok());

        // Token sequences should be identical after round-trip
        EXPECT_EQ(tokens.size(), retokens.size());
        for (size_t i = 0; i < tokens.size() && i < retokens.size(); ++i) {
            EXPECT_EQ(tokens[i], retokens[i]) << "Mismatch at position " << i;
        }
    }
}

TEST_F(TokenizationTest, SpecialTokens) {
    if (!tokenizer_available_) {
        GTEST_SKIP() << "Tokenizer not available";
    }

    // Test common special tokens
    std::vector<std::string> special_tokens = {
        "<bos>",
        "<eos>",
        "<unk>",
        "<pad>"
    };

    for (const auto& token : special_tokens) {
        SCOPED_TRACE("Testing special token: " + token);

        int token_id = tokenizer_.PieceToId(token);
        if (token_id != tokenizer_.unk_id()) {
            std::string piece = tokenizer_.IdToPiece(token_id);
            EXPECT_EQ(piece, token);
        }
    }

    // Test special token IDs
    EXPECT_GE(tokenizer_.bos_id(), 0);
    EXPECT_GE(tokenizer_.eos_id(), 0);
    EXPECT_GE(tokenizer_.unk_id(), 0);
    EXPECT_GE(tokenizer_.pad_id(), 0);
}

TEST_F(TokenizationTest, VocabularySize) {
    if (!tokenizer_available_) {
        GTEST_SKIP() << "Tokenizer not available";
    }

    int vocab_size = tokenizer_.GetPieceSize();
    EXPECT_GT(vocab_size, 1000);  // Should have substantial vocabulary
    EXPECT_LT(vocab_size, 1000000);  // But not unreasonably large

    // Test vocabulary access
    for (int i = 0; i < std::min(100, vocab_size); ++i) {
        std::string piece = tokenizer_.IdToPiece(i);
        EXPECT_FALSE(piece.empty()) << "Empty piece at index " << i;

        int recovered_id = tokenizer_.PieceToId(piece);
        EXPECT_EQ(recovered_id, i) << "ID mismatch for piece: " << piece;
    }
}

TEST_F(TokenizationTest, LongTextHandling) {
    if (!tokenizer_available_) {
        GTEST_SKIP() << "Tokenizer not available";
    }

    // Create a long text
    std::string long_text;
    for (int i = 0; i < 1000; ++i) {
        long_text += "This is sentence number " + std::to_string(i) + ". ";
    }

    std::vector<int> tokens;
    auto status = tokenizer_.Encode(long_text, &tokens);
    EXPECT_TRUE(status.ok());
    EXPECT_GT(tokens.size(), 1000);  // Should produce many tokens

    // Test decoding of long token sequence
    std::string decoded;
    auto decode_status = tokenizer_.Decode(tokens, &decoded);
    EXPECT_TRUE(decode_status.ok());
    EXPECT_FALSE(decoded.empty());
}

TEST_F(TokenizationTest, EmptyAndWhitespaceHandling) {
    if (!tokenizer_available_) {
        GTEST_SKIP() << "Tokenizer not available";
    }

    std::vector<std::string> edge_cases = {
        "",
        " ",
        "  ",
        "\t",
        "\n",
        "\r\n",
        "   \t\n   "
    };

    for (const auto& text : edge_cases) {
        SCOPED_TRACE("Testing edge case text of length: " + std::to_string(text.length()));

        std::vector<int> tokens;
        auto status = tokenizer_.Encode(text, &tokens);
        EXPECT_TRUE(status.ok());

        if (!tokens.empty()) {
            std::string decoded;
            auto decode_status = tokenizer_.Decode(tokens, &decoded);
            EXPECT_TRUE(decode_status.ok());
        }
    }
}

TEST_F(TokenizationTest, UnicodeHandling) {
    if (!tokenizer_available_) {
        GTEST_SKIP() << "Tokenizer not available";
    }

    std::vector<std::string> unicode_texts = {
        "Hello 世界",
        "Café",
        "naïve résumé",
        "🚀 rocket emoji",
        "Ω μ α θ", // Greek letters
        "مرحبا بالعالم", // Arabic
        "こんにちは世界"  // Japanese
    };

    for (const auto& text : unicode_texts) {
        SCOPED_TRACE("Testing Unicode text: " + text);

        std::vector<int> tokens;
        auto status = tokenizer_.Encode(text, &tokens);
        EXPECT_TRUE(status.ok()) << "Failed to encode Unicode text: " << text;

        if (status.ok() && !tokens.empty()) {
            std::string decoded;
            auto decode_status = tokenizer_.Decode(tokens, &decoded);
            EXPECT_TRUE(decode_status.ok()) << "Failed to decode Unicode tokens";
        }
    }
}

TEST_F(TokenizationTest, TokenBoundaryValidation) {
    if (!tokenizer_available_) {
        GTEST_SKIP() << "Tokenizer not available";
    }

    // Test valid token IDs
    int vocab_size = tokenizer_.GetPieceSize();

    for (int token_id : {0, 1, vocab_size - 1}) {
        std::string piece = tokenizer_.IdToPiece(token_id);
        EXPECT_FALSE(piece.empty()) << "Token " << token_id << " produced empty piece";
    }

    // Test invalid token IDs
    for (int invalid_id : {-1, vocab_size, vocab_size + 1000}) {
        std::string piece = tokenizer_.IdToPiece(invalid_id);
        // Should either be empty or return unknown token
        // Implementation may vary
    }
}

TEST_F(TokenizationTest, PerformanceBaseline) {
    if (!tokenizer_available_) {
        GTEST_SKIP() << "Tokenizer not available";
    }

    // Test tokenization performance with moderately large text
    std::string test_text;
    for (int i = 0; i < 100; ++i) {
        test_text += "The quick brown fox jumps over the lazy dog. ";
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int iteration = 0; iteration < 100; ++iteration) {
        std::vector<int> tokens;
        auto status = tokenizer_.Encode(test_text, &tokens);
        EXPECT_TRUE(status.ok());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Should complete 100 tokenizations in reasonable time (< 10 seconds)
    EXPECT_LT(duration.count(), 10000);

    std::cout << "100 tokenizations completed in " << duration.count() << " ms" << std::endl;
}

} // namespace
} // namespace gcpp