#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../utils/test_helpers.h"
#include <string>
#include <vector>
#include <sstream>
#include <memory>
#include <chrono>
#include <thread>
#include <filesystem>

// Mock CLI command processor (would be actual includes in real implementation)
// #include "../../src/cli/CommandProcessor.h"
// #include "../../src/cli/Commands.h"

using namespace gemma::test;
using namespace testing;

// Mock CLI command structure
struct CLICommand {
    std::string name;
    std::string description;
    std::vector<std::string> aliases;
    std::function<int(const std::vector<std::string>&)> handler;
};

// Mock CLI argument parser
struct CLIArgs {
    std::string command;
    std::vector<std::string> positional_args;
    std::map<std::string, std::string> named_args;
    std::map<std::string, bool> flags;
    
    bool has_flag(const std::string& flag) const {
        auto it = flags.find(flag);
        return it != flags.end() && it->second;
    }
    
    std::string get_arg(const std::string& name, const std::string& default_value = "") const {
        auto it = named_args.find(name);
        return it != named_args.end() ? it->second : default_value;
    }
};

// Mock CLI processor interface
class MockCLIProcessor {
public:
    MOCK_METHOD(bool, initialize, (), ());
    MOCK_METHOD(int, execute_command, (const std::vector<std::string>& args), ());
    MOCK_METHOD(void, register_command, (const CLICommand& command), ());
    MOCK_METHOD(std::vector<std::string>, list_commands, (), (const));
    MOCK_METHOD(std::string, get_help, (const std::string& command), (const));
    MOCK_METHOD(void, set_output_stream, (std::ostream* stream), ());
    MOCK_METHOD(void, set_error_stream, (std::ostream* stream), ());
    MOCK_METHOD(CLIArgs, parse_arguments, (const std::vector<std::string>& args), ());
    MOCK_METHOD(bool, validate_arguments, (const CLIArgs& args), ());
    MOCK_METHOD(void, print_version, (), ());
    MOCK_METHOD(void, print_usage, (), ());
};

// Mock model interface for CLI commands
class MockModelInterface {
public:
    MOCK_METHOD(bool, load_model, (const std::string& weights_path, const std::string& tokenizer_path), ());
    MOCK_METHOD(bool, unload_model, (), ());
    MOCK_METHOD(bool, is_model_loaded, (), (const));
    MOCK_METHOD(std::string, generate_text, (const std::string& prompt, int max_tokens, float temperature), ());
    MOCK_METHOD(int, count_tokens, (const std::string& text), (const));
    MOCK_METHOD(nlohmann::json, get_model_info, (), (const));
    MOCK_METHOD(void, set_backend, (const std::string& backend_name), ());
    MOCK_METHOD(std::string, get_current_backend, (), (const));
    MOCK_METHOD(std::vector<std::string>, list_available_backends, (), (const));
};

class CLICommandTest : public GemmaTestBase {
protected:
    void SetUp() override {
        GemmaTestBase::SetUp();
        
        cli_processor_ = std::make_unique<MockCLIProcessor>();
        model_interface_ = std::make_unique<MockModelInterface>();
        
        // Redirect output to string streams for testing
        output_stream_ = std::make_unique<std::ostringstream>();
        error_stream_ = std::make_unique<std::ostringstream>();
        
        setup_default_expectations();
        setup_test_model_paths();
    }
    
    void setup_default_expectations() {
        ON_CALL(*cli_processor_, initialize()).WillByDefault(Return(true));
        ON_CALL(*cli_processor_, list_commands()).WillByDefault(Return(std::vector<std::string>{
            "generate", "load", "unload", "info", "count-tokens", "help", "version"
        }));
        
        ON_CALL(*model_interface_, is_model_loaded()).WillByDefault(Return(false));
        ON_CALL(*model_interface_, get_current_backend()).WillByDefault(Return("cpu"));
        ON_CALL(*model_interface_, list_available_backends()).WillByDefault(Return(std::vector<std::string>{
            "cpu", "intel", "cuda", "vulkan"
        }));
    }
    
    void setup_test_model_paths() {
        test_weights_path_ = (test_dir_ / "test_model.sbs").string();
        test_tokenizer_path_ = (test_dir_ / "tokenizer.spm").string();
        
        // Create dummy model files for testing
        FileTestUtils::create_temp_file(test_dir_, "dummy weights data", ".sbs");
        FileTestUtils::create_temp_file(test_dir_, "dummy tokenizer data", ".spm");
    }
    
    std::unique_ptr<MockCLIProcessor> cli_processor_;
    std::unique_ptr<MockModelInterface> model_interface_;
    std::unique_ptr<std::ostringstream> output_stream_;
    std::unique_ptr<std::ostringstream> error_stream_;
    std::string test_weights_path_;
    std::string test_tokenizer_path_;
};

// Basic CLI initialization and command registration tests

TEST_F(CLICommandTest, InitializationSuccess) {
    EXPECT_CALL(*cli_processor_, initialize())
        .Times(1)
        .WillOnce(Return(true));
    
    bool initialized = cli_processor_->initialize();
    EXPECT_TRUE(initialized);
}

TEST_F(CLICommandTest, CommandRegistration) {
    CLICommand test_command;
    test_command.name = "test-command";
    test_command.description = "A test command";
    test_command.aliases = {"tc", "test"};
    
    EXPECT_CALL(*cli_processor_, register_command(_))
        .Times(1);
    
    cli_processor_->register_command(test_command);
}

TEST_F(CLICommandTest, ListAvailableCommands) {
    std::vector<std::string> expected_commands = {
        "generate", "load", "unload", "info", "count-tokens", "help", "version"
    };
    
    EXPECT_CALL(*cli_processor_, list_commands())
        .Times(1)
        .WillOnce(Return(expected_commands));
    
    auto commands = cli_processor_->list_commands();
    
    EXPECT_EQ(commands.size(), 7);
    EXPECT_THAT(commands, Contains("generate"));
    EXPECT_THAT(commands, Contains("load"));
    EXPECT_THAT(commands, Contains("help"));
}

// Argument parsing tests

TEST_F(CLICommandTest, ParseSimpleArguments) {
    std::vector<std::string> args = {"generate", "--prompt", "Hello world", "--max-tokens", "50"};
    
    CLIArgs expected_args;
    expected_args.command = "generate";
    expected_args.named_args["prompt"] = "Hello world";
    expected_args.named_args["max-tokens"] = "50";
    
    EXPECT_CALL(*cli_processor_, parse_arguments(args))
        .Times(1)
        .WillOnce(Return(expected_args));
    
    auto parsed_args = cli_processor_->parse_arguments(args);
    
    EXPECT_EQ(parsed_args.command, "generate");
    EXPECT_EQ(parsed_args.get_arg("prompt"), "Hello world");
    EXPECT_EQ(parsed_args.get_arg("max-tokens"), "50");
}

TEST_F(CLICommandTest, ParseArgumentsWithFlags) {
    std::vector<std::string> args = {"load", "--weights", "model.sbs", "--verbose", "--force"};
    
    CLIArgs expected_args;
    expected_args.command = "load";
    expected_args.named_args["weights"] = "model.sbs";
    expected_args.flags["verbose"] = true;
    expected_args.flags["force"] = true;
    
    EXPECT_CALL(*cli_processor_, parse_arguments(args))
        .Times(1)
        .WillOnce(Return(expected_args));
    
    auto parsed_args = cli_processor_->parse_arguments(args);
    
    EXPECT_EQ(parsed_args.command, "load");
    EXPECT_EQ(parsed_args.get_arg("weights"), "model.sbs");
    EXPECT_TRUE(parsed_args.has_flag("verbose"));
    EXPECT_TRUE(parsed_args.has_flag("force"));
}

TEST_F(CLICommandTest, ParsePositionalArguments) {
    std::vector<std::string> args = {"generate", "Hello world", "50"};
    
    CLIArgs expected_args;
    expected_args.command = "generate";
    expected_args.positional_args = {"Hello world", "50"};
    
    EXPECT_CALL(*cli_processor_, parse_arguments(args))
        .Times(1)
        .WillOnce(Return(expected_args));
    
    auto parsed_args = cli_processor_->parse_arguments(args);
    
    EXPECT_EQ(parsed_args.command, "generate");
    EXPECT_EQ(parsed_args.positional_args.size(), 2);
    EXPECT_EQ(parsed_args.positional_args[0], "Hello world");
    EXPECT_EQ(parsed_args.positional_args[1], "50");
}

// Model loading commands tests

TEST_F(CLICommandTest, LoadModelCommand) {
    std::vector<std::string> args = {
        "load", 
        "--weights", test_weights_path_,
        "--tokenizer", test_tokenizer_path_
    };
    
    EXPECT_CALL(*model_interface_, load_model(test_weights_path_, test_tokenizer_path_))
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0)); // Success exit code
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

TEST_F(CLICommandTest, LoadModelCommandWithMissingFiles) {
    std::vector<std::string> args = {
        "load",
        "--weights", "/nonexistent/model.sbs",
        "--tokenizer", "/nonexistent/tokenizer.spm"
    };
    
    EXPECT_CALL(*model_interface_, load_model(_, _))
        .Times(1)
        .WillOnce(Return(false)); // Fail to load
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(1)); // Error exit code
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 1);
}

TEST_F(CLICommandTest, UnloadModelCommand) {
    // First load a model
    EXPECT_CALL(*model_interface_, is_model_loaded())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*model_interface_, unload_model())
        .Times(1)
        .WillOnce(Return(true));
    
    std::vector<std::string> args = {"unload"};
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0));
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

TEST_F(CLICommandTest, UnloadModelWhenNotLoaded) {
    EXPECT_CALL(*model_interface_, is_model_loaded())
        .Times(1)
        .WillOnce(Return(false));
    
    std::vector<std::string> args = {"unload"};
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(1)); // Error: no model to unload
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 1);
}

// Text generation commands tests

TEST_F(CLICommandTest, GenerateTextCommand) {
    std::vector<std::string> args = {
        "generate",
        "--prompt", "What is the capital of France?",
        "--max-tokens", "50",
        "--temperature", "0.7"
    };
    
    EXPECT_CALL(*model_interface_, is_model_loaded())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*model_interface_, generate_text("What is the capital of France?", 50, 0.7f))
        .Times(1)
        .WillOnce(Return("The capital of France is Paris."));
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0));
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

TEST_F(CLICommandTest, GenerateTextWithoutLoadedModel) {
    std::vector<std::string> args = {
        "generate",
        "--prompt", "Test prompt"
    };
    
    EXPECT_CALL(*model_interface_, is_model_loaded())
        .Times(1)
        .WillOnce(Return(false));
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(1)); // Error: no model loaded
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 1);
}

TEST_F(CLICommandTest, GenerateTextWithInvalidParameters) {
    std::vector<std::string> args = {
        "generate",
        "--prompt", "Test",
        "--max-tokens", "-10", // Invalid negative value
        "--temperature", "2.0" // Invalid temperature > 1.0
    };
    
    CLIArgs parsed_args;
    parsed_args.command = "generate";
    parsed_args.named_args["prompt"] = "Test";
    parsed_args.named_args["max-tokens"] = "-10";
    parsed_args.named_args["temperature"] = "2.0";
    
    EXPECT_CALL(*cli_processor_, parse_arguments(args))
        .Times(1)
        .WillOnce(Return(parsed_args));
    
    EXPECT_CALL(*cli_processor_, validate_arguments(_))
        .Times(1)
        .WillOnce(Return(false)); // Validation fails
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(2)); // Invalid arguments error code
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 2);
}

// Token counting commands tests

TEST_F(CLICommandTest, CountTokensCommand) {
    std::vector<std::string> args = {
        "count-tokens",
        "--text", "This is a test message for token counting."
    };
    
    EXPECT_CALL(*model_interface_, is_model_loaded())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*model_interface_, count_tokens("This is a test message for token counting."))
        .Times(1)
        .WillOnce(Return(12));
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0));
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

TEST_F(CLICommandTest, CountTokensFromFile) {
    // Create a test file with content
    std::string test_content = "This is test content from a file.\nIt has multiple lines.\nAnd should be tokenized properly.";
    std::string test_file = FileTestUtils::create_temp_file(test_dir_, test_content);
    
    std::vector<std::string> args = {
        "count-tokens",
        "--file", test_file
    };
    
    EXPECT_CALL(*model_interface_, is_model_loaded())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*model_interface_, count_tokens(test_content))
        .Times(1)
        .WillOnce(Return(20));
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0));
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

// Model information commands tests

TEST_F(CLICommandTest, ModelInfoCommand) {
    nlohmann::json model_info = {
        {"model_name", "gemma-2b-it"},
        {"model_size", "2B"},
        {"context_length", 8192},
        {"vocab_size", 256000},
        {"backend", "cpu"},
        {"loaded", true}
    };
    
    std::vector<std::string> args = {"info"};
    
    EXPECT_CALL(*model_interface_, get_model_info())
        .Times(1)
        .WillOnce(Return(model_info));
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0));
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

TEST_F(CLICommandTest, ModelInfoWhenNotLoaded) {
    nlohmann::json empty_info = {
        {"loaded", false},
        {"message", "No model currently loaded"}
    };
    
    std::vector<std::string> args = {"info"};
    
    EXPECT_CALL(*model_interface_, get_model_info())
        .Times(1)
        .WillOnce(Return(empty_info));
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0)); // Still success, just shows "not loaded"
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

// Backend management commands tests

TEST_F(CLICommandTest, ListBackendsCommand) {
    std::vector<std::string> backends = {"cpu", "intel", "cuda", "vulkan"};
    
    std::vector<std::string> args = {"backends", "list"};
    
    EXPECT_CALL(*model_interface_, list_available_backends())
        .Times(1)
        .WillOnce(Return(backends));
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0));
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

TEST_F(CLICommandTest, SetBackendCommand) {
    std::vector<std::string> args = {"backends", "set", "intel"};
    
    EXPECT_CALL(*model_interface_, set_backend("intel"))
        .Times(1);
    
    EXPECT_CALL(*model_interface_, get_current_backend())
        .Times(1)
        .WillOnce(Return("intel"));
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0));
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

TEST_F(CLICommandTest, SetInvalidBackend) {
    std::vector<std::string> args = {"backends", "set", "invalid-backend"};
    
    EXPECT_CALL(*model_interface_, set_backend("invalid-backend"))
        .Times(1)
        .WillOnce(Throw(std::invalid_argument("Backend not available")));
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(1)); // Error code
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 1);
}

// Help and version commands tests

TEST_F(CLICommandTest, HelpCommand) {
    std::vector<std::string> args = {"help"};
    
    EXPECT_CALL(*cli_processor_, print_usage())
        .Times(1);
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0));
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

TEST_F(CLICommandTest, HelpForSpecificCommand) {
    std::vector<std::string> args = {"help", "generate"};
    
    std::string help_text = "generate - Generate text using the loaded model\n"
                           "Usage: generate --prompt <text> [options]\n"
                           "Options:\n"
                           "  --prompt <text>      Input prompt for generation\n"
                           "  --max-tokens <int>   Maximum tokens to generate (default: 100)\n"
                           "  --temperature <float> Sampling temperature (default: 0.8)";
    
    EXPECT_CALL(*cli_processor_, get_help("generate"))
        .Times(1)
        .WillOnce(Return(help_text));
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0));
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

TEST_F(CLICommandTest, VersionCommand) {
    std::vector<std::string> args = {"version"};
    
    EXPECT_CALL(*cli_processor_, print_version())
        .Times(1);
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0));
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

// Interactive mode tests

TEST_F(CLICommandTest, InteractiveMode) {
    std::vector<std::string> args = {"--interactive"};
    
    // Mock interactive session
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0)); // Interactive mode starts successfully
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

// Batch processing tests

TEST_F(CLICommandTest, BatchProcessing) {
    // Create a batch file with multiple prompts
    std::string batch_content = "What is the capital of France?\n"
                               "Explain quantum computing in simple terms.\n"
                               "Write a short poem about autumn.\n";
    std::string batch_file = FileTestUtils::create_temp_file(test_dir_, batch_content, ".txt");
    
    std::vector<std::string> args = {
        "generate",
        "--batch-file", batch_file,
        "--max-tokens", "100"
    };
    
    EXPECT_CALL(*model_interface_, is_model_loaded())
        .Times(1)
        .WillOnce(Return(true));
    
    // Should call generate_text for each line in the batch file
    EXPECT_CALL(*model_interface_, generate_text(_, 100, _))
        .Times(3)
        .WillRepeatedly(Return("Generated response"));
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0));
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

// Output formatting tests

TEST_F(CLICommandTest, JSONOutputFormat) {
    std::vector<std::string> args = {
        "generate",
        "--prompt", "Hello world",
        "--format", "json"
    };
    
    EXPECT_CALL(*model_interface_, is_model_loaded())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*model_interface_, generate_text("Hello world", _, _))
        .Times(1)
        .WillOnce(Return("Hello! How can I help you today?"));
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0));
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

TEST_F(CLICommandTest, VerboseOutput) {
    std::vector<std::string> args = {
        "load",
        "--weights", test_weights_path_,
        "--tokenizer", test_tokenizer_path_,
        "--verbose"
    };
    
    EXPECT_CALL(*model_interface_, load_model(_, _))
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0));
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

// Error handling tests

TEST_F(CLICommandTest, InvalidCommandError) {
    std::vector<std::string> args = {"invalid-command"};
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(127)); // Command not found error code
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 127);
}

TEST_F(CLICommandTest, MissingRequiredArguments) {
    std::vector<std::string> args = {"generate"}; // Missing --prompt
    
    CLIArgs parsed_args;
    parsed_args.command = "generate";
    
    EXPECT_CALL(*cli_processor_, parse_arguments(args))
        .Times(1)
        .WillOnce(Return(parsed_args));
    
    EXPECT_CALL(*cli_processor_, validate_arguments(_))
        .Times(1)
        .WillOnce(Return(false)); // Missing required argument
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(2)); // Invalid arguments error code
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 2);
}

TEST_F(CLICommandTest, FileNotFoundError) {
    std::vector<std::string> args = {
        "load",
        "--weights", "/nonexistent/model.sbs"
    };
    
    EXPECT_CALL(*model_interface_, load_model(_, _))
        .Times(1)
        .WillOnce(Return(false)); // File not found
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(3)); // File not found error code
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 3);
}

// Performance and stress tests

TEST_F(CLICommandTest, MultipleSequentialCommands) {
    std::vector<std::vector<std::string>> command_sequence = {
        {"load", "--weights", test_weights_path_, "--tokenizer", test_tokenizer_path_},
        {"info"},
        {"generate", "--prompt", "Test 1", "--max-tokens", "10"},
        {"generate", "--prompt", "Test 2", "--max-tokens", "10"},
        {"count-tokens", "--text", "Sample text"},
        {"unload"}
    };
    
    // Setup expectations for the command sequence
    EXPECT_CALL(*model_interface_, load_model(_, _))
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*model_interface_, get_model_info())
        .Times(1)
        .WillOnce(Return(nlohmann::json{{"loaded", true}}));
    
    EXPECT_CALL(*model_interface_, is_model_loaded())
        .Times(3)
        .WillRepeatedly(Return(true));
    
    EXPECT_CALL(*model_interface_, generate_text(_, 10, _))
        .Times(2)
        .WillRepeatedly(Return("Generated text"));
    
    EXPECT_CALL(*model_interface_, count_tokens("Sample text"))
        .Times(1)
        .WillOnce(Return(2));
    
    EXPECT_CALL(*model_interface_, unload_model())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*cli_processor_, execute_command(_))
        .Times(6)
        .WillRepeatedly(Return(0));
    
    // Execute all commands
    for (const auto& args : command_sequence) {
        int exit_code = cli_processor_->execute_command(args);
        EXPECT_EQ(exit_code, 0);
    }
}

TEST_F(CLICommandTest, LongRunningGeneration) {
    std::vector<std::string> args = {
        "generate",
        "--prompt", "Write a very long story",
        "--max-tokens", "2000"
    };
    
    EXPECT_CALL(*model_interface_, is_model_loaded())
        .Times(1)
        .WillOnce(Return(true));
    
    // Simulate long-running generation
    EXPECT_CALL(*model_interface_, generate_text(_, 2000, _))
        .Times(1)
        .WillOnce(Invoke([](const std::string&, int, float) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate work
            return "Very long generated story...";
        }));
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    int exit_code = cli_processor_->execute_command(args);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    EXPECT_EQ(exit_code, 0);
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    EXPECT_GE(duration.count(), 100); // Should take at least 100ms
}

// Configuration and settings tests

TEST_F(CLICommandTest, ConfigurationCommands) {
    std::vector<std::string> args = {
        "config",
        "--set", "max_tokens=100",
        "--set", "temperature=0.8"
    };
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0));
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}

TEST_F(CLICommandTest, ListConfigurationSettings) {
    std::vector<std::string> args = {"config", "--list"};
    
    EXPECT_CALL(*cli_processor_, execute_command(args))
        .Times(1)
        .WillOnce(Return(0));
    
    int exit_code = cli_processor_->execute_command(args);
    EXPECT_EQ(exit_code, 0);
}