#!/usr/bin/env python3
"""
Basic test for the document processor module to verify functionality
"""

import json
import os
from pathlib import Path
import sys
import tempfile


def test_basic_functionality():
    """Test basic document processing functionality"""
    print("Testing document processor basic functionality...")

    try:
        # Try to import the module
        import gemma_extensions as ge

        print("✓ Successfully imported gemma_extensions")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        print("Build the extensions first with: uv run maturin develop --release")
        return False

    try:
        # Test format detection
        print("Testing format detection...")
        assert ge.detect_format("test.pdf").value == "Pdf"
        assert ge.detect_format("test.csv").value == "Csv"
        assert ge.detect_format("test.json").value == "Json"
        print("✓ Format detection works")

        # Test configuration
        print("Testing configuration...")
        config = ge.DocumentConfig.default()
        assert config.max_file_size > 0
        assert config.chunk_size > 0
        print("✓ Configuration works")

        # Test document processor creation
        print("Testing processor creation...")
        processor = ge.DocumentProcessor(config)
        print("✓ Processor creation works")

        # Test text processing
        print("Testing text processing...")
        text_content = "Hello, world! This is a test document with some words."
        result = processor.process_bytes(text_content.encode("utf-8"), ge.DocumentFormat("text"))

        assert result.content == text_content
        assert result.metadata.word_count == 11
        assert result.metadata.char_count == len(text_content)
        print("✓ Text processing works")

        # Test CSV processing
        print("Testing CSV processing...")
        csv_content = "name,age,city\nJohn,30,NYC\nJane,25,LA\n"
        csv_result = processor.process_bytes(csv_content.encode("utf-8"), ge.DocumentFormat("csv"))

        assert csv_result.metadata.row_count == 2
        assert csv_result.metadata.column_count == 3
        assert csv_result.metadata.column_names == ["name", "age", "city"]
        print("✓ CSV processing works")

        # Test JSON processing
        print("Testing JSON processing...")
        json_data = {"name": "John", "age": 30, "city": "NYC"}
        json_content = json.dumps(json_data)
        json_result = processor.process_bytes(
            json_content.encode("utf-8"), ge.DocumentFormat("json")
        )

        assert "John" in json_result.content
        assert "NYC" in json_result.content
        assert json_result.metadata.json_object_count == 1
        print("✓ JSON processing works")

        # Test file processing
        print("Testing file processing...")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test file.\nIt has multiple lines.\nAnd some content.")
            temp_path = f.name

        try:
            file_result = processor.process_file(temp_path)
            assert file_result.metadata.file_size > 0
            assert "test file" in file_result.content
            print("✓ File processing works")
        finally:
            os.unlink(temp_path)

        # Test batch processing
        print("Testing batch processing...")
        files = []
        try:
            for i in range(3):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                    f.write(f"Test file {i} content.")
                    files.append(f.name)

            batch_results = processor.process_batch(files)
            assert len(batch_results) == 3
            for result in batch_results:
                assert result.metadata.word_count > 0
            print("✓ Batch processing works")

        finally:
            for file_path in files:
                if os.path.exists(file_path):
                    os.unlink(file_path)

        print("\n🎉 All tests passed! Document processor is working correctly.")
        return True

    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling"""
    print("\nTesting error handling...")

    try:
        import gemma_extensions as ge

        processor = ge.DocumentProcessor()

        # Test file not found
        try:
            processor.process_file("nonexistent_file.txt")
            print("✗ Should have raised an error for missing file")
            return False
        except Exception:
            print("✓ Correctly handles missing file")

        # Test invalid format
        try:
            processor.process_bytes(b"test", ge.DocumentFormat("unknown"))
            print("✗ Should have raised an error for unknown format")
            return False
        except Exception:
            print("✓ Correctly handles unknown format")

        return True

    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Document Processor Test Suite")
    print("=" * 40)

    success = True
    success &= test_basic_functionality()
    success &= test_error_handling()

    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
