import sys
import os
import pytest

def main():
    """Run the tests."""
    # Add the src directory to the python path
    src_path = os.path.abspath("src")
    sys.path.insert(0, src_path)

    # Run pytest
    sys.exit(pytest.main())

if __name__ == "__main__":
    main()
