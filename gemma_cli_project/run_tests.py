import sys
import os
import pytest

def main():
    """Run the tests."""
    os.environ["PYTHONPATH"] = os.path.abspath("src")
    sys.exit(pytest.main())

if __name__ == "__main__":
    main()

