#!/usr/bin/env python3
"""
Direct test of Redis connectivity on port 6380
"""

import redis
import json
from datetime import datetime

def test_redis_direct():
    """Test direct Redis connectivity"""
    print("=" * 60)
    print("Direct Redis Connectivity Test")
    print("=" * 60)

    try:
        # Connect to Redis on port 6380
        r = redis.Redis(host='127.0.0.1', port=6380, decode_responses=True)

        # Test PING
        pong = r.ping()
        print(f"\n1. PING test: {'SUCCESS' if pong else 'FAILED'}")

        # Write test data
        test_key = f"test:connectivity:{int(datetime.now().timestamp())}"
        test_value = json.dumps({
            "test": "MCP Redis Integration",
            "timestamp": datetime.now().isoformat(),
            "port": 6380
        })

        r.set(test_key, test_value)
        print(f"2. WRITE test: SUCCESS (key: {test_key})")

        # Read test data
        retrieved = r.get(test_key)
        if retrieved:
            data = json.loads(retrieved)
            print(f"3. READ test: SUCCESS")
            print(f"   Data: {data}")
        else:
            print(f"3. READ test: FAILED")

        # List keys
        keys = r.keys("test:*")
        print(f"\n4. Keys matching 'test:*': {len(keys)} found")
        for key in keys[:5]:  # Show first 5
            print(f"   - {key}")

        # Check memory keys
        mem_keys = r.keys("mem:*")
        print(f"\n5. Memory keys (mem:*): {len(mem_keys)} found")

        # Check document keys
        doc_keys = r.keys("doc:*")
        print(f"6. Document keys (doc:*): {len(doc_keys)} found")

        # Cleanup
        r.delete(test_key)
        print(f"\n7. CLEANUP: Deleted test key")

        print("\n" + "=" * 60)
        print("Redis is FULLY OPERATIONAL on port 6380")
        print("=" * 60)

        return True

    except redis.ConnectionError as e:
        print(f"\nERROR: Cannot connect to Redis on port 6380")
        print(f"Details: {e}")
        return False
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = test_redis_direct()
    sys.exit(0 if success else 1)