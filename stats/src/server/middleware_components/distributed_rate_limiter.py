"""Distributed rate limiting using Redis for production scalability.

This module provides a Redis-backed rate limiter that works across multiple
application instances, preventing DoS attacks in distributed deployments.
"""

import asyncio
import json
import time
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import redis.asyncio as redis
from pydantic import BaseModel
from pydantic import Field
from redis.exceptions import RedisError


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""

    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""

    max_requests: int = Field(60, description="Maximum requests allowed")
    window_seconds: int = Field(60, description="Time window in seconds")
    strategy: RateLimitStrategy = Field(RateLimitStrategy.SLIDING_WINDOW)
    burst_size: int = Field(10, description="Burst size for token bucket")
    block_duration: int = Field(300, description="Block duration in seconds after limit exceeded")

    # Different limits for different endpoints
    endpoint_limits: dict[str, dict[str, int]] = Field(
        default_factory=lambda: {
            "/v1/chat/completions": {"max_requests": 30, "window_seconds": 60},
            "/v1/embeddings": {"max_requests": 100, "window_seconds": 60},
            "/health": {"max_requests": 1000, "window_seconds": 60},
        }
    )


class RateLimitResult(BaseModel):
    """Result of rate limit check."""

    allowed: bool
    limit: int
    remaining: int
    reset_at: int
    retry_after: int | None = None
    reason: str | None = None


class DistributedRateLimiter:
    """Redis-backed distributed rate limiter with multiple strategies."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        config: RateLimitConfig | None = None,
        key_prefix: str = "ratelimit",
    ):
        """Initialize the distributed rate limiter.

        Args:
            redis_url: Redis connection URL
            config: Rate limiting configuration
            key_prefix: Prefix for Redis keys
        """
        self.redis_url = redis_url
        self.config = config or RateLimitConfig()
        self.key_prefix = key_prefix
        self.redis_client: redis.Redis | None = None
        self._local_cache: dict[str, tuple[int, float]] = {}
        self._cache_ttl = 1.0  # Local cache TTL in seconds

    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = await redis.from_url(
                self.redis_url, encoding="utf-8", decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Connected to Redis for distributed rate limiting")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()

    def _get_redis_key(self, identifier: str, endpoint: str | None = None) -> str:
        """Generate Redis key for rate limiting.

        Args:
            identifier: Client identifier (IP, user ID, API key)
            endpoint: Optional endpoint for endpoint-specific limits

        Returns:
            Redis key string
        """
        if endpoint:
            return f"{self.key_prefix}:{endpoint}:{identifier}"
        return f"{self.key_prefix}:global:{identifier}"

    def _get_block_key(self, identifier: str) -> str:
        """Generate Redis key for blocked clients."""
        return f"{self.key_prefix}:blocked:{identifier}"

    async def _check_blocked(self, identifier: str) -> tuple[bool, int | None]:
        """Check if client is blocked.

        Args:
            identifier: Client identifier

        Returns:
            Tuple of (is_blocked, seconds_until_unblock)
        """
        block_key = self._get_block_key(identifier)
        ttl = await self.redis_client.ttl(block_key)

        if ttl > 0:
            return True, ttl
        return False, None

    async def _block_client(self, identifier: str):
        """Block a client for configured duration.

        Args:
            identifier: Client identifier
        """
        block_key = self._get_block_key(identifier)
        await self.redis_client.setex(
            block_key,
            self.config.block_duration,
            json.dumps({"blocked_at": time.time(), "reason": "rate_limit_exceeded"}),
        )
        logger.warning(f"Blocked client {identifier} for {self.config.block_duration} seconds")

    async def check_rate_limit(
        self, identifier: str, endpoint: str | None = None, cost: int = 1
    ) -> RateLimitResult:
        """Check if request is within rate limits.

        Args:
            identifier: Client identifier (IP, user ID, API key)
            endpoint: Optional endpoint for specific limits
            cost: Request cost (for weighted rate limiting)

        Returns:
            RateLimitResult with details
        """
        # Check if client is blocked
        is_blocked, retry_after = await self._check_blocked(identifier)
        if is_blocked:
            return RateLimitResult(
                allowed=False,
                limit=0,
                remaining=0,
                reset_at=int(time.time()) + retry_after,
                retry_after=retry_after,
                reason="Client temporarily blocked due to rate limit violations",
            )

        # Get appropriate limits
        if endpoint and endpoint in self.config.endpoint_limits:
            limits = self.config.endpoint_limits[endpoint]
            max_requests = limits["max_requests"]
            window_seconds = limits["window_seconds"]
        else:
            max_requests = self.config.max_requests
            window_seconds = self.config.window_seconds

        # Apply rate limiting based on strategy
        if self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            result = await self._sliding_window_check(
                identifier, endpoint, max_requests, window_seconds, cost
            )
        elif self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            result = await self._token_bucket_check(
                identifier, endpoint, max_requests, window_seconds, cost
            )
        else:
            result = await self._fixed_window_check(
                identifier, endpoint, max_requests, window_seconds, cost
            )

        # Block client if limit exceeded multiple times
        if not result.allowed:
            await self._track_violation(identifier)

        return result

    async def _sliding_window_check(
        self,
        identifier: str,
        endpoint: str | None,
        max_requests: int,
        window_seconds: int,
        cost: int,
    ) -> RateLimitResult:
        """Sliding window rate limiting implementation.

        Args:
            identifier: Client identifier
            endpoint: Optional endpoint
            max_requests: Maximum allowed requests
            window_seconds: Window size in seconds
            cost: Request cost

        Returns:
            RateLimitResult
        """
        key = self._get_redis_key(identifier, endpoint)
        now = time.time()
        window_start = now - window_seconds

        # Use Lua script for atomic operations
        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local max_requests = tonumber(ARGV[3])
        local cost = tonumber(ARGV[4])
        local window_seconds = tonumber(ARGV[5])

        -- Remove old entries
        redis.call('ZREMRANGEBYSCORE', key, 0, window_start)

        -- Count current requests
        local current_count = redis.call('ZCARD', key)

        -- Check if within limits
        if current_count + cost <= max_requests then
            -- Add request with microsecond precision to handle concurrent requests
            for i = 1, cost do
                redis.call('ZADD', key, now + i * 0.0001, now .. ':' .. i)
            end
            redis.call('EXPIRE', key, window_seconds * 2)
            return {1, max_requests - current_count - cost}
        else
            return {0, 0}
        end
        """

        try:
            result = await self.redis_client.eval(
                lua_script, 1, key, now, window_start, max_requests, cost, window_seconds
            )

            allowed = bool(result[0])
            remaining = result[1] if allowed else 0

            return RateLimitResult(
                allowed=allowed,
                limit=max_requests,
                remaining=remaining,
                reset_at=int(now + window_seconds),
                retry_after=window_seconds if not allowed else None,
                reason=(
                    None
                    if allowed
                    else f"Rate limit exceeded: {max_requests} requests per {window_seconds}s"
                ),
            )

        except RedisError as e:
            logger.error(f"Redis error in sliding window check: {e}")
            # Fallback to allow request on Redis error (fail open)
            return RateLimitResult(
                allowed=True,
                limit=max_requests,
                remaining=max_requests,
                reset_at=int(now + window_seconds),
            )

    async def _token_bucket_check(
        self,
        identifier: str,
        endpoint: str | None,
        max_requests: int,
        window_seconds: int,
        cost: int,
    ) -> RateLimitResult:
        """Token bucket rate limiting implementation.

        Args:
            identifier: Client identifier
            endpoint: Optional endpoint
            max_requests: Bucket capacity
            window_seconds: Refill interval
            cost: Request cost in tokens

        Returns:
            RateLimitResult
        """
        key = self._get_redis_key(identifier, endpoint)
        now = time.time()
        refill_rate = max_requests / window_seconds

        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local refill_rate = tonumber(ARGV[3])
        local cost = tonumber(ARGV[4])
        local burst_size = tonumber(ARGV[5])

        -- Get current bucket state
        local bucket = redis.call('HGETALL', key)
        local tokens = capacity
        local last_refill = now

        if #bucket > 0 then
            for i = 1, #bucket, 2 do
                if bucket[i] == 'tokens' then
                    tokens = tonumber(bucket[i + 1])
                elseif bucket[i] == 'last_refill' then
                    last_refill = tonumber(bucket[i + 1])
                end
            end

            -- Refill tokens
            local time_passed = now - last_refill
            local new_tokens = tokens + (time_passed * refill_rate)
            tokens = math.min(new_tokens, capacity + burst_size)
        end

        -- Check if request can be served
        if tokens >= cost then
            tokens = tokens - cost
            redis.call('HSET', key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', key, capacity * 2)
            return {1, math.floor(tokens)}
        else
            -- Calculate when tokens will be available
            local tokens_needed = cost - tokens
            local wait_time = tokens_needed / refill_rate
            return {0, wait_time}
        end
        """

        try:
            result = await self.redis_client.eval(
                lua_script, 1, key, now, max_requests, refill_rate, cost, self.config.burst_size
            )

            if result[0] == 1:
                return RateLimitResult(
                    allowed=True,
                    limit=max_requests,
                    remaining=int(result[1]),
                    reset_at=int(now + window_seconds),
                )
            else:
                wait_time = int(result[1])
                return RateLimitResult(
                    allowed=False,
                    limit=max_requests,
                    remaining=0,
                    reset_at=int(now + wait_time),
                    retry_after=wait_time,
                    reason="Token bucket exhausted",
                )

        except RedisError as e:
            logger.error(f"Redis error in token bucket check: {e}")
            return RateLimitResult(
                allowed=True,
                limit=max_requests,
                remaining=max_requests,
                reset_at=int(now + window_seconds),
            )

    async def _fixed_window_check(
        self,
        identifier: str,
        endpoint: str | None,
        max_requests: int,
        window_seconds: int,
        cost: int,
    ) -> RateLimitResult:
        """Fixed window rate limiting implementation.

        Args:
            identifier: Client identifier
            endpoint: Optional endpoint
            max_requests: Maximum requests per window
            window_seconds: Window size
            cost: Request cost

        Returns:
            RateLimitResult
        """
        now = time.time()
        window_id = int(now // window_seconds)
        key = f"{self._get_redis_key(identifier, endpoint)}:{window_id}"

        try:
            # Increment counter atomically
            current = await self.redis_client.incrby(key, cost)

            # Set expiry on first request in window
            if current == cost:
                await self.redis_client.expire(key, window_seconds * 2)

            if current <= max_requests:
                return RateLimitResult(
                    allowed=True,
                    limit=max_requests,
                    remaining=max_requests - current,
                    reset_at=int((window_id + 1) * window_seconds),
                )
            else:
                reset_at = int((window_id + 1) * window_seconds)
                return RateLimitResult(
                    allowed=False,
                    limit=max_requests,
                    remaining=0,
                    reset_at=reset_at,
                    retry_after=reset_at - int(now),
                    reason="Fixed window limit exceeded",
                )

        except RedisError as e:
            logger.error(f"Redis error in fixed window check: {e}")
            return RateLimitResult(
                allowed=True,
                limit=max_requests,
                remaining=max_requests,
                reset_at=int(now + window_seconds),
            )

    async def _track_violation(self, identifier: str):
        """Track rate limit violations for potential blocking.

        Args:
            identifier: Client identifier
        """
        violation_key = f"{self.key_prefix}:violations:{identifier}"

        try:
            violations = await self.redis_client.incr(violation_key)
            await self.redis_client.expire(violation_key, 3600)  # Track for 1 hour

            # Block after 5 violations in an hour
            if violations >= 5:
                await self._block_client(identifier)
                await self.redis_client.delete(violation_key)

        except RedisError as e:
            logger.error(f"Error tracking violations: {e}")

    async def reset_limits(self, identifier: str, endpoint: str | None = None):
        """Reset rate limits for a specific identifier.

        Args:
            identifier: Client identifier
            endpoint: Optional endpoint
        """
        key = self._get_redis_key(identifier, endpoint)
        block_key = self._get_block_key(identifier)
        violation_key = f"{self.key_prefix}:violations:{identifier}"

        try:
            await self.redis_client.delete(key, block_key, violation_key)
            logger.info(f"Reset rate limits for {identifier}")
        except RedisError as e:
            logger.error(f"Error resetting limits: {e}")

    async def get_usage_stats(self, identifier: str) -> dict[str, Any]:
        """Get usage statistics for an identifier.

        Args:
            identifier: Client identifier

        Returns:
            Usage statistics
        """
        stats = {"identifier": identifier, "endpoints": {}, "blocked": False, "violations": 0}

        try:
            # Check if blocked
            is_blocked, retry_after = await self._check_blocked(identifier)
            stats["blocked"] = is_blocked
            if retry_after:
                stats["unblock_in"] = retry_after

            # Get violations
            violation_key = f"{self.key_prefix}:violations:{identifier}"
            violations = await self.redis_client.get(violation_key)
            stats["violations"] = int(violations) if violations else 0

            # Get usage for different endpoints
            for endpoint in self.config.endpoint_limits:
                key = self._get_redis_key(identifier, endpoint)

                if self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                    count = await self.redis_client.zcard(key)
                    stats["endpoints"][endpoint] = {"requests": count}

        except RedisError as e:
            logger.error(f"Error getting usage stats: {e}")

        return stats


# Export main components
__all__ = ["DistributedRateLimiter", "RateLimitConfig", "RateLimitResult", "RateLimitStrategy"]
