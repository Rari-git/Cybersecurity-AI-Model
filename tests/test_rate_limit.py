import pytest
from utils.rate_limit import TokenBucketRateLimiter


def test_bucket_starts_with_burst_and_then_limits():
    limiter = TokenBucketRateLimiter(rate=2.0, burst=2.0, max_keys=4)
    assert limiter.allow("sensor", 10.0)
    assert limiter.allow("sensor", 10.0)
    assert not limiter.allow("sensor", 10.0)
    assert limiter.allow("sensor", 10.5)
    assert not limiter.allow("sensor", 10.5)


def test_clock_rollback_does_not_create_tokens():
    limiter = TokenBucketRateLimiter(rate=1.0, burst=1.0, max_keys=2)
    assert limiter.allow("sensor", 100.0)
    assert not limiter.allow("sensor", 99.0)
    assert not limiter.allow("sensor", 99.5)
    assert limiter.allow("sensor", 101.0)


def test_zero_rate_allows_initial_burst_but_never_refills():
    limiter = TokenBucketRateLimiter(rate=0.0, burst=1.0, max_keys=2)
    assert limiter.allow("critical", 100.0)
    assert not limiter.allow("critical", 100.0)
    assert not limiter.allow("critical", 101.0)
    assert not limiter.allow("critical", 10000.0)


def test_zero_rate_supports_multiple_initial_burst_tokens():
    limiter = TokenBucketRateLimiter(rate=0.0, burst=2.0, max_keys=2)
    assert limiter.allow("critical", 0.0)
    assert limiter.allow("critical", 0.0)
    assert not limiter.allow("critical", 999.0)


def test_bucket_count_is_bounded():
    limiter = TokenBucketRateLimiter(rate=1.0, burst=1.0, max_keys=2)
    assert limiter.allow("a", 1.0)
    assert limiter.allow("b", 1.0)
    assert limiter.allow("c", 1.0)
    assert len(limiter.buckets) == 2 and "a" not in limiter.buckets


@pytest.mark.parametrize("kwargs", [
    {"rate": -0.01}, {"rate": float("nan")}, {"rate": float("inf")},
    {"burst": 0}, {"burst": float("inf")}, {"max_keys": 0}, {"max_keys": 1.5},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(**kwargs)


def test_nonfinite_timestamp_is_rejected():
    with pytest.raises(ValueError):
        TokenBucketRateLimiter().allow("sensor", float("nan"))
