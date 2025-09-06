import redis

# Connect to Redis
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Simulate service keys from both servers
test_keys = [
    "services:public|public:ws-user-root/rolling-update-server-1:hypha-login@1234",
    "services:public|public:ws-user-root/rolling-update-server-2:hypha-login@5678",
    "services:public|public:ws-user-root/rolling-update-server-1-abc123:some-service@9999",
    "services:public|public:public/rolling-update-server-2-xyz789:other-service@1111"
]

# The patterns used in the cleanup code
patterns = [
    "services:*|*:*/rolling-update-server-1:*@*",  # Direct server services
    "services:*|*:*/rolling-update-server-1-*:*@*",  # Services from server-generated clients
    "services:*|*:*/manager-rolling-update-server-1:*@*",  # Manager services
]

print("Testing pattern matching for server cleanup:")
print("=" * 60)

for key in test_keys:
    print(f"\nKey: {key}")
    for pattern in patterns:
        # Redis pattern matching uses simple wildcards
        # Convert to Python regex for simulation
        import re
        regex_pattern = pattern.replace("*", ".*").replace("?", ".")
        regex_pattern = "^" + regex_pattern + "$"
        
        if re.match(regex_pattern, key):
            print(f"  ✓ Matches pattern: {pattern}")
        else:
            print(f"  ✗ Does not match: {pattern}")

print("\n" + "=" * 60)
print("\nProblem identified:")
print("The pattern 'services:*|*:*/rolling-update-server-1-*:*@*' will match:")
print("  - rolling-update-server-1-abc123 (intended)")
print("But the wildcard after server-1 means it also incorrectly matches:")
print("  - rolling-update-server-2 (because '2' matches the wildcard after 'server-1-')")
