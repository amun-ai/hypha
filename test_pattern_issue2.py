import fnmatch

# Simulate service keys from both servers
test_keys = [
    "services:public|public:ws-user-root/rolling-update-server-1:hypha-login@1234",
    "services:public|public:ws-user-root/rolling-update-server-2:hypha-login@5678",
    "services:public|public:ws-user-root/rolling-update-server-1-abc123:some-service@9999",
    "services:public|public:public/rolling-update-server-2-xyz789:other-service@1111"
]

# The patterns used in the cleanup code for server-1
patterns_server1 = [
    "services:*|*:*/rolling-update-server-1:*@*",  # Direct server services
    "services:*|*:*/rolling-update-server-1-*:*@*",  # Services from server-generated clients
    "services:*|*:*/manager-rolling-update-server-1:*@*",  # Manager services
]

print("Testing pattern matching for rolling-update-server-1 cleanup:")
print("=" * 70)

for key in test_keys:
    print(f"\nKey: {key}")
    matched = False
    for pattern in patterns_server1:
        # Redis uses glob-style pattern matching
        if fnmatch.fnmatch(key, pattern):
            print(f"  ✓ MATCHES: {pattern}")
            matched = True
            break
    if not matched:
        print(f"  ✗ No match")

print("\n" + "=" * 70)
print("\nAnalysis:")
print("Pattern 'services:*|*:*/rolling-update-server-1-*:*@*' is problematic!")
print("It matches both:")
print("  - rolling-update-server-1-abc123:... (intended)")
print("  - rolling-update-server-2:... (NOT intended!)")
print("\nThis happens because 'rolling-update-server-1-*' matches 'rolling-update-server-2'")
print("where the wildcard '*' after 'server-1-' matches '2:hypha-login'")
