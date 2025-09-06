import fnmatch

# The services that remained after cleanup
remaining_services = [
    'services:public|built-in:public/rolling-update-server-1:built-in@*',
    'services:public|generic:public/rolling-update-server-1:queue@*',
    'services:public|functions:public/rolling-update-server-1:hypha-login@*'
]

# The patterns used for cleanup
cleanup_patterns = [
    "services:*|*:*/rolling-update-server-1:*@*",  # Direct server services
    "services:*|*:*/rolling-update-server-1-*:*@*",  # Services from server-generated clients
    "services:*|*:*/manager-rolling-update-server-1:*@*",  # Manager services
]

print("Testing cleanup patterns against remaining services:")
print("=" * 60)

for service in remaining_services:
    print(f"\nService: {service}")
    matched = False
    for pattern in cleanup_patterns:
        if fnmatch.fnmatch(service, pattern):
            print(f"  ✓ SHOULD match pattern: {pattern}")
            matched = True
            break
    if not matched:
        print(f"  ✗ Does not match any cleanup pattern")
        
print("\n" + "=" * 60)
print("\nProblem: These services should have been cleaned up!")
print("The pattern 'services:*|*:*/rolling-update-server-1:*@*' should match them.")
