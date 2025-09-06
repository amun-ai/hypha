"""Test to reproduce the search_services bug."""

# Test data that might cause the issue
test_data = [
    {"config": {"visibility": "public"}},
    None,  # This is what causes the AttributeError
    {"config": {"visibility": "protected"}},
]

# Simulate the problematic code from search_services
for item in test_data:
    try:
        service_visibility = item.get("config", {}).get("visibility")
        print(f"Service visibility: {service_visibility}")
    except AttributeError as e:
        print(f"Error: 'NoneType' object has no attribute 'get'")
        print(f"Item causing error: {item}")
        print("This is the bug in search_services!")
