import pytest
from hypha.core.auth import parse_reconnection_token, generate_reconnection_token
from hypha.core import UserInfo

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_reconnection_token():
    """Test the reconnection token."""
    expires_in = 60 * 60 * 5  # 5 hours
    user_info = UserInfo(
        id="root",
        is_anonymous=False,
        email=None,
        parent=None,
        roles=[],
        scopes=[],
        expires_at=None,
    )
    client_id = "123"
    token = generate_reconnection_token(
        user_info, client_id, "my-workspace", expires_in=expires_in
    )
    assert token
    parsed = parse_reconnection_token(token)
    assert parsed
