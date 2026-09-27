from jev_proxy.config import Config
from jev_proxy.router import ProviderRouter


def test_provider_router():
    cfg = Config()
    router = ProviderRouter(cfg)

    # Test 1: Standard Anthropic Path
    route1 = router.resolve_route(
        headers={}, query_params={}, req_path="/v1/messages"
    )
    assert route1.provider == "anthropic"
    assert route1.target_url == "https://api.anthropic.com/v1/messages"

    # Test 2: Standard OpenAI Path
    route2 = router.resolve_route(
        headers={}, query_params={}, req_path="/v1/chat/completions"
    )
    assert route2.provider == "openai"
    assert route2.target_url == "https://api.openai.com/v1/chat/completions"

    # Test 3: Path Prefix /proxy/anthropic/v1/messages
    route3 = router.resolve_route(
        headers={}, query_params={}, req_path="/proxy/anthropic/v1/messages"
    )
    assert route3.provider == "anthropic"
    assert route3.target_url == "https://api.anthropic.com/v1/messages"

    # Test 4: Custom Header X-Upstream-URL
    route4 = router.resolve_route(
        headers={"X-Upstream-URL": "https://api.anthropic.com"},
        query_params={},
        req_path="/v1/messages",
    )
    assert route4.is_custom is True
    assert route4.target_url == "https://api.anthropic.com/v1/messages"

    # Test 5: Query Parameter ?upstream=
    route5 = router.resolve_route(
        headers={},
        query_params={"upstream": "https://openrouter.ai/api"},
        req_path="/v1/chat/completions",
    )
    assert route5.is_custom is True
    assert route5.target_url == "https://openrouter.ai/api/v1/chat/completions"

    # Test 6: Provider Override via Header X-Provider
    route6 = router.resolve_route(
        headers={"X-Provider": "gemini"},
        query_params={},
        req_path="/v1/chat/completions",
    )
    assert route6.provider == "gemini"
