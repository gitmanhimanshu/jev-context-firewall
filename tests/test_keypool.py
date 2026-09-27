from jev_proxy.config import Config, parse_keys_from_env
from jev_proxy.jev_client import JevClient


async def test_api_key_pool_rotation():
    cfg = Config(
        jev_api_key="key-single",
        jev_api_keys=["key-1", "key-2", "key-3"],
        jev_concurrency_per_key=3,
    )

    keys = cfg.get_api_keys()
    assert len(keys) == 4
    assert cfg.get_concurrency() == 12

    client = JevClient(cfg)

    seen = {}
    for _ in range(8):
        k = await client.next_api_key()
        seen[k] = seen.get(k, 0) + 1

    for k in keys:
        assert seen[k] == 2


def test_comma_separated_keys():
    cfg = Config(
        jev_api_key="key-a, key-b , key-c",
        jev_concurrency_per_key=4,
    )
    keys = cfg.get_api_keys()
    assert len(keys) == 3
    assert keys == ["key-a", "key-b", "key-c"]
    assert cfg.get_concurrency() == 12


def test_render_format_keys():
    render_input = """sk-codiv-key1, sk-codiv-key2,
sk-codiv-key3
"sk-codiv-key4";sk-codiv-key5"""

    keys = parse_keys_from_env(render_input)
    assert len(keys) == 5
    assert keys == [
        "sk-codiv-key1",
        "sk-codiv-key2",
        "sk-codiv-key3",
        "sk-codiv-key4",
        "sk-codiv-key5",
    ]
