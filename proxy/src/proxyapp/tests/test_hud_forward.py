import asyncio

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from proxyapp.proxy import UnifiedProxy


class _FakeResp:
    status = 204

    async def read(self):
        return b""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    captured = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def request(self, method, target, headers=None, data=None):
        type(self).captured = {
            "method": method,
            "target": target,
            "headers": {k: v for k, v in (headers or {}).items()},
            "data": data,
        }
        return _FakeResp()


def _admin_header(headers):
    for key, value in headers.items():
        if key.lower() == "x-hud-admin-key":
            return value
    return None


async def _forward_once(monkeypatch, inbound_headers):
    _FakeSession.captured = None
    monkeypatch.setenv("HUD_ADMIN_API_KEY", "env-admin-secret")
    monkeypatch.setattr("aiohttp.ClientSession", _FakeSession)
    proxy = UnifiedProxy.__new__(UnifiedProxy)
    app = web.Application()
    app.router.add_route("*", "/hud/{tail:.*}", proxy._hud_forward)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post("/hud/brief", headers=inbound_headers, json={})
        await response.read()
    finally:
        await client.close()
    return _FakeSession.captured


def test_forward_does_not_inject_admin_key(monkeypatch):
    captured = asyncio.run(_forward_once(monkeypatch, inbound_headers={}))
    assert captured is not None
    assert _admin_header(captured["headers"]) is None


def test_forward_preserves_client_admin_key(monkeypatch):
    captured = asyncio.run(
        _forward_once(
            monkeypatch,
            inbound_headers={"X-HUD-Admin-Key": "client-admin-secret"},
        )
    )
    assert captured is not None
    assert _admin_header(captured["headers"]) == "client-admin-secret"
