"""Every HTTP call of betfair_client.py carries a timeout (EX-05, 24 Nov 2026): login and logout had none, so a
silent identity server could hang the live trader's re-login mid-session. A stub session records the kwargs; no
network call is made."""

import betfair_client as bc


class _Resp:
    def __init__(self, body):
        self._body = body
        self.text = ""

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


class _Session:
    def __init__(self):
        self.calls = []

    def post(self, url, **kw):
        self.calls.append((url, kw))
        if "certlogin" in url:
            return _Resp({"loginStatus": "SUCCESS", "sessionToken": "t"})
        if "login" in url:
            return _Resp({"status": "SUCCESS", "token": "t"})
        return _Resp({})


def _client():
    c = bc.BetfairClient.__new__(bc.BetfairClient)
    c.username, c.password, c.app_key = "u", "p", "k"
    c.cert_file, c.key_file = "c.crt", "c.key"
    c.session_token = None
    c._session = _Session()
    return c


def test_login_logout_and_api_calls_all_carry_a_timeout():
    c = _client()
    c._interactive_login()
    c._cert_login()
    c.session_token = "t"
    c._api_call("listEventTypes", {})
    c.logout()
    assert len(c._session.calls) == 4
    for url, kw in c._session.calls:
        assert kw.get("timeout"), url
    assert [kw["timeout"] for _, kw in c._session.calls] == [bc.LOGIN_TIMEOUT, bc.LOGIN_TIMEOUT, 30, bc.LOGIN_TIMEOUT]


def test_every_post_in_the_file_names_a_timeout():
    """The line held in the source: no self._session.post(...) call without a timeout argument."""
    import ast
    from pathlib import Path
    tree = ast.parse(Path(bc.__file__).read_text())
    posts = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr in ("post", "get") and "session" in ast.unparse(n.func.value)]
    assert posts and all(any(k.arg == "timeout" for k in n.keywords) for n in posts)
