"""The horseracebase login tries again when the site has a bad moment (daily_predictions.login), and
the morning job fails out loud when it priced nothing (pipeline.predict).

Mechanics only: a stand-in session and a stand-in model run, no network."""

import pytest
import requests

import daily_predictions as dp

FORM = ('<html><head><title>HorseRaceBase Login</title></head><body><form method="post">'
        '<input type="hidden" name="csrf_token" value="abc"></form></body></html>')
NO_FORM = '<html><head><title>Just a moment...</title></head><body></body></html>'
RESULTS = '<html><body><form><input name="user" value="42"></form></body></html>'


class _Resp:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class _Session:
    """Serves the login page from a list, one per visit; the post and the results page always work."""

    def __init__(self, login_pages, post_text="Welcome back"):
        self.login_pages = list(login_pages)
        self.post_text = post_text
        self.posts = 0

    def get(self, url):
        if url != dp.LOGIN_URL:
            return _Resp(RESULTS)
        page = self.login_pages.pop(0)
        if isinstance(page, Exception):
            raise page
        return page if isinstance(page, _Resp) else _Resp(page)

    def post(self, url, data):
        self.posts += 1
        return _Resp(self.post_text)


@pytest.fixture
def waited(monkeypatch):
    monkeypatch.setenv("HRB_USERNAME", "someone")
    monkeypatch.setenv("HRB_PASSWORD", "secret")
    waits = []
    monkeypatch.setattr(dp.time, "sleep", waits.append)
    return waits


def test_a_page_without_its_form_is_tried_again(waited):
    s = _Session([NO_FORM, FORM])
    assert dp.login(s) == "42"
    assert s.posts == 1 and waited == [dp.LOGIN_WAITS[0]]


def test_a_request_that_fails_is_tried_again(waited):
    s = _Session([requests.ConnectionError("reset"), _Resp("Service Unavailable", 503), FORM])
    assert dp.login(s, waits=(1, 2)) == "42"
    assert waited == [1, 2]


def test_credentials_the_site_rejects_are_not_tried_again(waited):
    s = _Session([FORM, FORM], post_text="Your login details were not recognised")
    assert dp.login(s) is None
    assert s.posts == 1 and waited == []


def test_it_gives_up_after_the_last_wait(waited):
    s = _Session([NO_FORM] * 4)
    assert dp.login(s, waits=(1, 2, 3)) is None
    assert waited == [1, 2, 3] and s.posts == 0


def test_the_morning_job_fails_when_it_priced_nothing(monkeypatch):
    import pipeline.predict as job
    import ultra_betting.model.predict as model_predict

    monkeypatch.setattr(job, "ensure_database", lambda: None)
    monkeypatch.setattr(model_predict, "run_predictions", lambda **kw: [])
    with pytest.raises(SystemExit) as stopped:
        job.main()
    assert stopped.value.code == 1
