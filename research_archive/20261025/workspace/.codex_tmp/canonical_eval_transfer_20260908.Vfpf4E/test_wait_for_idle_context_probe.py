from wait_for_idle_context_probe import update_idle_since


def test_busy_period_resets_idle_window():
    since = update_idle_since(busy=False, now=10, idle_since=None)
    assert since == 10
    since = update_idle_since(busy=False, now=50, idle_since=since)
    assert since == 10
    since = update_idle_since(busy=True, now=55, idle_since=since)
    assert since is None
    assert update_idle_since(busy=False, now=60, idle_since=since) == 60
