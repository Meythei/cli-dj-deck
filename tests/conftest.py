import pytest


@pytest.fixture(autouse=True)
def isolated_clidj_home(tmp_path_factory, monkeypatch):
    """Every test gets its own CLIDJ_HOME, so nothing reads or writes the
    developer's real config, library or cache."""
    home = tmp_path_factory.mktemp("clidj-home")
    monkeypatch.setenv("CLIDJ_HOME", str(home))
    return home
