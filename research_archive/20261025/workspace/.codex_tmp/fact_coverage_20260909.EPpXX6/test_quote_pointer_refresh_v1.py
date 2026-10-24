from pathlib import Path
import refresh_quote_pointer_after_chain_fix_v1 as refresh


def setup(monkeypatch):
    pointer = refresh.pointer
    original_file = Path(pointer.__file__)
    monkeypatch.setattr(pointer, "dependencies", lambda: {str(original_file): refresh.sha(original_file)})
    monkeypatch.setattr(pointer, "configure_base", lambda: None)
    for name in ("PREFLIGHT", "LOG", "RECEIPT"):
        monkeypatch.setattr(pointer, name, getattr(pointer, name))
    refresh.configure()
    return pointer, original_file


def test_launch_path_switch_preserves_original_dependency_and_restores_module(monkeypatch):
    pointer, original_file = setup(monkeypatch)
    before = pointer.dependencies()
    def launch():
        assert Path(pointer.__file__) == Path(refresh.__file__)
        assert pointer.dependencies() == before
        assert str(original_file) in before and str(Path(refresh.__file__)) in before
        return 7
    monkeypatch.setattr(pointer, "launch", launch)
    assert refresh.launch() == 7
    assert Path(pointer.__file__) == original_file


def test_paths_only_use_new_receipts_and_preflight(monkeypatch):
    pointer, _ = setup(monkeypatch)
    assert pointer.PREFLIGHT == refresh.NEW_PREFLIGHT != refresh.OLD_PREFLIGHT
    assert pointer.RECEIPT == refresh.NEW_RECEIPT != refresh.OLD_RECEIPT
    assert pointer.LOG == refresh.NEW_LOG
