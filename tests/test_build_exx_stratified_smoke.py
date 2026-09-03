import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_exx_stratified_smoke.py"
SPEC = importlib.util.spec_from_file_location("build_exx_stratified_smoke", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_token_count_handles_mapping_with_flat_list() -> None:
    assert MODULE.token_count({"input_ids": [1, 2, 3]}) == 3


def test_token_count_handles_one_row_tensor_like_shape() -> None:
    class TensorLike:
        shape = (1, 17)

    assert MODULE.token_count({"input_ids": TensorLike()}) == 17


def test_token_count_handles_nested_lists() -> None:
    assert MODULE.token_count({"input_ids": [[1, 2, 3, 4]]}) == 4
