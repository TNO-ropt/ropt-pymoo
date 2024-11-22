import importlib
import shutil
from pathlib import Path
from typing import Any

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _load_from_file(name: str, sub_path: str | None = None) -> Any:
    path = EXAMPLES_DIR
    if sub_path is not None:
        path = path / sub_path
    path = path / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rosenbrock_de(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("rosenbrock_nm")
    module.main()


def test_discrete_ga(tmp_path: Path, monkeypatch: Any) -> None:
    shutil.copyfile(
        EXAMPLES_DIR / "discrete_ga.yml",
        tmp_path / "discrete_ga.yml",
    )
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("discrete_ga")
    module.main()
