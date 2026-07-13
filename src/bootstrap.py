"""
Ensure runtime dependencies are installed (notebooks / local runs).

Usage in a notebook (first cell):

    import sys
    from pathlib import Path
    ROOT = Path("..").resolve()  # if cwd is notebooks/
    sys.path.insert(0, str(ROOT))
    from src.bootstrap import ensure_notebook_deps, setup_project_path
    setup_project_path()
    ensure_notebook_deps()
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Iterable

# import_name -> pip package name
_NOTEBOOK_DEPS: dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "loguru": "loguru",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "xgboost": "xgboost",
    "shap": "shap",
    "optuna": "optuna",
    "dotenv": "python-dotenv",
    "yaml": "pyyaml",
    "sqlalchemy": "sqlalchemy",
}

_EDA_DEPS: dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
}


def _missing(modules: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    return missing


def _pip_install(packages: list[str]) -> None:
    if not packages:
        return
    # Install into the *current* interpreter (the Jupyter kernel)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--disable-pip-version-check",
        *packages,
    ]
    print("Installing into:", sys.executable)
    print("Packages:", ", ".join(packages))
    subprocess.check_call(cmd)


def ensure_packages(
    mapping: dict[str, str] | None = None,
    *,
    quiet: bool = False,
) -> list[str]:
    """Install any missing packages from mapping {import_name: pip_name}."""
    mapping = mapping or _NOTEBOOK_DEPS
    missing_imports = _missing(mapping.keys())
    if not missing_imports:
        if not quiet:
            print("All required packages already installed.")
        return []

    to_install = sorted({mapping[name] for name in missing_imports})
    _pip_install(to_install)

    # Invalidate caches so newly installed packages are importable immediately
    importlib.invalidate_caches()

    still = _missing(missing_imports)
    if still:
        raise ModuleNotFoundError(
            "Failed to import after install: "
            + ", ".join(still)
            + "\n\nFix:\n"
            "  1) Restart the Jupyter kernel (Kernel → Restart)\n"
            "  2) Re-run the setup cell\n"
            "  3) Or manually: "
            f"%pip install {' '.join(to_install)}"
        )
    if not quiet:
        print("Installed and verified:", ", ".join(to_install))
        print("If imports still fail, restart the kernel once and re-run setup.")
    return to_install


def ensure_notebook_deps() -> list[str]:
    """Full ML stack for model / SHAP notebooks."""
    return ensure_packages(_NOTEBOOK_DEPS)


def ensure_eda_deps() -> list[str]:
    """Minimal deps for EDA notebook."""
    return ensure_packages(_EDA_DEPS)


def setup_project_path() -> str:
    """Add repo root to sys.path (works from notebooks/ or project root)."""
    from pathlib import Path

    here = Path.cwd().resolve()
    candidates = [here, here.parent, here.parent.parent]
    for root in candidates:
        if (root / "src").is_dir() and (root / "scripts").is_dir():
            root_s = str(root)
            if root_s not in sys.path:
                sys.path.insert(0, root_s)
            return root_s

    root = Path(__file__).resolve().parents[1]
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root_s
