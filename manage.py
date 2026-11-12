"""Cross-platform task runner (no make needed on Windows).

python manage.py setup      venv, dependencies, embedding models, Ollama model
python manage.py seed       build kb_v1/kb_v2 from data/, point alias kb_live -> kb_v2
python manage.py run        start the API on http://127.0.0.1:8000
python manage.py test       pytest (extra args are passed through, e.g. -m integration)
python manage.py demo       cited answer, conflict, abstain and a blocked injection (needs run)
python manage.py incident   reproduce the reindex incident step by step (offline)
python manage.py eval       fine-tuning evaluation (see finetune/)
python manage.py lint       ruff check + format check
"""

import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
DEFAULT_MODEL = "llama3.2:3b"


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(*cmd: str | Path) -> int:
    env = {**os.environ, "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONUTF8": "1"}
    return subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env).returncode


def venv_python() -> Path:
    if not VENV_PY.exists():
        die("virtualenv not found: run `python manage.py setup` first")
    return VENV_PY


def env_setting(name: str, default: str) -> str:
    """Read a setting from the environment or .env without importing app deps."""
    if name in os.environ:
        return os.environ[name]
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text("utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == name:
                return value.strip()
    return default


def ollama_reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def find_ollama() -> str | None:
    candidates = [
        env_setting("OLLAMA_BIN", ""),  # for installs outside PATH, e.g. a custom drive
        shutil.which("ollama"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
    ]
    return next((c for c in candidates if c and Path(c).exists()), None)


def setup() -> int:
    # Runs under whatever system Python invoked manage.py, which may be older than the project's.
    if sys.version_info < (3, 11):  # noqa: UP036
        die(f"Python 3.11+ required, found {sys.version.split()[0]}")
    if not VENV_PY.exists():
        print("creating .venv")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    py = venv_python()
    if run(py, "-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements.txt"):
        die("dependency install failed (see pip output above)")
    print("downloading embedding models (first run only)")
    probe = "from app.config import get_settings; from app.clients.embeddings import Embedder; "
    probe += "Embedder(get_settings()); print('embedding models ready')"
    if run(py, "-c", probe):
        die("embedding model download failed; delete .cache/fastembed and retry")
    return pull_ollama_model()


def pull_ollama_model() -> int:
    url = env_setting("OLLAMA_URL", "http://localhost:11434")
    model = env_setting("OLLAMA_MODEL", DEFAULT_MODEL)
    binary = find_ollama()
    if binary is None:
        print(
            "WARNING: Ollama not installed. Install it from https://ollama.com/download, then "
            "rerun setup. Tests do not need it; `run`/`demo` do."
        )
        return 0
    if not ollama_reachable(url):
        print(
            f"WARNING: Ollama not running at {url}: start it with `ollama serve`, then rerun setup."
        )
        return 0
    print(f"pulling {model} via Ollama (skipped quickly if already present)")
    return run(binary, "pull", model)


def seed() -> int:
    code = run(venv_python(), "scripts/build_index.py", "seed")
    if code:
        print("Hint: embedded Qdrant allows one process. Stop `manage.py run` before seeding.")
    return code


def serve() -> int:
    url = env_setting("OLLAMA_URL", "http://localhost:11434")
    if env_setting("LLM_BACKEND", "ollama") == "ollama" and not ollama_reachable(url):
        print(f"WARNING: Ollama not running at {url}: start it with `ollama serve`. /ask will 503.")
    return run(
        venv_python(), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"
    )


COMMANDS = {
    "setup": lambda args: setup(),
    "seed": lambda args: seed(),
    "run": lambda args: serve(),
    "test": lambda args: run(venv_python(), "-m", "pytest", *args),
    "demo": lambda args: run(venv_python(), "scripts/demo.py", *args),
    "incident": lambda args: run(venv_python(), "scripts/incident.py", *args),
    "eval": lambda args: run(venv_python(), "finetune/eval.py", *args),
    "lint": lambda args: (
        run(venv_python(), "-m", "ruff", "check", ".")
        or run(venv_python(), "-m", "ruff", "format", "--check", ".")
    ),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(COMMANDS[sys.argv[1]](sys.argv[2:]))


if __name__ == "__main__":
    main()
