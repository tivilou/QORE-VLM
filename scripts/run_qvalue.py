"""One command from any working directory; choose the caller's Python environment."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from applications.rag.qvalue_portable_v001 import main

if __name__ == '__main__':
    raise SystemExit(main())
