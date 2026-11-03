"""One-command DPR/BGE top-eight experiment matrix. No implicit downloads."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from applications.rag.top8_suite_v002 import main
if __name__=='__main__':raise SystemExit(main())
