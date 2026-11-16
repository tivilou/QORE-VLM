"""One-command DPR/BGE top-eight experiment matrix. No implicit downloads."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.check_top8_delivery import check_delivery
check_delivery()
from applications.rag.top8_suite_v002 import main
if __name__=='__main__':
    from applications.rag.top8_cache_v002 import close_sequences
    try:raise SystemExit(main())
    finally:close_sequences()
