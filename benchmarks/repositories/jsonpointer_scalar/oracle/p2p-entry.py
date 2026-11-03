from pathlib import Path
import runpy,sys
if len(sys.argv)!=2:raise ValueError("Candidate required")
root=Path(__file__).resolve().parent
sys.argv=[str(root/"historical_check.py"),str(root/"case.json"),sys.argv[1],'p2p']
runpy.run_path(str(root/"historical_check.py"),run_name="__main__")
