from pathlib import Path
import runpy
HERE=Path(__file__).resolve().parent
for p in sorted(HERE.glob('figure_*.py')):
 print('running',p.name); runpy.run_path(str(p),run_name='__main__')
