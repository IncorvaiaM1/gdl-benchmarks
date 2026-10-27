"""Export task instructions and public inputs without evaluator answers."""
import argparse
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[1]
TASKS = ('01-block-maxima-mle', '02-density-comparison', '03-gan-objectives')
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=TASKS)
    args = p.parse_args()
    src = ROOT / 'tasks' / args.task
    dst = ROOT / 'exports' / args.task
    if dst.exists(): p.error('Export already exists; relocate it before creating a fresh workspace.')
    (dst / 'task/solution').mkdir(parents=True)
    (dst / 'task/results').mkdir()
    for name in ('TASK.md', 'INSTALL.md', 'PROJECT_INSTRUCTION.md'):
        f = src / 'task' / name
        if f.exists(): shutil.copy2(f, dst / 'task' / name)
    for name in ('train.csv', 'images.csv', 'DOWNLOAD.md'):
        f = src / 'data' / name
        if f.exists():
            (dst / 'data').mkdir(exist_ok=True)
            shutil.copy2(f, dst / 'data' / name)
    print(dst)
if __name__ == '__main__': main()
