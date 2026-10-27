"""Check reference-generated answers against original task graders."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
def main():
    for i, task in enumerate(sorted((ROOT / 'tasks').iterdir())):
        spec = importlib.util.spec_from_file_location('reference', task / 'gt_code/reference_solution.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if i == 0:
            y = np.loadtxt(task / 'data/train.csv', delimiter=',', skiprows=1)
            answer = {'lambda_hat': float(mod.mle_grouped_max(y, 4))}
            name = 'estimate.json'
        elif i == 1:
            answer = mod.solve(mod.load_images(), **mod.PARAMS)
            name = 'answer.json'
        else:
            answer = mod.solve()
            name = 'answer.json'
        dst = task / 'task/results' / name
        prior = dst.read_bytes() if dst.exists() else None
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.write_text(json.dumps(answer, indent=2), encoding='utf-8')
            subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q'], cwd=task, check=True)
        finally:
            if prior is None: dst.unlink(missing_ok=True)
            else: dst.write_bytes(prior)
if __name__ == '__main__': main()
