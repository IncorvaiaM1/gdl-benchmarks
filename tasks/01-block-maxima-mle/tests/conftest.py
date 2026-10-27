"""Emits tests/statistics.json from the pytest run: which testcases passed,
with each testcase's own docstring as its description. Uses only pytest's
built-in hooks so the suite gains no extra dependency."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_results = []


def pytest_runtest_logreport(report):
    if report.when != "call":
        return
    _results.append({
        "name": report.head_line.split("::")[-1] if "::" in report.head_line else report.head_line,
        "passed": report.outcome == "passed",
        "duration_seconds": round(report.duration, 4),
    })


def pytest_collectstart(collector):
    pass


def pytest_sessionfinish(session, exitstatus):
    # Attach docstrings by re-importing the test module.
    import tests.test_estimator as mod
    doc_by_name = {}
    for name in dir(mod):
        if name.startswith("test_"):
            fn = getattr(mod, name)
            doc_by_name[name] = " ".join((fn.__doc__ or "").split())

    tests_out = []
    for r in _results:
        tests_out.append({
            "name": r["name"],
            "passed": r["passed"],
            "duration_seconds": r["duration_seconds"],
            "description": doc_by_name.get(r["name"], ""),
        })

    summary = {
        "total": len(tests_out),
        "passed": sum(1 for t in tests_out if t["passed"]),
        "failed": sum(1 for t in tests_out if not t["passed"]),
        "exit_status": int(exitstatus),
    }

    out_path = Path(__file__).resolve().parent / "statistics.json"
    out_path.write_text(json.dumps({"summary": summary, "tests": tests_out}, indent=2))
