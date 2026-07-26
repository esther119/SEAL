"""Stage 2 — APPS scoring harness (CPU, subprocess sandbox).

Replaces the MBPP assert-runner. GRADING SEMANTICS MIRROR SEAL's own grader
(code_evaluation/testing_util.py) — verified by adversarial audit:

  stdin/stdout (no fn_name): run the extracted program (with SEAL's
      import_string prepended, as grade_stdio does via make_function) once per
      test in a fresh `python -I` subprocess, pipe the input, swallow
      SystemExit (reference behavior), and compare stdout with the reference's
      all-or-nothing per-line rule: stripped lines equal exactly, OR both
      lines fully convert to Decimal token lists that compare EXACTLY equal
      ('0.50' == '0.5000'; deliberately NO isclose/tolerance — see the
      big-int gotcha comment in testing_util).
  call-based (fn_name): a static runner execs the solution (import_string
      prepended, matching grade_call_based), resolves the function
      (LeetCode-style `Solution` class or module-level), calls it per test
      with a per-test SIGALRM, and compares with plain == after top-level
      tuple->list normalization (testing_util lines 229-232 semantics; no
      float tolerance). Storage adaptations for codeparrot/apps, from the
      original Hendrycks APPS harness: args passed RAW (stored expected
      outputs assume the double-encoded strings as-is), expected accepted in
      raw or json-decoded form, and 1-element-list wrapping accepted.

Every child process runs isolated (`python -I`), in its own session (so a
timeout kills the whole process group), with a 4GB address-space rlimit set by
a bootstrap — NOT a security sandbox, same caveat as SEAL's reliability_guard.

Reused at eval time for pass@1 (same pass/fail semantics as extraction).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation

DEFAULT_TIMEOUT = 10       # seconds per test case
MEM_LIMIT_BYTES = 4 * 1024 ** 3

# APPS test data contains 9000+ digit integers; lift Python >=3.11's 4300-digit
# int<->str cap here (json/str of test values) and in every child (solutions
# that print huge ints would otherwise spuriously fail).
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

# Sets rlimits inside the child, then runs the target script as __main__ with
# sys.argv shifted so the script sees only its own args. Doing limits in-child
# avoids preexec_fn (unsafe with threads).
_BOOT_COMMON = (
    "import resource,runpy,sys\n"
    "hasattr(sys,'set_int_max_str_digits') and sys.set_int_max_str_digits(0)\n"
    "_target, _cpu = sys.argv[1], int(sys.argv[2])\n"
    "sys.argv = [_target] + sys.argv[3:]\n"
    "try:\n"
    f"    resource.setrlimit(resource.RLIMIT_AS,({MEM_LIMIT_BYTES},{MEM_LIMIT_BYTES}))\n"
    "except Exception:\n"
    "    pass\n"
    "try:\n"
    "    resource.setrlimit(resource.RLIMIT_CPU,(_cpu+1,_cpu+2))\n"
    "except Exception:\n"
    "    pass\n"
)
# stdio: swallow SystemExit like the reference (testing_util.call_method's
# `except SystemExit: pass`) — a program that prints the right answer then
# sys.exit(1) is graded on its output, not its exit code.
_BOOT_STDIO = _BOOT_COMMON + (
    "try:\n"
    "    runpy.run_path(_target, run_name='__main__')\n"
    "except SystemExit:\n"
    "    pass\n"
)
# call runner: the runner reports its verdict THROUGH the exit code, so
# SystemExit must propagate.
_BOOT = _BOOT_COMMON + "runpy.run_path(_target, run_name='__main__')\n"

# SEAL's testing_util.import_string, VERBATIM. The reference prepends it to
# BOTH stdio (grade_stdio -> make_function) and call-based (grade_call_based)
# solutions, so we do exactly the same for both kinds.
_IMPORT_STRING = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"

_CALL_RUNNER = r'''
import json, signal, sys

def _norm(x):
    # reference semantics (testing_util.py 229-231): top-level tuple -> list
    # only, plain == after that; NO float tolerance anywhere
    if isinstance(x, tuple):
        return list(x)
    return x

def _eq(a, b):
    return _norm(a) == _norm(b)

class _Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise _Timeout()

def _maybe_json(x):
    # APPS call-based data sometimes double-encodes string values
    # ('"aababcaab"' with literal quotes); the original Hendrycks harness
    # json-decodes them in a try/except, so we do too
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x

def main():
    with open(sys.argv[1]) as f:
        payload = json.load(f)
    with open(sys.argv[2]) as f:
        src = f.read()
    ns = {}
    try:
        exec(compile(src, "solution.py", "exec"), ns)
    except Exception:
        sys.exit(2)
    fn_name = payload["fn_name"]
    sol_cls = ns.get("Solution")
    if isinstance(sol_cls, type):
        try:
            fn = getattr(sol_cls(), fn_name, None)
        except Exception:
            sys.exit(2)
    else:
        fn = ns.get(fn_name)
    if not callable(fn):
        sys.exit(3)
    signal.signal(signal.SIGALRM, _alarm)
    per = int(payload.get("per_test_timeout", 10))
    for inp, exp in zip(payload["inputs"], payload["outputs"]):
        # args stay RAW: APPS expected outputs were generated with the
        # double-encoded strings passed as-is (e.g. balancedString('"QWER"')
        # -> -1 is the stored answer; the unquoted call would give 0)
        args = inp if isinstance(inp, list) else [inp]
        signal.alarm(per)
        try:
            res = fn(*args)
        except _Timeout:
            sys.exit(4)
        except Exception:
            sys.exit(2)
        finally:
            signal.alarm(0)
        # APPS ground truth sometimes wraps the expected value in a 1-list,
        # and sometimes double-encodes it — accept raw or decoded
        exps = [exp] + ([_maybe_json(exp)] if isinstance(exp, str) else [])
        ok = any(_eq(res, e) or _eq([res], e)
                 or (isinstance(e, list) and len(e) == 1 and _eq(res, e[0]))
                 for e in exps)
        if not ok:
            sys.exit(1)
    sys.exit(0)

main()
'''


def extract_code(gen):
    """Final code from a reasoning trace: prefer the answer AFTER </think>, take
    the LAST fenced block there (same policy as v1 gen_mbpp_vllm.extract_code)."""
    ans = gen.split("</think>", 1)[1] if "</think>" in gen else gen
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", ans, re.DOTALL)
    return blocks[-1].strip() if blocks else ans.strip()


def _run_child(argv, stdin_data, timeout):
    """Run one sandboxed child; returns (status, stdout) with
    status in {"ok", "error", "timeout"} (status "ok" = exit code 0).
    Binary-safe: non-UTF-8 output is replaced, never raises."""
    try:
        p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception:
        return "error", ""
    try:
        out_b, _ = p.communicate(input=stdin_data.encode("utf-8", "replace"),
                                 timeout=timeout)
        out = (out_b or b"").decode("utf-8", "replace")
        if p.returncode == 0:
            return "ok", out
        # RLIMIT_CPU fires before the wall clock for busy loops
        if p.returncode == -signal.SIGXCPU:
            return "timeout", ""
        return "error", out
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            p.kill()
        except Exception:
            pass
        try:
            p.communicate(timeout=5)
        except Exception:
            pass
        return "timeout", ""


def _as_text(x):
    """APPS inputs/outputs entries are usually strings but sometimes lists of
    lines — join them, like testing_util.call_method does."""
    if isinstance(x, (list, tuple)):
        return "\n".join(_as_text(v) for v in x)
    return str(x)


def _line_to_decimals(line):
    # testing_util.convert_line_to_decimals verbatim semantics: the ENTIRE
    # line must convert, token by token, or the line is not comparable
    try:
        return True, [Decimal(t) for t in line.split()]
    except (InvalidOperation, ValueError, ArithmeticError):
        return False, []


def compare_stdout(got, expected):
    """testing_util.grade_stdio comparison, verbatim semantics: same stripped
    line count; each line pair matches exactly, OR both lines fully convert to
    Decimal token lists that compare EXACTLY equal ('0.50' == '0.5000').
    Deliberately no isclose/tolerance — see the reference's big-int gotcha
    comment (np.isclose(5e16, 5e16+1) is True)."""
    g = [ln.strip() for ln in got.strip().split("\n")]
    e = [ln.strip() for ln in _as_text(expected).strip().split("\n")]
    if len(g) != len(e):
        return False
    for gl, el in zip(g, e):
        if gl == el:
            continue
        ok_g, gd = _line_to_decimals(gl)
        if not ok_g:
            return False
        ok_e, ed = _line_to_decimals(el)
        if not ok_e:
            return False
        if gd != ed:
            return False
    return True


def _child_argv(script_path, timeout, boot=_BOOT):
    return [sys.executable, "-I", "-c", boot, script_path, str(int(timeout))]


def score_stdio(code, tests, timeout, max_tests):
    inputs, outputs = tests["inputs"], tests["outputs"]
    if max_tests and max_tests > 0:
        inputs, outputs = inputs[:max_tests], outputs[:max_tests]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "solution.py")
        with open(path, "w") as f:
            # reference prepends import_string to stdio code too (make_function)
            f.write(_IMPORT_STRING + "\n" + code)
        for i, (inp, exp) in enumerate(zip(inputs, outputs)):
            status, out = _run_child(_child_argv(path, timeout, boot=_BOOT_STDIO),
                                     _as_text(inp), timeout + 2)
            if status != "ok":
                return {"passed": False, "n_run": i + 1, "fail": status}
            if not compare_stdout(out, exp):
                return {"passed": False, "n_run": i + 1, "fail": "wrong_answer"}
    return {"passed": True, "n_run": len(inputs), "fail": None}


def score_call_based(code, tests, timeout, max_tests):
    inputs, outputs = tests["inputs"], tests["outputs"]
    if max_tests and max_tests > 0:
        inputs, outputs = inputs[:max_tests], outputs[:max_tests]
    _FAIL = {1: "wrong_answer", 2: "runtime_error", 3: "fn_not_found", 4: "timeout"}
    with tempfile.TemporaryDirectory() as td:
        runner = os.path.join(td, "runner.py")
        sol = os.path.join(td, "solution.py")
        payload = os.path.join(td, "payload.json")
        with open(runner, "w") as f:
            f.write(_CALL_RUNNER)
        with open(sol, "w") as f:
            # reference prepends import_string to call-based code (grade_call_based)
            f.write(_IMPORT_STRING + "\n" + code)
        with open(payload, "w") as f:
            json.dump({"fn_name": tests["fn_name"], "inputs": inputs,
                       "outputs": outputs, "per_test_timeout": int(timeout)}, f)
        # per-test SIGALRM lives in the runner; overall wall clock covers all tests
        overall = timeout * len(inputs) + 10
        argv = [sys.executable, "-I", "-c", _BOOT, runner, str(int(overall)),
                payload, sol]
        try:
            p = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL,
                                 start_new_session=True)
        except Exception:
            return {"passed": False, "n_run": 0, "fail": "spawn_error"}
        try:
            rc = p.wait(timeout=overall)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
            try:
                p.kill()
                p.wait(timeout=5)
            except Exception:
                pass
            return {"passed": False, "n_run": len(inputs), "fail": "timeout"}
        if rc == 0:
            return {"passed": True, "n_run": len(inputs), "fail": None}
        return {"passed": False, "n_run": len(inputs),
                "fail": _FAIL.get(rc, f"exit_{rc}")}


def score_generation(gen, tests, timeout=DEFAULT_TIMEOUT, max_tests=0):
    """Score one reasoning trace against one problem's parsed tests.

    Returns {passed, kind, n_tests, n_run, fail}."""
    kind = "call" if tests.get("fn_name") else "stdio"
    code = extract_code(gen)
    if not code:
        return {"passed": False, "kind": kind, "n_tests": tests["n_tests"],
                "n_run": 0, "fail": "no_code"}
    if kind == "call":
        r = score_call_based(code, tests, timeout, max_tests)
    else:
        r = score_stdio(code, tests, timeout, max_tests)
    r.update({"kind": kind, "n_tests": tests["n_tests"]})
    return r


def score_many(gens, tests_list, timeout=DEFAULT_TIMEOUT, max_tests=0, workers=None):
    """Parallel scoring (threads; real work happens in child processes).
    Order-preserving."""
    workers = workers or min(16, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(
            lambda gt: score_generation(gt[0], gt[1], timeout, max_tests),
            zip(gens, tests_list)))


def main():
    """Re-score an existing math_eval.jsonl against APPS (sanity / eval reuse)."""
    from apps_data import load_apps, parse_tests
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_file", required=True, help="math_eval.jsonl to re-score")
    ap.add_argument("--split", default="train")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--max_tests", type=int, default=0, help="0 = all tests")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    with open(args.eval_file) as f:
        rows = [json.loads(ln) for ln in f]
    ds = load_apps(args.split)
    by_id = {ds[i]["problem_id"]: ds[i] for i in range(len(ds))}
    gens, tests_list = [], []
    for r in rows:
        row = by_id[r["problem_id"]]
        gens.append(r["model_generation"][0])
        tests_list.append(parse_tests(row))
    results = score_many(gens, tests_list, args.timeout, args.max_tests, args.workers)
    agree = sum(bool(r["all_eval"][0]) == res["passed"]
                for r, res in zip(rows, results))
    npass = sum(res["passed"] for res in results)
    print(f"[rescore] {len(rows)} traces | pass {npass} | "
          f"agreement with stored labels {agree}/{len(rows)}")


if __name__ == "__main__":
    main()
