"""Memory tests: the streamed fits stay inside a memory limit on data that
would not fit in it.

Run with ``pytest -m resource`` (the default run leaves them out: they take
about three minutes). Each fit runs in a child process under stream-limits'
``resource_runner``: a watchdog on every platform, or a kernel limit when
``STREAM_LIMITS_TEST_CGROUP_PARENT`` names a cgroup set up for it.

The binary logit is hdfe-stream's ``feglm_stream``, whose memory tests are
hdfe-stream's; these are this package's own models: the multinomial and
ordered logit with a fixed effect absorbed, and the conditional logit. With
``max_memory_gb=1.5`` the fits hold 0.7-1.2 GB of private memory (not
counting memory-mapped scratch files, which the kernel can reclaim), most of
it buffers whose sizes are capped. The limit is 1.6 GB. Measured on Linux
(WSL2, 2 threads):

- The multinomial and ordered logit, on 20 million people (1.05 GB in
  memory): 0.94 and 0.73 GB, about the same as on 2.5 million. A fit that
  held the data would need at least 1.8 GB.
- The conditional logit, on 8 million choice sets (21.6 million rows,
  0.98 GB): 1.0-1.3 GB from run to run, about what it held on 4 million
  before its pass 0 checked the sets a block at a time (1.17-1.22 GB). A
  fit that held the data would need at least 2 GB (one stopped at 1.63).

(A smaller budget is not a smaller fit: the buffers are there before the
multinomial logit checks that the fixed effect's blocks fit, and with
``max_memory_gb=1.0`` it finds no room for them and stops.)

The limit is on private memory, which stream-limits measures on Linux and
Windows. On macOS it measures resident memory, which counts the mapped
scratch files, so there the limit is a looser safety net.
On Windows, private memory is committed memory: a buffer or an allocator
segment counts in full once allocated, touched or not (an hdfe-stream logit
fit had 1.61 GB committed while 1.14 GB was resident). Measured in CI
(GitHub's windows-latest, 2 threads): multinomial 1.70 GB, ordered 1.76 GB,
conditional 1.79 GB; the limit there is 2.1 GB, which a fit holding the data
(at least 2.7 GB) would still exceed.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from logit_stream.simulate import RACES, simulate_choices, simulate_people, truth_people

pytestmark = pytest.mark.resource

N_PEOPLE = 20_000_000
N_CASES = 8_000_000                              # 21.6 million rows
BUDGET_GB = 1.5                                  # the fit's max_memory_gb
LIMIT_GB = {"darwin": 2.6, "win32": 2.1}.get(sys.platform, 1.6)
X = "age10 + female + log_income + owner + urban"
XS = X.split(" + ")


@pytest.fixture(scope="module")
def people(tmp_path_factory):
    path = tmp_path_factory.mktemp("big") / "people.parquet"
    d = simulate_people(N_PEOPLE, seed=1)
    d.write_parquet(path)
    assert d.estimated_size() > 1.0e9
    return path


@pytest.fixture(scope="module")
def choices(tmp_path_factory):
    path = tmp_path_factory.mktemp("big") / "choices.parquet"
    d = simulate_choices(N_CASES, seed=1)
    d.write_parquet(path)
    assert d.estimated_size() > 0.95e9
    return path


def _fit_in_child(resource_runner, call: str, path, workdir) -> dict:
    code = f"""
import json
from logit_stream import clogit_stream, mlogit_stream, ologit_stream
r = {call}
print(json.dumps(dict(converged=bool(r.converged),
                      coef={{k: float(v) for k, v in r.coef().items()}})))
"""
    code = code.replace("SRC", repr(str(path))).replace("WORK", repr(str(workdir)))
    result = resource_runner([sys.executable, "-c", code], memory_gb=LIMIT_GB,
                             threads=2, timeout=900)
    report = {k: result.as_dict()[k] for k in ("status", "mode", "memory_measure",
                                               "peak_private_bytes", "peak_rss_bytes",
                                               "elapsed_s")}
    print(json.dumps(report))
    assert result.status == "ok", (report, result.stderr[-3000:])
    if result.kernel:
        assert result.kernel["events"].get("oom_kill", 0) == 0
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["converged"]
    return out["coef"]


# The checks against the simulated coefficients are sanity checks only:
# agreement with statsmodels and scikit-learn is the rest of the suite's job.

def test_multinomial_logit_streams_inside_the_limit(resource_runner, people, tmp_path):
    coef = _fit_in_child(
        resource_runner,
        f"mlogit_stream('race ~ {X} | puma', SRC, workdir=WORK, verbose=False, "
        f"max_memory_gb={BUDGET_GB})",
        people, tmp_path / "work")
    # the base level is the first in sort order (AIAN); the truth's is White
    truth = truth_people()["race"]
    for race in RACES[1:]:
        for x in XS:
            fit = coef.get(f"{race}:{x}", 0.0) - coef[f"White:{x}"]
            assert abs(fit - truth[f"{race}:{x}"]) < 0.1, (race, x, fit)


def test_ordered_logit_streams_inside_the_limit(resource_runner, people, tmp_path):
    coef = _fit_in_child(
        resource_runner,
        f"ologit_stream('education ~ {X} | puma', SRC, workdir=WORK, verbose=False, "
        f"max_memory_gb={BUDGET_GB})",
        people, tmp_path / "work")
    truth = truth_people()["education"]
    np.testing.assert_allclose([coef[x] for x in XS], [truth[x] for x in XS], atol=0.03)


def test_conditional_logit_streams_inside_the_limit(resource_runner, choices, tmp_path):
    coef = _fit_in_child(
        resource_runner,
        "clogit_stream('chosen ~ cost + time + C(alt) + income:C(alt)', SRC, case='case', "
        f"workdir=WORK, verbose=False, max_memory_gb={BUDGET_GB})",
        choices, tmp_path / "work")
    # simulate_choices' defaults
    np.testing.assert_allclose([coef["cost"], coef["time"]], [-1.1, -1.3], atol=0.03)
