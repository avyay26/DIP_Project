import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_demo_runs_on_cpu(tmp_path):
    start = time.time()
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_test.py"), "--device", "cpu",
                          "--steps", "10", "--out", str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr[-2000:]
    assert (tmp_path / "smoke.mp4").stat().st_size > 0
    assert len(list(tmp_path.glob("pair*_pred.png"))) == 9
    assert "expected 4*(4-1)+1 = 13" in res.stdout and "wrote 13 frames" in res.stdout
    assert time.time() - start < 60
