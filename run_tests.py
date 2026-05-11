"""
run_tests.py  —  Script tiện ích để chạy unit tests từ bất kỳ thư mục nào.

Cách dùng (từ thư mục project/ HOẶC bất kỳ đâu trong project):
    python run_tests.py
    python run_tests.py -v          # verbose
    python run_tests.py -k soc      # chỉ chạy test liên quan đến SOC
"""
import os
import sys
import subprocess

# Tìm project root
_this_dir  = os.path.dirname(os.path.abspath(__file__))
_proj_root = _this_dir  # run_tests.py đặt ở project/

# Đảm bảo chạy pytest với working dir = project root
args = sys.argv[1:]  # forward thêm flags như -v, -k, ...

cmd = [
    sys.executable, "-m", "pytest",
    "tests/test_physics.py",
    "--tb=short",
    "--no-header",
    "-v",
] + args

print(f">>> Chạy: {' '.join(cmd)}")
print(f">>> Từ  : {_proj_root}\n")

result = subprocess.run(cmd, cwd=_proj_root)
sys.exit(result.returncode)
