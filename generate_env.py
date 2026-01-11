import subprocess
import sys
from pathlib import Path

env_name = sys.argv[1] if len(sys.argv) > 1 else sys.exit("Usage: python generate_env.py <conda_env_name>")

# get conda prefix
out = subprocess.check_output(["conda", "env", "list"], text=True)

prefix = None
for line in out.splitlines():
    if line.startswith(env_name + " "):
        prefix = line.split()[-1]
        break

if not prefix:
    sys.exit(f"Conda env '{env_name}' not found")

# write .env
Path(".env").write_text(
    f"CUDA_HOME={prefix}\n"
    f"LD_LIBRARY_PATH={prefix}/lib\n"
)

print(f".env generated for {env_name}")
