"""The runner's disk and memory (22 Oct; read-only): iteration 111's feature-matrix build was shut down twice by the
runner (exit 143) on GitHub's new image. The sizes that decide it: the root disk and /mnt (the swapfile's home), the
database, and what a matrix build writes."""
import os
import shutil
import subprocess

for p in ("/", "/mnt", "/tmp", os.getcwd()):
    try:
        u = shutil.disk_usage(p)
        print(f"{p:12s} total {u.total / 1e9:6.1f} GB  used {u.used / 1e9:6.1f} GB  free {u.free / 1e9:6.1f} GB  dev {os.stat(p).st_dev}")
    except OSError as e:
        print(p, e)
print("horse_racing.db", round(os.path.getsize("horse_racing.db") / 1e9, 2), "GB")
print(subprocess.run(["free", "-h"], capture_output=True, text=True).stdout)
print(subprocess.run(["df", "-h"], capture_output=True, text=True).stdout)
print(subprocess.run(["du", "-sh", ".feature_cache"], capture_output=True, text=True).stdout or "no .feature_cache restored")
