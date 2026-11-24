"""13 Nov, read-only: the runner as the new swap step leaves it (cores, memory, swap files, disk)."""
import subprocess

for cmd in ("nproc", "free -h", "swapon --show", "df -h / /mnt", "ls -la /ashcroft_swap /mnt/swapfile"):
    print("$", cmd)
    print(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout)
