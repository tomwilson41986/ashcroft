"""The live tracker on 21 Oct, mid-afternoon (the owner: "how are we doing today so far?"): done/track_1021_b.py again
(the day, the three changes as built, every market's EV and CLV now, the races settled so far). Read-only."""
import os
import runpy
import sys

sys.argv = ["track_1021_b.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "track_1021_b.py"), run_name="__main__")
