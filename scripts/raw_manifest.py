#!/usr/bin/env python3
"""Record SHA-256 and size of gitignored round-4 raw outputs in results/round4/raw_manifest.json."""

import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent.parent
files = subprocess.run(["git", "ls-files", "--others", "--ignored", "--exclude-standard", "results/round4"],
                       cwd=root, capture_output=True, text=True, check=True).stdout.split()
man = {f: {"bytes": (root / f).stat().st_size, "sha256": hashlib.sha256((root / f).read_bytes()).hexdigest()}
       for f in sorted(files) if not f.endswith(".DS_Store")}
(root / "results/round4/raw_manifest.json").write_text(json.dumps(man, indent=2) + "\n")
print(f"{len(man)} files")
