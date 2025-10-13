#!/bin/bash

singularity exec ../vllm/vllm-openai_latest.sif python3 - <<'EOF'
import crypt, getpass, pathlib

username = "edr"
password = getpass.getpass("Password for edr: ")
hashed = crypt.crypt(password, crypt.mksalt(crypt.METHOD_SHA512))

pathlib.Path(".htpasswd").write_text(f"{username}:{hashed}\n")
print("✅ .htpasswd generated successfully at ./ .")
EOF