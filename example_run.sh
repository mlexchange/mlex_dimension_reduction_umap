#!/usr/bin/env bash
set -a                # export all variables
source .env           # read all key=value pairs
set +a                # auto-export off

# Create a temporary file for parameter substitution
TMPFILE=$(mktemp)

# Substitute environment variables in the temporary YAML file
envsubst < example_yamls/example_umap.yaml > "$TMPFILE"

# Run UMAP
python src/umap_run.py "$TMPFILE"

# Cleanup the temporary file
rm "$TMPFILE"
