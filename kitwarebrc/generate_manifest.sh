#!/bin/bash

# Check if a root folder was provided, if not use the current directory.
ROOT_FOLDER="${1:-.}"

# Output manifest file
MANIFEST="video_manifest.txt"

# Remove manifest file if it already exists.
if [ -f "$MANIFEST" ]; then
  rm "$MANIFEST"
fi

# Change to the provided root folder and get its absolute path.
cd "$ROOT_FOLDER" || { echo "Cannot change directory to $ROOT_FOLDER"; exit 1; }
ABS_ROOT=$(pwd)

# Find all .mp4 files in the root folder (and its subdirectories) and save their full paths.
find "$ABS_ROOT" -type f -name "*.mp4" > "../$MANIFEST"

echo "Manifest generated at $(realpath "../$MANIFEST")"
