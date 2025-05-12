#!/bin/bash

# Usage: ./rename_subject_folders.sh <root_dir>
# Example: ./rename_subject_folders.sh /path/to/data

set -e

ROOT="$1"
if [[ -z "$ROOT" ]]; then
    echo "Usage: $0 <root_dir>"
    exit 1
fi

cd "$ROOT"

# List subjectID directories, sort, store in array
subjects=($(ls -d */ | grep -E '^K[0-9]+/' | sed 's#/##' | sort))

mapping_file="subjectID_mapping.txt"
> "$mapping_file"

pad_width=5

for idx in "${!subjects[@]}"; do
    subj="${subjects[$idx]}"
    newname=$(printf "%0${pad_width}d" "$idx")
    # Only rename if different
    if [[ "$subj" != "$newname" ]]; then
        # Check for conflicts
        if [[ -e "$newname" ]]; then
            echo "ERROR: $newname already exists. Aborting."
            exit 2
        fi
        mv "$subj" "$newname"
        echo "$subj $newname" >> "$mapping_file"
    fi
done

echo "Renaming complete. Mapping saved to $mapping_file"
