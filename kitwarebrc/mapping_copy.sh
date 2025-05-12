#!/bin/bash

# Usage: ./copy_subject_folders_with_new_mapping.sh <source_root_dir> <target_root_dir> [mapping_file]
# Example: ./copy_subject_folders_with_new_mapping.sh /path/to/source /path/to/target subjectID_mapping.txt

set -e

SOURCE_ROOT="$1"
TARGET_ROOT="$2"
MAPPING_FILE="$3"

if [[ -z "$SOURCE_ROOT" || -z "$TARGET_ROOT" ]]; then
    echo "Usage: $0 <source_root_dir> <target_root_dir> [mapping_file]"
    exit 1
fi

# If no mapping file is provided, create one in the target directory
if [[ -z "$MAPPING_FILE" ]]; then
    MAPPING_FILE="$TARGET_ROOT/subjectID_mapping.txt"
    echo "Mapping file not provided. Creating new mapping file at $MAPPING_FILE."
    mkdir -p "$TARGET_ROOT"
    > "$MAPPING_FILE" # Create an empty mapping file
fi

# Create the target root directory if it doesn't exist
mkdir -p "$TARGET_ROOT"

# Generate mapping if the file is empty
if [[ ! -s "$MAPPING_FILE" ]]; then
    echo "Generating new mapping file..."
    subjects=($(ls -d "$SOURCE_ROOT"/* | xargs -n 1 basename | sort)) # List subject folders
    pad_width=5

    for idx in "${!subjects[@]}"; do
        oldname="${subjects[$idx]}"
        newname=$(printf "%0${pad_width}d" "$idx")
        echo "$oldname $newname" >> "$MAPPING_FILE"
    done
fi

# Loop through the mapping file and copy directories
while read -r oldname newname; do
    # Skip empty lines or comments in the mapping file
    [[ -z "$oldname" || -z "$newname" || "$oldname" == "#"* ]] && continue

    # Check if the source folder exists
    if [[ ! -d "$SOURCE_ROOT/$oldname" ]]; then
        echo "WARNING: Directory $SOURCE_ROOT/$oldname not found, skipping."
        continue
    fi

    # Check for conflicts in the target directory
    if [[ -e "$TARGET_ROOT/$newname" ]]; then
        echo "ERROR: Target directory $TARGET_ROOT/$newname already exists. Aborting."
        exit 2
    fi

    # Copy the folder to the new location with the new name
    cp -r "$SOURCE_ROOT/$oldname" "$TARGET_ROOT/$newname"
    echo "Copied $SOURCE_ROOT/$oldname -> $TARGET_ROOT/$newname"
done < "$MAPPING_FILE"

echo "Copying complete using mapping file $MAPPING_FILE."