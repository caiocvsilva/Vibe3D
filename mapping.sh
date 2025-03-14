#!/bin/bash

# Define the directory containing the files to rename
# get the path from the user
DIRECTORY=$1

# Read the mapping file line by line
# Read the mapping file line by line
while IFS=' ' read -r old_id new_id; do
  echo "Renaming files from $old_id* to $new_id.npy"

  # Use find to locate all files with the old ID in their names
  for file in "$DIRECTORY"/"$old_id"*".npy"; do
    if [ -e "$file" ]; then
      base="${file##$old_id}"                # Remove the old_id prefix from the filename
      new_filename="$new_id.npy"              # Construct the new filename

      echo "Renaming '$file' to '$DIRECTORY/$new_filename'"
      
      # Execute the mv command to rename the file
      mv "$file" "$DIRECTORY/$new_filename"
    fi
  done


done < $2

echo "All files have been renamed based on the mapping."