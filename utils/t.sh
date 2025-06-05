#!/bin/zsh
TARGET_DIRECTORY="/Volumes/MyPassport/new_data/Qing/20250526/MV-CA060-11GM (00G97005722)"
for item in "$TARGET_DIRECTORY"/*; do
  if [ -f "$item" ]; then
    # Check if the file name ends with '.avi' (case-insensitive match could be added with 'shopt -s nocasematch')
    if [[ "$item" == *.avi ]]; then
      echo "" # Newline for readability
      echo "--> Found AVI file: $(basename "$item")"
      echo "    Executing Python script..."
      echo "${item}"
      python ./extract_img.py --video_path "${item}"
    fi
  fi
done
