#!/usr/bin/env bash

cd exp/rajashree/jailbreaks
for f in */logs/*/*/run*.log */*/logs/*/*/run*.log; do
    # grep "Processing [0-9]* rows" limit 1, take number
    rows="$(grep -o -m 1 'Processing [0-9]* rows' "$f" | grep -o '[0-9]*')"
    printf "%d/%d %s\n" "$(grep 'Completed row' "$f" | wc -l)" "$rows" "$f"
done
