#!/bin/bash

tool="./bin/curate-json"
path="data"
rank="$path/dndcombat.com"
stat="$path/5e.tools"
file="$path/dnd-5e-monsters.csv"

echo "Curating D&D Monster data-set"
echo "  using Elo ranks from $rank"
echo "  and Stat-blocks from $stat"
echo ""
CMD="$tool -r '${rank}/*.json' -s '${stat}/bestiary-*.json' -o '${file}'"
echo "$CMD"
$CMD
