#!/bin/sh
# edge_probing.sh
# author: Julie Kallini

echo "
-------------------------------------------------------------------------------
Arguments
-------------------------------------------------------------------------------
"
echo "Random seed: $1"
NO_POS_ENCODINGS=${2:-''}
echo "No pos encodings: $NO_POS_ENCODINGS"

echo "
-------------------------------------------------------------------------------
Run edge probing for each perturbation type
-------------------------------------------------------------------------------
"

COMMAND="python3 edge_probing.py hop_control 100M $1 randinit mean $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 edge_probing.py hop_tokens4 100M $1 randinit mean $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 edge_probing.py hop_words4 100M $1 randinit mean $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 edge_probing.py reverse_control 100M $1 randinit mean $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 edge_probing.py reverse_full 100M $1 randinit mean $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 edge_probing.py reverse_partial 100M $1 randinit mean $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

echo "
-------------------------------------------------------------------------------
Done!
-------------------------------------------------------------------------------
"
