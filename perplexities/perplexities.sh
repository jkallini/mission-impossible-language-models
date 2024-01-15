#!/bin/sh
# perplexities.sh
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
Run perplexities for each perturbation type
-------------------------------------------------------------------------------
"

COMMAND="python3 perplexities.py hop_control hop_control 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py hop_tokens4 hop_tokens4 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py hop_words4 hop_words4 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py reverse_control reverse_control 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py reverse_full reverse_full 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py reverse_partial reverse_partial 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_control shuffle_control 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_nondeterministic shuffle_nondeterministic 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_deterministic21 shuffle_deterministic21 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_deterministic21 shuffle_deterministic57 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_deterministic21 shuffle_deterministic84 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_deterministic21 shuffle_nondeterministic 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_deterministic57 shuffle_deterministic21 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_deterministic57 shuffle_deterministic57 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_deterministic57 shuffle_deterministic84 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_deterministic57 shuffle_nondeterministic 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_deterministic84 shuffle_deterministic21 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_deterministic84 shuffle_deterministic57 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_deterministic84 shuffle_deterministic84 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_deterministic84 shuffle_nondeterministic 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_local3 shuffle_local3 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_local5 shuffle_local5 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_local10 shuffle_local10 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

COMMAND="python3 perplexities.py shuffle_even_odd shuffle_even_odd 100M $1 randinit $NO_POS_ENCODINGS"
echo $COMMAND
eval $COMMAND
echo "
"

echo "
-------------------------------------------------------------------------------
Done!
-------------------------------------------------------------------------------
"
