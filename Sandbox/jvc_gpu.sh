#!/bin/sh
N=$1
RANGE=$2
REPEAT=1

echo "size: $N"
echo "range: $RANGE"

cd ../competitors/jvc-gpu/Src/

# ./test_gpu -memory 600000000 -table_min $TABLE_MIN -table_max $TABLE_MAX -random -integer -int -range $RANGE -single -runs $REPEAT 
./test_gpu -memory 600000000 -table_min $N -table_max $N -random -integer -int -single -range $RANGE -runs $REPEAT
