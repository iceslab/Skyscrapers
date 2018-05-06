#!/bin/bash

if [ $# -ne 4 ]
then
	echo "generate_boards.bash <path to SkyscrapersCUDA> <start size> <end size> <number of boards per size>"
else
	GENERATOR=$1
	GENERATOR_DIR=$(dirname "${GENERATOR}")
	RESULT_DIR="./generated_boards"
	START_SIZE=$2
	END_SIZE=$3
	BORADS_PER_SIZE=$4

	if [ -e $GENERATOR ]
	then
		[ -d $RESULT_DIR ] || mkdir $RESULT_DIR
		for (( size=$START_SIZE; size<=$END_SIZE; size++ ))
		do
			for (( board=1; board<=$BORADS_PER_SIZE; board++ ))
			do
				echo "Generating board #$board of size $size"
				eval $GENERATOR -md $size
				mv "./lastRun.txt" "./generated_boards/${size}x${size}_${board}.txt"
			done
		done
		echo "Done generating boards"
	else
		echo "Invalid path to SkyscrapersCUDA"
	fi
fi
