#!/bin/bash

c=$2
i=0
n=$(cat $1 |grep "^[0-9][0-9][0-9][[:space:]][0-9]\.[0-9]" |wc -l)
while (( $i < $n ))
do
	j=$((i+11))
	v1=$(echo $(cat $1 |grep "^[0-9][0-9][0-9][[:space:]][0-9]\.[0-9]" |tr "\n" ":" |tr "[[:space:]]" " " |tr ":" "\n" |cut -d " " -f 13 |head -$j |tail -n 11) |tr " " "+" |bc -l)
	v1=$(echo $v1/11 |bc -l)

	j=$((i+22))
	v2=$(echo $(cat $1 |grep "^[0-9][0-9][0-9][[:space:]][0-9]\.[0-9]" |tr "\n" ":" |tr "[[:space:]]" " " |tr ":" "\n" |cut -d " " -f 13 |head -$j |tail -n 11) |tr " " "+" |bc -l)
	v2=$(echo $v2/11 |bc -l)

	j=$((i+33))
	v3=$(echo $(cat $1 |grep "^[0-9][0-9][0-9][[:space:]][0-9]\.[0-9]" |tr "\n" ":" |tr "[[:space:]]" " " |tr ":" "\n" |cut -d " " -f 13 |head -$j |tail -n 11) |tr " " "+" |bc -l)
	v3=$(echo $v3/11 |bc -l)

	echo $c $v1 $v2 $v3 $(echo "($v1+$v2+$v3)/3" |bc -l)

	i=$((i+33))
	c=$((c+5))
done
