#!/bin/sh

if [ $# -ne 1 ]
then
	echo "preass filename"
	exit 1
fi

sed  \
	-e 's/\$zero/\$0/g' \
	-e 's/\$v0/\$2/g' \
	-e  's/\$v1/\$3/g'  \
	-e  's/\$a0/\$4/g' \
	-e  's/\$a1/\$5/g' \
	-e  's/\$a2/\$6/g' \
	-e  's/\$a3/\$7/g' \
	-e  's/\$t0/\$8/g' \
	-e  's/\$t1/\$9/g' \
	-e  's/\$t2/\$10/g' \
	-e  's/\$t3/\$11/g' \
	-e  's/\$t4/\$12/g' \
	-e  's/\$t5/\$13/g' \
	-e  's/\$t6/\$14/g'  \
	-e  's/\$t7/\$15/g' \
	-e  's/\$t8/\$24/g' \
	-e  's/\$t9/\$25/g' \
	-e  's/\$s0/\$16/g' \
	-e  's/\$s1/\$17/g' \
	-e  's/\$s2/\$18/g' \
	-e  's/\$s3/\$19/g' \
	-e  's/\$s4/\$20/g' \
	-e  's/\$s5/\$21/g' \
	-e  's/\$s6/\$22/g' \
	-e  's/\$s7/\$23/g'  \
	-e  's/\$k0/\$26/g' \
	-e  's/\$k1/\$27/g' \
	-e  's/\$gp/\$28/g' \
	-e  's/\$sp/\$29/g' \
	-e  's/\$fp/\$30/g' \
	-e  's/\$ra/\$31/g' $1