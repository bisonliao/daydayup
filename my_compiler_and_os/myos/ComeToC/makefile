
all: pm.bin ext.bin boot.bin mycp
	mycp a.img pm.bin ext.bin boot.bin 
	sz a.img 

pm.bin:pm.asm
	nasm -o pm.bin pm.asm
ext.bin:ext.asm
	nasm -o ext.bin ext.asm

boot.bin: head.S a.c
	gcc -c a.c
	gcc -c head.S
	# By default the linker ld page-aligns the data segment. 
	# We can turn this off by using the parameter -N with the linker ld.
	# then data segment is right after the code segment
	# of course, we can use -Tdata to set data segment start address
	ld  -N -Ttext 51200  -e startup_32  a.o head.o -o boot.bin --oformat binary

mycp:mycp.c
	gcc -g3 -o mycp mycp.c

clean:
	rm -f pm.bin ext.bin boot.bin mycp head.o a.o
