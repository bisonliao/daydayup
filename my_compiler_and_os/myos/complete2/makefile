BIN_DIR		= ./bin
OBJ_DIR		=./obj
OBJS 		= make_libs $(BIN_DIR)/bootsec.bin $(BIN_DIR)/ext.bin $(BIN_DIR)/kernel.bin $(BIN_DIR)/mycp 

SWITCH_FLAG = -DSWITCH_BY_MYSELF 
CFLAGS 		= -Wall -O2 -I./inc -I./lib -I./fs -fno-builtin $(SWITCH_FLAG)
CC			= gcc


all: $(OBJS)
	$(BIN_DIR)/mycp a.img $(BIN_DIR)/bootsec.bin $(BIN_DIR)/ext.bin $(BIN_DIR)/kernel.bin 
	sz a.img 

make_libs:
	(cd ./lib; make)
	(cd ./fs; make)

inc/offset.h: inc/struct.h
	cd tool; gcc -o struct_member_offset struct_member_offset.c -I../inc; ./struct_member_offset >../inc/offset.h

$(BIN_DIR)/bootsec.bin:boot/bootsec.asm
	nasm -o $(BIN_DIR)/bootsec.bin boot/bootsec.asm
$(BIN_DIR)/ext.bin:boot/ext.asm
	nasm -o $(BIN_DIR)/ext.bin boot/ext.asm

$(BIN_DIR)/kernel.bin: $(OBJ_DIR)/head.o  $(OBJ_DIR)/main.o   $(OBJ_DIR)/asm.o \
	$(OBJ_DIR)/syscall.o $(OBJ_DIR)/test_proc.o $(OBJ_DIR)/api.o $(OBJ_DIR)/schedule.o\
	$(OBJ_DIR)/keyboard.o  $(OBJ_DIR)/task.o  $(OBJ_DIR)/proc_paging.o $(OBJ_DIR)/tty.o $(OBJ_DIR)/syscall_entry.o\
	$(OBJ_DIR)/hd.o $(OBJ_DIR)/hd_asm.o     $(OBJ_DIR)/libfs.a  $(OBJ_DIR)/global.o \
	$(OBJ_DIR)/time.o $(OBJ_DIR)/tool.o $(OBJ_DIR)/packet.o $(OBJ_DIR)/rtl8139.o $(OBJ_DIR)/dwio.o $(OBJ_DIR)/libcommon.a  \
    $(OBJ_DIR)/redefine.o
	ld   -Map system.map -Ttext C800  -e startup_32  $^ -o $(BIN_DIR)/kernel.bin --oformat binary


$(OBJ_DIR)/asm.o:  init/asm.S inc/offset.h
	$(CC) -o$@ -c $< -I./inc $(SWITCH_FLAG)
$(OBJ_DIR)/syscall_entry.o: init/syscall_entry.S inc/offset.h
	$(CC) -o$@ -c $< -I./inc
$(OBJ_DIR)/head.o: boot/head.S inc/offset.h
	$(CC) -o$@ -c $< -I./inc
$(OBJ_DIR)/hd_asm.o: kernel/blk_dev/hd_asm.S inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)

$(OBJ_DIR)/time.o: kernel/time.c inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/task.o: kernel/task.c inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/proc_paging.o: kernel/proc_paging.c inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/rtl8139.o: kernel/net_dev/rtl8139.c inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/packet.o: kernel/net_dev/packet.c inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/dwio.o: kernel/net_dev/dwio.S inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/hd.o: kernel/blk_dev/hd.c inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/tty.o: kernel/chr_dev/tty.c inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/keyboard.o: kernel/chr_dev/keyboard.c inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/syscall.o: init/syscall.c inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/tool.o: init/tool.c inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/global.o: init/global.c inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/schedule.o: init/schedule.c inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/main.o: init/main.c inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/test_proc.o: init/test_proc.c inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/api.o: coreutil/api.S inc/offset.h
	$(CC) -o$@ -c $< $(CFLAGS)

$(BIN_DIR)/mycp:tool/mycp.c inc/offset.h
	$(CC) -g3 -o $(BIN_DIR)/mycp tool/mycp.c

clean:
	rm -f $(OBJ_DIR)/*.o $(OBJS)  inc/offset.h
	(cd ./lib;make clean)
	(cd ./fs;make clean)
