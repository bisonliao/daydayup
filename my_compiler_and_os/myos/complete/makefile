BIN_DIR		= ./bin
OBJ_DIR		=./obj
OBJS 		= $(BIN_DIR)/bootsec.bin $(BIN_DIR)/ext.bin $(BIN_DIR)/kernel.bin $(BIN_DIR)/mycp 

CFLAGS 		=  -O2 -I./inc -I./lib -I./fs -fno-builtin
CC			= gcc

all: $(OBJS)
	$(BIN_DIR)/mycp a.img $(BIN_DIR)/bootsec.bin $(BIN_DIR)/ext.bin $(BIN_DIR)/kernel.bin 
	sz a.img 

$(BIN_DIR)/bootsec.bin:boot/bootsec.asm
	nasm -o $(BIN_DIR)/bootsec.bin boot/bootsec.asm
$(BIN_DIR)/ext.bin:boot/ext.asm
	nasm -o $(BIN_DIR)/ext.bin boot/ext.asm

$(BIN_DIR)/kernel.bin: $(OBJ_DIR)/head.o  $(OBJ_DIR)/main.o   $(OBJ_DIR)/asm.o \
	$(OBJ_DIR)/global.o $(OBJ_DIR)/syscall.o $(OBJ_DIR)/test_proc.o $(OBJ_DIR)/api.o $(OBJ_DIR)/schedule.o\
	$(OBJ_DIR)/keyboard.o  $(OBJ_DIR)/task.o $(OBJ_DIR)/tty.o $(OBJ_DIR)/syscall_entry.o\
	$(OBJ_DIR)/hd.o $(OBJ_DIR)/hd_asm.o  $(OBJ_DIR)/libfs.a $(OBJ_DIR)/libcommon.a
	ld   -Map system.map -Ttext C800  -e startup_32  $^ -o $(BIN_DIR)/kernel.bin --oformat binary

$(OBJ_DIR)/libcommon.a:
	(cd ./lib; make)
$(OBJ_DIR)/libfs.a:
	(cd ./fs; make)

$(OBJ_DIR)/asm.o: init/asm.S
	$(CC) -o$@ -c $^ -I./inc
$(OBJ_DIR)/syscall_entry.o: init/syscall_entry.S
	$(CC) -o$@ -c $^ -I./inc
$(OBJ_DIR)/head.o: boot/head.S
	$(CC) -o$@ -c $^ -I./inc
$(OBJ_DIR)/hd_asm.o: kernel/blk_dev/hd_asm.S
	$(CC) -o$@ -c $< $(CFLAGS)

$(OBJ_DIR)/task.o: kernel/task.c
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/hd.o: kernel/blk_dev/hd.c
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/tty.o: kernel/chr_dev/tty.c
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/keyboard.o: kernel/chr_dev/keyboard.c
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/syscall.o: init/syscall.c
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/global.o: init/global.c
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/schedule.o: init/schedule.c
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/main.o: init/main.c
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/test_proc.o: init/test_proc.c
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/api.o: init/api.S
	$(CC) -o$@ -c $< $(CFLAGS)

$(BIN_DIR)/mycp:tool/mycp.c
	$(CC) -g3 -o $(BIN_DIR)/mycp tool/mycp.c

clean:
	rm -f $(OBJ_DIR)/*.o $(OBJS) 
	(cd ./lib;make clean)
	(cd ./fs;make clean)
