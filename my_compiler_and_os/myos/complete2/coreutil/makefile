BIN_DIR		= .
OBJ_DIR		= .
OBJS 		=  $(BIN_DIR)/sh.bin $(BIN_DIR)/sh2.bin

CFLAGS 		= -Wall  -I../inc -I../lib -I../fs -fno-builtin
CC			= gcc

all: $(OBJS)

install:
	cd ../tool/fdisk; ./fdisk; sz c.img

####################################################
#他妈的，-Ttext的参数默认是16进制，一旦错了，要调半天
####################################################

$(BIN_DIR)/sh.bin: $(OBJ_DIR)/start.o $(OBJ_DIR)/sh.o  $(OBJ_DIR)/use_fpu_and_switch.o $(OBJ_DIR)/api.o ../obj/libcommon.a   ../obj/redefine.o
	ld   -Map system_sh.map -Ttext 400000  -e _start  $^ -o $(BIN_DIR)/sh.bin --oformat binary

$(BIN_DIR)/sh2.bin: $(OBJ_DIR)/start.o $(OBJ_DIR)/sh2.o $(OBJ_DIR)/use_fpu.o  $(OBJ_DIR)/api.o ../obj/libcommon.a  ../obj/redefine.o
	ld   -Map system_sh2.map -Ttext 400000  -e _start  $^ -o $(BIN_DIR)/sh2.bin --oformat binary


$(OBJ_DIR)/use_fpu.o: ./use_fpu.S
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/use_fpu_and_switch.o: ./use_fpu_and_switch.S
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/sh2.o: ./sh2.c
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/start.o: ./start.S
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/sh.o: ./sh.c
	$(CC) -o$@ -c $< $(CFLAGS)
$(OBJ_DIR)/api.o: ./api.S
	$(CC) -o$@ -c $< $(CFLAGS)

clean:
	rm -f $(OBJ_DIR)/*.o $(OBJS) 
