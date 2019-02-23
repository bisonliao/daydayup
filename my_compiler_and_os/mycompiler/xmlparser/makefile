CC			=g++
CFLAGS		=-I./inc -I./src -D_DEBUG  -DYY_SKIP_YYWRAP

SRC_DIR		=./src
INC_DIR		=./inc
OBJ_DIR		=./obj
LIB_DIR		=./lib
BIN_DIR		=./bin

objs=$(OBJ_DIR)/xmlnode.o $(OBJ_DIR)/xmlread.yy.o $(OBJ_DIR)/AnsiString.o $(OBJ_DIR)/xml.o \
	$(OBJ_DIR)/main.o $(OBJ_DIR)/xmlread2.zz.o $(OBJ_DIR)/xml2.o
objs2=$(OBJ_DIR)/xmlnode.o $(OBJ_DIR)/xmlread.yy.o $(OBJ_DIR)/AnsiString.o $(OBJ_DIR)/xml.o \
	$(OBJ_DIR)/xmlread2.zz.o $(OBJ_DIR)/xml2.o

all:tt ar_file

ar_file:$(BIN_DIR)/libxmlparser.a

$(BIN_DIR)/libxmlparser.a:$(objs2)
	ar -qv $@ $(OBJ_DIR)/xmlnode.o $(OBJ_DIR)/xmlread.yy.o $(OBJ_DIR)/AnsiString.o $(OBJ_DIR)/xml.o $(OBJ_DIR)/xmlread2.zz.o $(OBJ_DIR)/xml2.o

tt:$(objs)
	$(CC) -o $@ $^

$(OBJ_DIR)/main.o:$(SRC_DIR)/main.cpp
	$(CC) -c -o $@ $^ $(CFLAGS)
$(OBJ_DIR)/xml.o:$(SRC_DIR)/xml.cpp
	$(CC) -c -o $@ $^ $(CFLAGS)
$(OBJ_DIR)/xml2.o:$(SRC_DIR)/xml2.cpp
	$(CC) -c -o $@ $^ $(CFLAGS)
$(OBJ_DIR)/xmlnode.o:$(SRC_DIR)/xmlnode.cpp
	$(CC) -c -o $@ $^ $(CFLAGS)
$(OBJ_DIR)/AnsiString.o:$(SRC_DIR)/AnsiString.cpp
	$(CC) -c -o $@ $^ $(CFLAGS)
$(OBJ_DIR)/xmlread.yy.o:$(SRC_DIR)/xmlread.yy.cpp
	$(CC) -c -o $@ $^ $(CFLAGS) -I./inc/lex1
$(OBJ_DIR)/xmlread2.zz.o:$(SRC_DIR)/xmlread2.zz.cpp
	$(CC) -c -o $@ $^ $(CFLAGS) -I./inc/lex2
$(SRC_DIR)/xmlread.yy.cpp:$(SRC_DIR)/xmlread.lex
	flex -o$@ $^
$(SRC_DIR)/xmlread2.zz.cpp:$(SRC_DIR)/xmlread2.lex
	flex -Pzz -o$@ $^


clean:
	-rm ./obj/*.o
