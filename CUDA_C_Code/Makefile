# NVCC_FLAGS := --extended-lambda -Xcompiler -O3 -rdc=true -gencode arch=compute_89,code=sm_89
NVCC_FLAGS := --extended-lambda -g -O0 -G -Xcompiler -Wall -rdc=true -gencode arch=compute_89,code=sm_89
LINK_FLAGS := -arch=sm_89  
INC_FLAGS := -lcudadevrt -lcurand -lcusparse -lcublas -lnvToolsExt -lcusolver -lcudart -lpthread -L/usr/lib/X11 -lX11 -I/usr/local/CImg-3.4.2
# C_FLAGS := -std=c++11 
NVCC := nvcc

INC := include
OBJ := obj
SRC := src
T_SRC := src_test
BIN := bin


IMM: $(BIN)/IMM
test: $(BIN)/test


$(BIN)/test: $(OBJ)/test.o $(OBJ)/err.o
	$(NVCC) $(LINK_FLAGS) $(INC_FLAGS) $^ -o $@

$(BIN)/IMM: $(OBJ)/main.o $(OBJ)/err.o
	$(NVCC) $(LINK_FLAGS) $(INC_FLAGS) $^ -o $@

$(OBJ)/test.o: $(T_SRC)/main_test.cu
	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@

$(OBJ)/main.o: $(SRC)/main.cu
	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@

$(OBJ)/err.o: $(INC)/GPUErrors.cu
	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@


clean:
	rm -rf $(OBJ)/*.o $(BIN)/*