NVCC_FLAGS := --extended-lambda -g -Xcompiler -O3 -rdc=true -gencode arch=compute_89,code=sm_89
# NVCC_FLAGS := --extended-lambda -g -G -Xcompiler -Wall -rdc=true -gencode arch=compute_89,code=sm_89
LINK_FLAGS := -arch=sm_89  
INC_FLAGS := -lcudadevrt -lcurand -lcusparse -lcublas -lnvToolsExt -lcusolver -lcudart
# C_FLAGS := -std=c++11 
NVCC := nvcc

INC := include
OBJ := obj
SRC := src
BIN := bin


IMM: $(BIN)/IMM

$(BIN)/IMM: $(OBJ)/main.o $(OBJ)/func.o $(OBJ)/err.o
	$(NVCC) $(LINK_FLAGS) $(INC_FLAGS) $^ -o $@

$(OBJ)/main.o: $(SRC)/main.cu
	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@

$(OBJ)/func.o: $(SRC)/functions.cu
	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@

# $(OBJ)/ml.o: $(SRC)/layers.cu
# 	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@

$(OBJ)/err.o: $(INC)/GPUErrors.cu
	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@


clean:
	rm -rf $(OBJ)/*.o $(BIN)/*