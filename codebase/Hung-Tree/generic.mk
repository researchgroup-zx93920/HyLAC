NVCC = nvcc
TARGET_EXEC ?= a.out

BUILD_DIR ?= ./build
SRC_DIRS ?= ./src
INC_DIRS ?= 
EXE_DIR ?= $(BUILD_DIR)/exec

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s -or -name *.cu)
SRCS_NAMES := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s -or -name *.cu -printf "%f\n")
OBJS := $(SRCS:%=$(BUILD_DIR)/obj/%.o)
EXES := $(SRCS:%=$(BUILD_DIR)/exe/%.exe)
DEPS := $(OBJS:.o=.d)

#INCL_DIRS := $(shell find $(INC_DIRS) -type d) ./include $(FREESTAND_DIR)/include 
INCL_DIRS := #./include $(FREESTAND_DIR)/include 
INC_FLAGS := $(addprefix -I,$(INCL_DIRS))
LDFLAGS := -lnvgraph
CPPFLAGS ?= $(INC_FLAGS) -Wall -pthread -MMD -MP -shared -fPIC -std=c++11 -O3 -mavx -ftree-vectorize -fopt-info-vec
CUDAFLAGS = $(INC_FLAGS) -g -w -lineinfo -std=c++11 -O3 -DCUDA -DNOT_IMPL -arch=sm_70 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75


all: objs exes

objs: $(OBJS)

exes: $(EXES)

$(BUILD_DIR)/exe/%.exe: $(BUILD_DIR)/obj/%.o
	$(MKDIR_P) $(dir $@)
	$(NVCC) $< -o $@ $(LDFLAGS)

# assembly
$(BUILD_DIR)/obj/%.s.o: %.s
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) -c $< -o $@

# c source
$(BUILD_DIR)/obj/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# c++ source
$(BUILD_DIR)/obj/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# cuda source
$(BUILD_DIR)/obj/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p