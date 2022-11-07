NVCC ?= nvcc
TARGET_EXEC ?= a.out

# EXEDIR=../test_execs
BUILD_DIR ?=./build
OBJ_DIR ?=$(BUILD_DIR)/o
EXE_DIR ?= $(BUILD_DIR)/exe

SRC_DIRS ?= ./src

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s -or -name *.cu)
SRCS_NAMES := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s -or -name *.cu -printf "%f\n")
OBJS := $(SRCS:%=$(BUILD_DIR)/obj/%.o)
EXES := $(SRCS:%=$(BUILD_DIR)/exe/%.exe)
DEBUG_OBJS := $(SRCS:%=$(BUILD_DIR)/debug_objs/%.o)
DEBUG_EXES := $(SRCS:%=$(BUILD_DIR)/debug_exes/%.exe)
DEPS := $(OBJS:.o=.d)

INCL_DIRS := #./include $(FREESTAND_DIR)/include 
INC_FLAGS := $(addprefix -I,$(INCL_DIRS))

LDFLAGS := -lcuda
CPPFLAGS ?= $(INC_FLAGS) -std=c++11 -O3
CUDAFLAGS ?= $(INC_FLAGS) -g -Xcompiler -rdynamic -lineinfo -O3 -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80
CUDADEBUGFLAGS ?= $(INC_FLAGS) -w -g -G -Xcompiler -rdynamic -O3 -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80

NVCCOPTIONS ?=

all: objs release_exes
dbg: debug_objs debug_exes

objs: $(OBJS)
debug_objs: $(DEBUG_OBJS)

release_exes: $(EXES)
debug_exes: $(DEBUG_EXES)

#Assemblies

$(BUILD_DIR)/exe/%.exe: $(BUILD_DIR)/obj/%.o
	$(MKDIR_P) $(dir $@)
	$(NVCC) $< -o $@ $(LDFLAGS)

$(BUILD_DIR)/debug_exes/%.exe: $(BUILD_DIR)/debug_objs/%.o
	$(MKDIR_P) $(dir $@)
	$(NVCC) $< -o $@ $(LDFLAGS)


# cuda source
$(BUILD_DIR)/obj/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

#cuda debug source
$(BUILD_DIR)/debug_objs/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(CUDADEBUGFLAGS) -c $< -o $@

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)
-include $(DEPS)

MKDIR_P ?= mkdir -p

# all: HungarianCUDA.cu
# 	nvcc -lineinfo --generate-code arch=compute_80,code=compute_80 HungarianCUDA.cu --compiler-options -mcmodel=medium \
# 	-std=c++11 -lcuda -o $(EXEDIR)/LAP19

# debug: HungarianCUDA.cu
# 	nvcc -rdc=true -G -g --generate-code arch=compute_70,code=compute_70 HungarianCUDA.cu \
# 	-std=c++11 -o $(DBGDIR)/LAP19_dbg

# clean:
# 	rm -f $(EXEDIR)/LAP19
# 	rm -f $(DBGDIR)/LAP19_dbg
# 	rm -f *.o