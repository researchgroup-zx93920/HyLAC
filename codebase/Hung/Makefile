NVCC ?= nvcc
TARGET_EXEC ?= a.out

# EXEDIR=../test_execs
BUILD_DIR ?=./build
OBJ_DIR ?=$(BUILD_DIR)/o
EXE_DIR ?= $(BUILD_DIR)/exe

SRC_DIRS ?= .

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s -or -name *.cu)
SRCS_NAMES := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s -or -name *.cu -printf "%f\n")
OBJS := $(SRCS:%=$(BUILD_DIR)/obj/%.o)
EXES := $(SRCS:%=$(BUILD_DIR)/exe/%.exe)
DEBUG_OBJS := $(SRCS:%=$(BUILD_DIR)/debug_objs/%.o)
DEBUG_EXES := $(SRCS:%=$(BUILD_DIR)/debug_exes/%.exe)
ARCH := 80
# ARCH := $(shell ~/get_SM.sh)


INCL_DIRS := #./include $(FREESTAND_DIR)/include 
INC_FLAGS := $(addprefix -I,$(INCL_DIRS))

LDFLAGS := -lcuda -lgomp
CPPFLAGS ?= $(INC_FLAGS) -std=c++11 -O3
CUDAFLAGS ?= $(INC_FLAGS) -g -Xcompiler -fopenmp -lineinfo -O3 -arch=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=sm_$(ARCH) \
						-gencode=arch=compute_$(ARCH),code=compute_$(ARCH)
CUDADEBUGFLAGS ?= $(INC_FLAGS) -g -G -Xcompiler -fopenmp -O3 -arch=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=sm_$(ARCH) 
						-gencode=arch=compute_$(ARCH),code=compute_$(ARCH)

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
	@echo SM_VALUE IS $(ARCH)
-include $(DEPS)


MKDIR_P ?= mkdir -p
