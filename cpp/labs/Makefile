# Build the labs:   make            (everything except the trained model)
#                   make trained    (the trained LightGBM model; run gen_trained_model.py first)
#                   make clean
# Override the compiler or flags:  make CXX=clang++   make OPT=-O1

CXX      ?= g++
OPT      ?= -O3
CXXFLAGS ?= -std=c++20 $(OPT) -march=native -g -Wall -Wextra
B        := build

LABS := lab01_memory_hierarchy lab02_data_layout lab03_branch_prediction lab04_timers_and_dce \
        lab05_noise_and_jitter lab06_simd_ofi lab08_parallel_scaling

all: $(addprefix $(B)/,$(LABS)) $(B)/lab03_branch_prediction_O3 $(B)/lab06_simd_ofi_fastmath

$(B)/%: %.cpp common.hpp | $(B)
	$(CXX) $(CXXFLAGS) $< -o $@

# Lab 03's default build forbids the compiler from removing the branch (if-conversion to cmov,
# vectorization), so you see what the hardware does with a real branch. The _O3 build is a normal build.
NOBRANCHOPT := -fno-if-conversion -fno-if-conversion2 -fno-tree-vectorize
$(B)/lab03_branch_prediction: lab03_branch_prediction.cpp common.hpp | $(B)
	$(CXX) $(subst $(OPT),-O2,$(CXXFLAGS)) $(NOBRANCHOPT) $< -o $@
$(B)/lab03_branch_prediction_O3: lab03_branch_prediction.cpp common.hpp | $(B)
	$(CXX) $(subst $(OPT),-O3,$(CXXFLAGS)) $< -o $@

# Lab 08 needs OpenMP
$(B)/lab08_parallel_scaling: lab08_parallel_scaling.cpp common.hpp | $(B)
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

$(B)/lab06_simd_ofi_fastmath: lab06_simd_ofi.cpp common.hpp ../include/features_avx2.hpp | $(B)
	$(CXX) $(CXXFLAGS) -ffast-math $< -o $@

trained: $(B)/trained_model_benchmark
$(B)/trained_model_benchmark: ../src/trained_model_benchmark.cpp ../generated/btc_lightgbm_model.hpp | $(B)
	$(CXX) $(CXXFLAGS) -I../generated $< -o $@

$(B):
	mkdir -p $(B)

clean:
	rm -rf $(B)

.PHONY: all trained clean
