all: meanShiftGlobalMem meanShiftSharedMem

meanShiftGlobalMem:meanShiftGlobalMem.cu
	nvcc meanShiftGlobalMem.cu -o glMem -lm

meanShiftSharedMem:meanShiftSharedMem.cu
	nvcc meanShiftSharedMem.cu -o shMem -lm

clear:
	rm glMem shMem
