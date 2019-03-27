#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <cassert>

// convert local index to global index
//index -> (row, col, inchimera=8)
//
struct RCIStrct{
	uint32_t i;
	uint32_t j;
	uint32_t index;
};

__host__ __device__ __forceinline__ uint32_t glIdx_RCI(
		uint32_t row, uint32_t col,
		uint32_t i, uint32_t j, uint32_t index){
	assert(i < row);
	assert(j < col);
	assert(index < 8);
	return (col*8)*i + 8*j + index;
}

__host__ __device__ __forceinline__ RCIStrct fromIdx_RCI(
		uint32_t row, uint32_t col,
		uint32_t glIdx){
	assert(glIdx < row*col*8);
	RCIStrct str;
	str.i = glIdx/(col*8);
	glIdx -= (col*8)*str.i;
	str.j = glIdx/8;
	glIdx -= 8*str.j;
	str.index = glIdx;
	return str;
}

// convert local index to global index
//index -> (trot, row, col, inchimera=8)
//
struct TRCIStrct{
	uint32_t t;
	uint32_t i;
	uint32_t j;
	uint32_t index;
};

__host__ __device__ __forceinline__ uint32_t glIdx_TRCI(
		uint32_t trot, uint32_t row, uint32_t col,
		uint32_t t, uint32_t i, uint32_t j, uint32_t index){
	assert(t < trot);
	assert(i < row);
	assert(j < col);
	assert(index < 8);
	return (row*col*8)*t + (col*8)*i + 8*j + index;
}

__host__ __device__ __forceinline__ uint32_t glIdx_TRCI_ext(
		uint32_t trot, uint32_t row, uint32_t col,
		int32_t t, int32_t i, int32_t j, uint32_t index){
	return glIdx_TRCI(trot+2, row+2, col+2, (uint32_t)(t+1), (uint32_t)(i+1), (uint32_t)(j+1), index);
}

__host__ __device__ __forceinline__ TRCIStrct fromIdx_TRCI(
		uint32_t trot, uint32_t row, uint32_t col,
		uint32_t glIdx){
	assert(glIdx < trot*row*col*8);
	TRCIStrct str;
	str.t = glIdx/(row*col*8);
	glIdx -= (row*col*8)*str.t;
	str.i = glIdx/(col*8);
	glIdx -= (col*8)*str.i;
	str.j = glIdx/8;
	glIdx -= 8*str.j;
	str.index = glIdx;
	return str;
}

// convert local index to global index
//index -> (gamma, beta)
struct GBStrct{
	uint32_t g;
	uint32_t b;
};

__host__ __device__ __forceinline__ uint32_t glIdx_GB(
		uint32_t num_gamma, uint32_t num_beta,
		uint32_t g, uint32_t b){
	assert(g < num_gamma);
	assert(b < num_beta);
	return (num_beta)*g + b;
}

__host__ __device__ __forceinline__ GBStrct fromIdx_GB(
		uint32_t num_gamma, uint32_t num_beta,
		uint32_t glIdx){
	assert((glIdx < num_gamma*num_beta));
	GBStrct str;
	str.g = glIdx/(num_beta);
	glIdx -= (num_beta)*str.g;
	str.b = glIdx;
	return str;
}
// convert local index to global index
//index -> (gamma, beta, trot, row, col, inchimera=8)
struct GBTRCIStrct{
	uint32_t g;
	uint32_t b;
	uint32_t t;
	uint32_t i;
	uint32_t j;
	uint32_t index;
};

__host__ __device__ __forceinline__ uint32_t glIdx_GBTRCI(
		uint32_t num_gamma, uint32_t num_beta, uint32_t num_trot, uint32_t row, uint32_t col,
		uint32_t g, uint32_t b, uint32_t t, uint32_t i, uint32_t j, uint32_t index){
	assert(g < num_gamma);
	assert(b < num_beta);
	assert(t < num_trot);
	assert(i < row);
	assert(j < col);
	assert(index < 8);
	return 
		(num_beta*num_trot*row*col*8)*g+
		(num_trot*row*col*8)*b+
		(row*col*8)*t+
		(col*8)*i+
		(8)*j+
		index;
}

__host__ __device__ __forceinline__ GBTRCIStrct fromIdx_GBTRCI(
		uint32_t num_gamma, uint32_t num_beta, uint32_t num_trot, uint32_t row, uint32_t col,
		uint32_t glIdx){
	assert(glIdx < num_gamma*num_beta*num_trot*row*col*8);
	GBTRCIStrct str;
	str.g = glIdx/(num_beta*num_trot*row*col*8);
	glIdx -= (num_beta*num_trot*row*col*8)*str.g;
	str.b = glIdx/(num_trot*row*col*8);
	glIdx -= (num_trot*row*col*8)*str.b;
	str.t = glIdx/(row*col*8);
	glIdx -= (row*col*8)*str.t;
	str.i = glIdx/(col*8);
	glIdx -= (col*8)*str.i;
	str.j = glIdx/8;
	glIdx -= 8*str.j;
	str.index = glIdx;
	return str;
}

// without trot
// convert local index to global index
//index -> (gamma, beta, row, col, inchimera=8)
struct GBRCIStrct{
	uint32_t g;
	uint32_t b;
	uint32_t i;
	uint32_t j;
	uint32_t index;
};

__host__ __device__ __forceinline__ uint32_t glIdx_GBRCI(
		uint32_t num_gamma, uint32_t num_beta, uint32_t row, uint32_t col,
		uint32_t g, uint32_t b, uint32_t i, uint32_t j, uint32_t index){
	assert(g < num_gamma);
	assert(b < num_beta);
	assert(i < row);
	assert(j < col);
	assert(index < 8);
	return 
		(num_beta*row*col*8)*g+
		(row*col*8)*b+
		(col*8)*i+
		(8)*j+
		index;
}

__host__ __device__ __forceinline__ GBRCIStrct fromIdx_GBRCI(
		uint32_t num_gamma, uint32_t num_beta, uint32_t row, uint32_t col,
		uint32_t glIdx){
	assert(glIdx < num_gamma*num_beta*row*col*8);
	GBRCIStrct str;
	str.g = glIdx/(num_beta*row*col*8);
	glIdx -= (num_beta*row*col*8)*str.g;
	str.b = glIdx/(row*col*8);
	glIdx -= (row*col*8)*str.b;
	str.i = glIdx/(col*8);
	glIdx -= (col*8)*str.i;
	str.j = glIdx/8;
	glIdx -= 8*str.j;
	str.index = glIdx;
	return str;
}
