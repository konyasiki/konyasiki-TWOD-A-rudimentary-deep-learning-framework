#ifndef CPU_H
#define CPU_H
#include "tensor.h"
// #include "MemoryPool.h"
void cpu_tensor_add(Tensor *tensor_one,Tensor *tensor_two,double_t *data);//

void cpu_tensor_broadcasted_add(Tensor *tensor_one,Tensor *tensor_two,double_t *data,int *broadcasted_shape,int broadcasted_size);//

void cpu_tensor_sum(Tensor *tensor,double_t *data,int size,int *shape,int axis);//

void cpu_tensor_max(Tensor *tensor,double_t *data,int size,int *resule_shape,int axis);//

void cpu_tensor_min(Tensor *tensor,double_t *data,int size,int *resule_shape,int axis);//

void cpu_tensor_sub(Tensor *tensor_one,Tensor *tensor_two,double_t *data);//

void cpu_tensor_broadcasted_sub(Tensor *tensor_one,Tensor *tensor_two,double_t *data,int *broadcasted_shape,int broadcasted_size);//

void cpu_tensor_mul_elementwise(Tensor *tensor_one,Tensor *tensor_two,double_t *data);//

void cpu_scalar_div_tensor(Tensor *tensor,double_t scalar,double_t *data);//

void cpu_tensor_div_scalar(Tensor *tensor,double_t scalar,double_t *data);//

void cpu_tensor_div_tensor(Tensor *tensor_one,Tensor *tensor_two,double_t *data);//

void cpu_tensor_matmul(Tensor *tensor_one,Tensor *tensor_two,double_t *data);//

void cpu_tensor_broadcasted_batched_matmul(Tensor *tensor_one,Tensor *tensor_two,double_t *data);//

void cpu_tensor_matmul_batched(Tensor *tensor_one,Tensor *tensor_two,double_t *data);//

void cpu_scalar_pow_tensor(double_t base,Tensor *tensor,double_t *data);//

void cpu_tensor_pow_scalar(Tensor *tensor,double_t exponent,double_t *data);//

void cpu_tensor_ln(Tensor *tensor,double_t *data);//

void cpu_scalar_mul_tensor(Tensor *tensor,double_t scalar,double_t *data);//

void cpu_tensor_equal(Tensor *tensor_one,Tensor *tensor_two,double_t *data);//

void cpu_tensor_broadcasted_equal(Tensor *tensor_one,Tensor *tensor_two,double_t *data,int *broadcasted_shape,int broadcasted_size);//

void cpu_tensor_ones_like(Tensor *tensor,double_t *data);//

void cpu_tensor_zeros_like(Tensor *tensor,double_t *data);//

void cpu_tensor_transpose1D(Tensor *tensor,double_t *data);//

void cpu_tensor_transpose2D(Tensor *tensor,double_t *data);//

void cpu_tensor_transpose3D(Tensor *tensor,double_t *data);//

void cpu_tensor_transpose_axes(Tensor *tensor,double_t *data,int *axis,int *shape_transpose);//

void cpu_tensor_assign(Tensor *tensor,double_t *data);
//复制张量使用同一地址

void cpu_tensor_make_contiguous(Tensor *tensor,double_t *data,int *new_strides);
//复制一份但是不储存在同一地址

void cpu_tensor_sin(Tensor *tensor,double_t *data);//

void cpu_tensor_cos(Tensor *tensor,double_t *data);//

#endif