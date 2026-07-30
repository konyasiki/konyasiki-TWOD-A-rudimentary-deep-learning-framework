#ifndef TENSOR_H
#define TENSOR_H
/*
目前仅支持cpu 
Currently only supports CPU
*/
#include <stdbool.h>
#include <node_api.h>
#include "MemoryPool.h"
#include <math.h>
typedef struct Tensor{
    double_t *data;//8byte
    int *shape;
    int *strides;//步幅
    int size;//总数
    int dimension;
}Tensor;

Tensor *create_tensor(double_t *data,int *shape,size_t dimension);
// 
void delete_tensor(Tensor *tensor);
//
void delete_shape(Tensor *tensor);
//
void delete_strides(Tensor *tensor);
//
void delete_data(Tensor *tensor);
//
//
// double_t get_item(Tensor *tensor,int *indices);
//
Tensor *tensor_add(Tensor *tensor_one,Tensor *tensor_two);
// √
Tensor *tensor_sum(Tensor *tensor_one,Tensor *tensor_two);

Tensor *tensor_min(Tensor *tensor,int axis,bool keepdims);
// √
Tensor *tensor_max(Tensor *tensor,int axis,bool keepdims);
// √
Tensor *tensor_sub(Tensor *tensor_one,Tensor *tensor_two);
// √
Tensor *tensor_div_scalar(Tensor *tensor,double_t scalar);
//
Tensor *tensor_mul_elementwise(Tensor *tensor_one,Tensor *tensor_two);
// 普通乘法(两个张量对应元素相乘)

Tensor *tensor_sub_broadcasted(Tensor* tensor_one,Tensor* tensor_two);
//
Tensor *tensor_add_broadcasted(Tensor *tensor_one,Tensor *tensor_two);
//
Tensor *scalar_mul_tensor(Tensor *tensor,double_t scalar);
// 普通乘法

Tensor *scalar_div_tensor(Tensor *tensor,double_t scalar);
//
Tensor *tensor_div_tensor(Tensor *tensor_one,Tensor *tensor_two);
//
Tensor *tensor_reshape(Tensor *tensor,int *new_shape,int new_ndim);
//
Tensor *tensor_matmul_batched_broadcasted(Tensor *tensor_one,Tensor *tensor_two);//哈达玛积
Tensor *tensor_matmul_batched(Tensor *tensor_one,Tensor *tensor_two);//
//以上两个还没封装
Tensor *tensor_matmul(Tensor *tensor_one,Tensor *tensor_two);
// 矩阵专用

Tensor *tensor_pow(Tensor *tensor,double_t exponent);
//
Tensor *scalar_pow_tensor(double_t base,Tensor *tensor);
//
Tensor *tensor_ln(Tensor *tensor);
//
Tensor *tensor_equal(Tensor *tensor_one,Tensor *tensor_two);
//
Tensor *tensor_equal_broadcasted(Tensor *tensor_one,Tensor *tensor_two);
//
void to_device(Tensor *tensor,char *device);//Currently, there are no plans to implement it.
Tensor *ones_like(Tensor *tensor);
//
Tensor *zeros_like(Tensor *tensor);
//
Tensor *tensor_sin(Tensor *tensor);
//
Tensor *tensor_cos(Tensor *tensor);
//
Tensor *tensor_axes_transpose(Tensor *tensor, int *axis,int axis_length);
//
void make_contiguous(Tensor *tensor);
//
#endif 