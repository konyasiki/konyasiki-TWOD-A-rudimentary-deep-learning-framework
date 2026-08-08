#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <node_api.h>
#include "c_to_node.h"

void cpu_tensor_add(Tensor *tensor_one,Tensor *tensor_two,double_t *data){
    for(int32_t i = 0;i < tensor_one->size;i++){
        data[i] = tensor_one->data[i] + tensor_two->data[i];
    }
}
void cpu_tensor_broadcasted_add(Tensor *tensor_one,Tensor *tensor_two,double_t *data,int *broadcasted_shape,int broadcasted_size){
    int32_t max_ndim = tensor_one->dimension > tensor_two->dimension ? tensor_one->dimension : tensor_two->dimension;
    int32_t *strides_one = (int32_t *)malloc(max_ndim * sizeof(int32_t));//max_ndim * sizeof(int32_t)
    int32_t *strides_two = (int32_t *)malloc(max_ndim * sizeof(int32_t));//max_ndim * sizeof(int32_t)
    if(strides_one == NULL || strides_two == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    int32_t stride_one = 1;
    int32_t stride_two = 1;
    for(int32_t i = max_ndim - 1;i >= 0;i--){
        int32_t dim_one = tensor_one->dimension - max_ndim + i >= 0 ? tensor_one->shape[tensor_one->dimension - max_ndim + i] : 1;
        int32_t dim_two = tensor_two->dimension - max_ndim + i >= 0 ? tensor_two->shape[tensor_two->dimension - max_ndim + i] : 1;
        strides_one[i] = dim_one == broadcasted_shape[i] ? stride_one : 0;
        strides_two[i] = dim_two == broadcasted_shape[i] ? stride_two : 0;
        stride_one *= (dim_one == broadcasted_shape[i] ? dim_one : 1);
        stride_two *= (dim_two == broadcasted_shape[i] ? dim_two : 1);
    }
    for(int32_t i = 0;i < broadcasted_size;i++){
        int32_t index_one = 0;
        int32_t index_two = 0;
        int32_t linear_index = i;
        for(int32_t j = max_ndim - 1;j >=0;j--){
            int32_t pos = linear_index % broadcasted_shape[j];
            linear_index /= broadcasted_shape[j];
            if(strides_one[j] != 0) index_one += pos * strides_one[j];
            if(strides_two[j] != 0) index_two += pos * strides_two[j];
        }
        data[i] = tensor_one->data[index_one] + tensor_two->data[index_two];
    }
    free(strides_one);
    free(strides_two);
}
void cpu_tensor_sub(Tensor *tensor_one,Tensor *tensor_two,double_t *data){
    for(int32_t i = 0;i < tensor_one->size;i++){
        data[i] = tensor_one->data[i] - tensor_two->data[i];
    }
}
void cpu_tensor_broadcasted_sub(Tensor *tensor_one,Tensor *tensor_two,double_t *data,int32_t *broadcasted_shape,int32_t broadcasted_size){
    int32_t max_ndim = tensor_one->dimension > tensor_two->dimension ? tensor_one->dimension : tensor_two->dimension;
    int32_t *strides_one = (int32_t*)malloc(max_ndim * sizeof(int32_t));//max_ndim * sizeof(int32_t)
    int32_t *strides_two = (int32_t*)malloc(max_ndim * sizeof(int32_t));//max_ndim * sizeof(int32_t)
    if(strides_one == NULL || strides_two == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    int32_t stride_one = 1;
    int32_t stride_two = 1;
    for(int32_t i = max_ndim - 1;i >= 0;i--){
        int32_t dim_one = tensor_one->dimension - max_ndim + i >= 0 ? tensor_one->shape[tensor_one->dimension - max_ndim + i] : 1;
        int32_t dim_two = tensor_two->dimension - max_ndim + i >= 0 ? tensor_two->shape[tensor_two->dimension - max_ndim + i] : 1;
        strides_one[i] = dim_one == broadcasted_shape[i] ? stride_one : 0;
        strides_two[i] = dim_two == broadcasted_shape[i] ? stride_two : 0;
        stride_one *= (dim_one == broadcasted_shape[i] ? dim_one : 1);
        stride_two *= (dim_two == broadcasted_shape[i] ? dim_two : 1);
    }
    for(int32_t i = 0;i < broadcasted_size;i++){
        int32_t index_one = 0;
        int32_t index_two = 0;
        int32_t linear_index = i;
        for(int32_t j = max_ndim - 1;j >=0;j--){
            int32_t pos = linear_index % broadcasted_shape[j];
            linear_index /= broadcasted_shape[j];
            if(strides_one[j] != 0) index_one += pos * strides_one[j];
            if(strides_two[j] != 0) index_two += pos * strides_two[j];
        }
        data[i] = tensor_one->data[index_one] - tensor_two->data[index_two];
    }
    free(strides_one);
    free(strides_two);
}
void cpu_tensor_mul_elementwise(Tensor *tensor_one,Tensor *tensor_two,double_t *data){//张量
    for(int32_t i = 0;i < tensor_one->size ;i++){
        data[i] = tensor_one->data[i] * tensor_two->data[i];
    }
}
void cpu_scalar_mul_tensor(Tensor *tensor,double_t scalar,double_t *data){
    for(int32_t i = 0;i < tensor->size;i++){
        data[i] = tensor->data[i] * scalar;
    }
}
void cpu_scalar_div_tensor(Tensor *tensor,double_t scalar,double_t *data){
    for (int32_t i = 0; i < tensor->size; i++){
        data[i] = scalar / tensor->data[i];
    }
}
void cpu_tensor_div_scalar(Tensor *tensor,double_t scalar,double_t *data){
    for(int32_t i = 0;i < tensor->size;i++){
        data[i] = tensor->data[i] / scalar;
    }
}
void cpu_tensor_div_tensor(Tensor *tensor_one,Tensor *tensor_two,double_t *data){
    for(int32_t i = 0;i < tensor_one->size;i++){
        data[i] = tensor_one->data[i] / tensor_two->data[i];
    }
}
void cpu_tensor_matmul(Tensor *tensor_one,Tensor *tensor_two,double_t *data){
    for(int32_t i = 0;i < tensor_one->shape[0];i++){
        for(int32_t j = 0;j < tensor_two->shape[1];j++){
            double_t sum = 0.0;
            for(int32_t k = 0;k < tensor_one->shape[1];k++){
                sum += tensor_one->data[i * tensor_one->shape[1] + k] * tensor_two->data[tensor_two->shape[1] + j];
            }
            data[i * tensor_two->shape[1] + j] = sum;
        }
    }
}
void cpu_tensor_broadcasted_batched_matmul(Tensor *tensor_one,Tensor *tensor_two,double_t *data){
    int32_t data_stride = tensor_one->shape[0] * tensor_two->shape[2];
    for(int32_t batch = 0;batch < tensor_two->shape[0];batch++){
        for(int32_t i = 0;i < tensor_two->shape[0];i++){
            for(int32_t j = 0;j < tensor_two->shape[2];j++){
                double_t sum = 0.0;
                for(int32_t k = 0;k < tensor_one->shape[1];k++){
                    sum += tensor_one->data[i * tensor_one->shape[1] + k] * tensor_two->data[batch * tensor_two->strides[0] + (k * tensor_two->shape[2] + j)];
                }
                data[(batch * data_stride) + (i * tensor_two->shape[2] + j)] = sum;
            }
        }
    }
}
void cpu_tensor_matmul_batched(Tensor *tensor_one,Tensor *tensor_two,double_t *data){
    int32_t data_stride = tensor_one->shape[1] * tensor_two->shape[2];
    for(int32_t batch = 0;batch < tensor_two->shape[0];batch++){
        for(int32_t i = 0;i < tensor_one->shape[1];i++){
            for(int32_t j = 0;j <tensor_two->shape[2];j++){
                double_t sum = 0.0;
                for(int32_t k = 0;k <tensor_one->shape[2];k++){
                    sum += tensor_one->data[(batch * tensor_one->strides[0]) + i * tensor_one->shape[2] + k] * tensor_two->data[batch * tensor_two->strides[0] + (k * tensor_two->shape[2] + j)];
                }
                data[(batch * data_stride) + (i * tensor_two->shape[2] + j)] = sum;
            }
        }
    }
}
void cpu_scalar_pow_tensor(double_t base,Tensor *tensor,double_t *data){
    for(int32_t i = 0;i < tensor->size; i++){
        data[i] = powf(base,tensor->data[i]);
    }
}
void cpu_tensor_pow_scalar(Tensor *tensor,double_t exponent,double_t *data){
    for(int32_t i = 0;i < tensor->size;i++){
        data[i] = powf(tensor->data[i],exponent);
    }
}
void cpu_tensor_ln(Tensor *tensor,double_t *data){
    for(int32_t i = 0;i <tensor->size;i++){
        data[i] = logf(tensor->data[i]);
    }
}
void cpu_tensor_sum(Tensor *tensor,double_t *data,int32_t size,int32_t *shape,int32_t axis){
    if(axis == -1){
        //sum all data
        double_t sum = 0.0;
        for(int32_t i = 0;i < tensor->size;i++){
            sum += tensor->data[i];
        }
        *data = sum;
    }else{
        if(axis < 0 || axis >= tensor->dimension){
            printf("Invalid axis");
            return;
        }
        int32_t axis_stride = tensor->strides[axis];//需要相加的轴的步长
        for(int32_t i = 0;i < tensor->shape[axis];i++){
            for(int32_t j = 0;j < size;j++){
                int32_t index = 0;
                int32_t remainder = j;
                for(int32_t k = tensor->dimension - 2;k >= 0;k--){//sum的形状是dim - 1 但是索引是减2
                    index +=(remainder % shape[k]) * tensor->strides[k < axis ? k : k + 1];
                    remainder /= shape[k];
                }
                data[j] += tensor->data[index + i * axis_stride];
            }
        }
    }
}
void cpu_tensor_max(Tensor *tensor,double_t *data,int32_t size,int32_t *result_shape,int32_t axis){
    if(axis == -1){
        double_t max_value = -INFINITY;//负无穷大
        for(int32_t i = 0;i < tensor->size;i++){
            max_value = fmax(max_value,tensor->data[i]);
        }
        *data = max_value;//所有中的最大
    }else{
        for(int32_t i = 0;i < size;i++){
            data[i] = -INFINITY;
        }
        if(axis < 0 || axis >= tensor->dimension){
            printf("Invalid axis");
            return;
        }
        int32_t axis_stride = tensor->strides[axis];
        for(int32_t i = 0; i < tensor->shape[axis];i++){
            for(int32_t j = 0;j < size;j++){
                int32_t index = 0;
                int32_t remainder = 0;
                for(int32_t k = tensor->dimension - 2;k >= 0; k--){
                    index +=(remainder % result_shape[k]) * tensor->strides[k < axis ? k : k + 1];
                    remainder /= result_shape[k];
                }
                data[j] = fmax(data[j],tensor->data[index + i * axis_stride]);
            }
        }
    }
}
void cpu_tensor_min(Tensor *tensor,double_t *data,int32_t size,int32_t *result_shape,int32_t axis){
    if(axis == -1){
        double_t min_value = INFINITY;
        for(int32_t i = 0;i < tensor->size;i++){
            min_value = fmin(min_value,tensor->data[i]);
        }
        *data = min_value;
    }else{
        for(int32_t i = 0;i < size;i++){
            data[i] = -INFINITY;
        }
        if(axis < 0 || axis >= tensor->dimension){
            printf("Invalid axis");
            return;
        }
        int32_t axis_stride = tensor->strides[axis];
        for(int32_t i = 0; i < tensor->shape[axis];i++){
            for(int32_t j = 0;j < size;j++){
                int32_t index = 0;
                int32_t remainder = 0;
                for(int32_t k = tensor->dimension - 2;k >= 0; k--){
                    index +=(remainder % result_shape[k]) * tensor->strides[k < axis ? k : k + 1];
                    remainder /= result_shape[k];
                }
                data[j] = fmin(data[j],tensor->data[index + i * axis_stride]);
            }
        }
    }
}
void cpu_tensor_equal(Tensor *tensor_one,Tensor *tensor_two,double_t *data){
    for(int32_t i = 0; i < tensor_one->size;i++){
        data[i] = (tensor_one->data[i] == tensor_two->data[i]) ? 1.0f : 0.0f;
    }
}
void cpu_tensor_broadcasted_equal(Tensor *tensor_one,Tensor *tensor_two,double_t *data,int32_t *broadcasted_shape,int32_t broadcasted_size){
    int32_t max_ndim = tensor_one->dimension > tensor_two->dimension ? tensor_one->dimension : tensor_two->dimension;
    int32_t *strides_one = (int32_t*)malloc(max_ndim * sizeof(int32_t));
    int32_t *strides_two = (int32_t*)malloc(max_ndim * sizeof(int32_t));
    if(strides_one == NULL || strides_two == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    int32_t stride_one = 1;
    int32_t stride_two = 1;
    for(int32_t i = max_ndim - 1; i >= 0;i--){
        int32_t dim_one = tensor_one->dimension - max_ndim + i >= 0 ? tensor_one->shape[tensor_one->dimension - max_ndim + i] : 1;
        int32_t dim_two = tensor_two->dimension - max_ndim + i >= 0 ? tensor_two->shape[tensor_two->dimension - max_ndim + i] : 1;
        strides_one[i] = dim_one == broadcasted_shape[i] ? stride_one : 0;
        strides_two[i] = dim_two == broadcasted_shape[i] ? stride_two : 0;
        stride_one *= (dim_one == broadcasted_shape[i]) ? dim_one : 1;
        stride_two *= (dim_two == broadcasted_shape[i]) ? dim_two : 1;
    }
    for(int32_t i = 0;i < broadcasted_size;i++){
        int32_t index_one = 0;
        int32_t index_two = 0;
        int32_t linear_index = i;
        for(int32_t j = max_ndim - 1;j >= 0 ;j--){
            int32_t pos = linear_index % broadcasted_shape[j];
            linear_index /= broadcasted_shape[j];
            if(strides_one[j] != 0) index_one += pos * strides_one[j];
            if(strides_two[j] != 0) index_two += pos * strides_two[j];
        }
        data[i] = (tensor_one->data[index_one] == tensor_two->data[index_two]) ? 1.0f : 0.0f;
    }
    free(strides_one);
    free(strides_two);
}
void cpu_tensor_ones_like(Tensor *tensor,double_t *data){
    for(int32_t i = 0; i < tensor->size ; i++){
        data[i] = 1.0;
    }
}
void cpu_tensor_zeros_like(Tensor *tensor,double_t *data){
    for(int32_t i = 0; i < tensor->size ; i++){
        data[i] = 0.0;
    }
}
void cpu_tensor_cos(Tensor *tensor,double_t *data){
    for(int32_t i = 0 ; i < tensor->size ;i++){
        data[i] = cosf(tensor->data[i]);
    }
}
void cpu_tensor_sin(Tensor *tensor,double_t *data){
    for(int32_t i = 0 ; i < tensor->size ;i++){
        data[i] = sinf(tensor->data[i]);
    }
}
void cpu_tensor_make_contiguous(Tensor *tensor,double_t *data,int32_t *new_strides){//复制一份但是不储存在同一地址
    for(int32_t i = 0;i < tensor->size ;i++){
        int32_t index = 0;
        int32_t offset = i;
        for(int32_t j = 0; j < tensor->dimension ;j++){
            index += (offset / new_strides[j]) * tensor->strides[j];
            offset %= new_strides[j];
        }
    }
    
    free(tensor->data);
    free(tensor->strides);
    tensor->data = data;
    tensor->strides = new_strides;
}
void cpu_tensor_transpose_axes(Tensor *tensor,double_t *data,int32_t *axis,int32_t *shape_transpose){
    int32_t *strides = (int32_t *)malloc(tensor->dimension * sizeof(int32_t));
    if(strides == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    int32_t stride = 1;
    for(int32_t i = tensor->dimension - 1; i >= 0 ;i--){
        strides[i] = stride;
        stride *= shape_transpose[i];//已转换的步长
    }
    for(int32_t i = 0;i < tensor->size;i++){
        int32_t linear_index = i;
        int32_t index = 0;
        for(int32_t j = tensor->dimension - 1;j >=0 ;j--){
            fprintf(stdout,"%d\n",axis[j]);
            int32_t pos = linear_index % tensor->shape[j];//
            linear_index /= tensor->shape[j];//每个维度对应走几步
            index += pos * strides[axis[j]];
        }
        data[index] = tensor->data[i];
    }
    free(strides);
}
void cpu_tensor_assign(Tensor *tensor,double_t *data){//复制张量
    for(int32_t i = 0;i < tensor->size;i++){
        data[i] = tensor->data[i];
    }
}