#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "tensor.h"
#include "cpu.h"
#include "MemoryPool.h"
MemoryPool *tensor_pool = NULL;
Tensor *create_tensor(double_t *data,int32_t *shape,size_t dimension){
    /*
    目前仅支持cpu 
    Currently only supports CPU
    */
    Tensor *tensor = NULL; 
    if(tensor_pool != NULL){
        tensor = (Tensor *)alloc_pool(tensor_pool);
    }else{
        tensor = (Tensor *)malloc(sizeof(Tensor));
    }

    if(tensor == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    tensor->data = data;
    tensor->shape = shape;
    tensor->dimension = dimension;
    
    tensor->size = 1;
    for(int32_t i = 0;i < dimension;i++){
        tensor->size *= shape[i]; 
    }
    tensor->strides = (int32_t *)malloc(dimension * sizeof(int32_t));// n * 4
    if(tensor->strides == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    int32_t stride = 1;
    for(int32_t i = dimension - 1; i >= 0;i--){
        tensor->strides[i] = stride;
    }
    return tensor;
}


void delete_tensor(Tensor *tensor){
    if(tensor != NULL){
        if(tensor_pool != NULL){
            free_pool(tensor_pool,(void *)tensor);
        }else{
            free(tensor);
        }
    }
}


Tensor *tensor_add(Tensor *tensor_one,Tensor *tensor_two){
    if(tensor_one->dimension != tensor_two->dimension){
        fprintf(stderr,"Tensors must have the same number of dimensions %d and %d for addition\n",tensor_one->dimension,tensor_two->dimension);
        exit(1);
    }
    
    int32_t ndim = tensor_one->dimension;
    int32_t *shape = (int32_t *)malloc(ndim * sizeof(int32_t));
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < ndim;i++){
        if(tensor_one->shape[i] != tensor_two->shape[i]){
            fprintf(stderr,"[ERROR]:Tensors must have the same shape[%d] and [%d] at index %d for addition\n",tensor_one->shape[i],tensor_two->shape[i],i);
            exit(1);
        }
        shape[i] = tensor_one->shape[i];
    }
    
    double_t *data = (double_t *)malloc(tensor_one->size * sizeof(double_t));
    if (data == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    cpu_tensor_add(tensor_one,tensor_two,data);
    return create_tensor(data,shape,ndim);
    
}
Tensor *tensor_min(Tensor *tensor,int32_t axis,bool keepdims){
    int32_t ndim;
    int32_t *shape;
    if(axis > tensor->dimension - 1){
        fprintf(stderr,"[ERROR]:axis argument %d must be smaller than tensor dimension %d", axis, tensor->dimension);
    }
    if (axis == -1){
        shape = (int32_t *)malloc(sizeof(int32_t));
        shape[0] = 1;
        ndim = 1; 
    }else{
        shape = (int32_t *)malloc((tensor->dimension - 1) * sizeof(int32_t));//sum后维度减1
        for(int32_t i = 0,j = 0;i < tensor->dimension;++i){
            if(i != axis){//如果i不等于轴的索引
                shape[j++] = tensor->shape[i];//新的形状则等于后一位
            }
        }
        ndim = tensor->dimension - 1; 
    }
    int32_t axis_size = 1;
    for(int32_t i = 0; i < ndim;i++){
        axis_size *= shape[i];//新形状的元素总数
    }
    
        double_t *data = (double_t *)calloc(axis_size,sizeof(double_t));
        if(data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_tensor_min(tensor,data,axis_size,shape,axis);
    if(keepdims){//保持维度不变
        if(axis == -1){
            // ndim
            ndim = tensor->dimension;
            shape = (int32_t *)malloc(tensor->dimension * sizeof(int32_t));
            for(int32_t i = 0;i < tensor->dimension;i++){
                shape[i] = 1;//[[[[max_number]]]]
            }
                
        }else{
            shape = (int32_t *)malloc(tensor->dimension * sizeof(int32_t));
            for(int32_t i = 0;i < tensor->dimension;i++){
                shape[i] = tensor->shape[i];
            }
            shape[axis] = 1;
            ndim = tensor->dimension;
        }
    }
    return create_tensor(data,shape,ndim);
    
}
Tensor *tensor_max(Tensor *tensor,int32_t axis,bool keepdims){
    int32_t ndim;
    int32_t *shape;
    if(axis > tensor->dimension - 1){
        fprintf(stderr,"[ERROR]:axis argument %d must be smaller than tensor dimension %d", axis, tensor->dimension);
    }
    if (axis == -1){
        shape = (int32_t *)malloc(sizeof(int32_t));
        shape[0] = 1;
        ndim = 1; 
    }else{
        shape = (int32_t *)malloc((tensor->dimension - 1) * sizeof(int32_t));//sum后维度减1
        for(int32_t i = 0,j = 0;i < tensor->dimension;++i){
            if(i != axis){//如果i不等于轴的索引
                shape[j++] = tensor->shape[i];//新的形状则等于后一位
            }
        }
        ndim = tensor->dimension - 1; 
    }
    int32_t axis_size = 1;
    for(int32_t i = 0; i < ndim;i++){
        axis_size *= shape[i];//新形状的元素总数
    }

        double_t *data = (double_t *)calloc(axis_size,sizeof(double_t));
        if(data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_tensor_max(tensor,data,axis_size,shape,axis);
        if(keepdims){
            if(axis == -1){
                // ndim
                
                ndim = tensor->dimension;
                shape = (int32_t *)malloc(tensor->dimension * sizeof(int32_t));
                for(int32_t i = 0;i < tensor->dimension;i++){
                    shape[i] = 1;
                }
                
            }else{
                shape = (int32_t *)malloc(tensor->dimension * sizeof(int32_t));
                for(int32_t i = 0;i < tensor->dimension;i++){
                    shape[i] = tensor->shape[i];
                }
                shape[axis] = 1;
                ndim = tensor->dimension;
            }
        }
        return create_tensor(data,shape,ndim);
    
}
Tensor *tensor_sub(Tensor *tensor_one,Tensor *tensor_two){
    if(tensor_one->dimension != tensor_two->dimension){
        fprintf(stderr,"[ERROR]:Tensors must have the same number of dimensions %d and %d for subtraction\n",tensor_one->dimension,tensor_two->dimension);
        exit(1);
    }
    
    int32_t ndim  = tensor_one->dimension;
    int32_t *shape = (int32_t *)malloc(ndim * sizeof(int32_t));
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < ndim;i++){
        if(tensor_one->shape[i] != tensor_two->shape[i]){
            fprintf(stderr,"[ERROR]:Tensors must have the same shape %d and %d at index %d for subtraction\n",tensor_one->shape[i],tensor_two->shape[i],i);
            exit(1);
        }
        shape[i] = tensor_one->shape[i];
    }
    
        double_t *data = (double_t *)malloc(tensor_one->size * sizeof(double_t));
        if(data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_tensor_sub(tensor_one,tensor_two,data);
        return create_tensor(data,shape,ndim);
    
}
Tensor *tensor_add_broadcasted(Tensor *tensor_one,Tensor *tensor_two){
    
    int32_t max_ndim = tensor_one->dimension > tensor_two->dimension ? tensor_one->dimension : tensor_two->dimension;
    int32_t *broadcasted_shape = (int32_t *)malloc(max_ndim * sizeof(int32_t));
    if(broadcasted_shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < max_ndim;i++){
        int32_t dim_one = i < tensor_one->dimension ? tensor_one->shape[tensor_one->dimension - 1 - i] : 1;
        int32_t dim_two = i < tensor_two->dimension ? tensor_two->shape[tensor_two->dimension - 1 - i] : 1;
        if(dim_one != dim_two && dim_one != 1 && dim_two != 1){
            fprintf(stderr,"[ERROR]:Shapes are not compatible for broadcasting\n");
            exit(1);
        }
        broadcasted_shape[max_ndim - 1 - i] = dim_one > dim_two ? dim_one : dim_two;
    }
    int32_t broadcasted_size = 1;
    for(int32_t i = 0; i < max_ndim;i++){
        broadcasted_size *= broadcasted_shape[i];
    }
    
        double_t *data = (double_t *)malloc(broadcasted_size * sizeof(double_t));//data
        if(data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_tensor_broadcasted_add(tensor_one,tensor_two,data,broadcasted_shape,broadcasted_size);
        return create_tensor(data,broadcasted_shape,max_ndim);
    
}
Tensor *tensor_sub_broadcasted(Tensor* tensor_one, Tensor* tensor_two){
    
    int32_t max_ndim = tensor_one->dimension > tensor_two->dimension ? tensor_one->dimension : tensor_two->dimension;
    int32_t *broadcasted_shape = (int32_t *)malloc(max_ndim * sizeof(int32_t));
    if(broadcasted_shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < max_ndim;i++){
        int32_t dim_one = i < tensor_one->dimension ? tensor_one->shape[tensor_one->dimension - 1 - i] : 1;
        int32_t dim_two = i < tensor_two->dimension ? tensor_two->shape[tensor_two->dimension - 1 - i] : 1;
        if(dim_one != dim_two && dim_one != 1 && dim_two != 1){
            fprintf(stderr,"[ERROR]:Shapes are not compatible for broadcasting\n");
            exit(1);
        }
        broadcasted_shape[max_ndim - 1 - i] = dim_one > dim_two ? dim_one : dim_two;
    }
    int32_t broadcasted_size = 1;
    for(int32_t i = 0; i < max_ndim;i++){
        broadcasted_size *= broadcasted_shape[i];
    }
    
        double_t *data = (double_t *)malloc(broadcasted_size * sizeof(double_t));
        if(data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_tensor_broadcasted_sub(tensor_one,tensor_two,data,broadcasted_shape,broadcasted_size);
        return create_tensor(data,broadcasted_shape,max_ndim);
    
}
Tensor *tensor_mul_elementwise(Tensor *tensor_one,Tensor *tensor_two){
    
    if(tensor_one->dimension != tensor_two->dimension){
        fprintf(stderr,"[ERROR]:Tensors must have the same number of dimensions %d and %d for element-wise multiplication\n",tensor_one->dimension,tensor_two->dimension);
        exit(1);
    }
    int32_t ndim = tensor_one->dimension;
    int32_t *shape = (int32_t *)malloc(ndim * sizeof(int32_t));
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < ndim;i++){
        if(tensor_one->shape[i] != tensor_two->shape[i]){
            fprintf(stderr,"[ERROR]:Tensors must have the same shape %d and %d at index %d for subtraction\n",tensor_one->shape[i],tensor_two->shape[i],i);
            exit(1);
        }
        shape[i] = tensor_one->shape[i];
    }
    
        double_t *data = (double_t *)malloc(tensor_one->size * sizeof(double_t));
        if(data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_tensor_mul_elementwise(tensor_one,tensor_two,data);
        return create_tensor(data,shape,ndim);
    
}
Tensor *scalar_mul_tensor(Tensor *tensor,double_t scalar){
    int32_t ndim = tensor->dimension;
    int32_t *shape = (int32_t *)malloc(ndim * sizeof(int32_t));
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < ndim;i++){
        shape[i] = tensor->shape[i];
    }
    
        double_t *data = (double_t *)malloc(tensor->size * sizeof(double_t));
        if(data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_scalar_mul_tensor(tensor,scalar,data);
        return create_tensor(data,shape,ndim);
    
}
Tensor *scalar_div_tensor(Tensor *tensor,double_t scalar){
    int32_t ndim = tensor->dimension;
    int32_t *shape = (int32_t *)malloc(ndim * sizeof(int32_t));
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < ndim;i++){
        shape[i] = tensor->shape[i];
    }
    
        double_t *data = (double_t *)malloc(tensor->size * sizeof(double_t));
        if(data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_scalar_div_tensor(tensor,scalar,data);
        return create_tensor(data,shape,ndim);
    
}
Tensor *tensor_div_scalar(Tensor *tensor,double_t scalar){
    int32_t ndim = tensor->dimension;
    int32_t *shape = (int32_t *)malloc(ndim * sizeof(int32_t));
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < ndim;i++){
        shape[i] = tensor->shape[i];
    }
    
        double_t *data = (double_t *)malloc(tensor->size * sizeof(double_t));
        if(data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_tensor_div_scalar(tensor,scalar,data);
        return create_tensor(data,shape,ndim);
}
Tensor *tensor_div_tensor(Tensor *tensor_one,Tensor *tensor_two){
    if(tensor_one->dimension != tensor_two->dimension){
        fprintf(stderr,"[ERROR]:Tensors must have the same number of dimensions %d and %d for element-wise division\n",tensor_one->dimension,tensor_two->dimension);
        exit(1);
    }
    
    int32_t ndim = tensor_one->dimension;
    int32_t *shape = (int32_t *)malloc(ndim * sizeof(int32_t));
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < ndim;i++){
        if(tensor_one->shape[i] != tensor_two->shape[i]){
            fprintf(stderr,"[ERROR]:Tensors must have the same shape %d and %d at index %d for division\n",tensor_one->shape[i],tensor_two->shape[i],i);
            exit(1);
        }
        shape[i] = tensor_one->shape[i];
    }
    
        double_t *data = (double_t *)malloc(tensor_one->size * sizeof(double_t));
        if(data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_tensor_div_tensor(tensor_one,tensor_two,data);
        return create_tensor(data,shape,ndim);
    
}
Tensor *tensor_matmul(Tensor *tensor_one,Tensor *tensor_two){//矩阵专用
    if(tensor_one->dimension != 2 && tensor_two->dimension != 2){
        fprintf(stderr,"[ERROR]:The input has to be a matrix (that is, a second-order tensor)\n");
        exit(1);
    }
    if(tensor_one->shape[1] != tensor_two->shape[0]){
        fprintf(stderr,"[ERROR]:Incompatible shapes for matrix multiplication %dx%d and %dx%d\n",tensor_one->shape[0],tensor_one->shape[1],tensor_two->shape[0],tensor_two->shape[1]);
        exit(1);
    }
    
    int32_t *shape = (int32_t *)malloc(2 * sizeof(int32_t));
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    shape[0] = tensor_one->shape[0];
    shape[1] = tensor_two->shape[1];
    int32_t size = shape[0] * shape[1];
    
        double_t *data = (double_t *)malloc(tensor_one->size * sizeof(double_t));
        if (data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_tensor_matmul(tensor_one,tensor_two,data);
        return create_tensor(data,shape,2);
    
}
Tensor *tensor_pow(Tensor *tensor,double_t exponent){
    int32_t ndim = tensor->dimension;
    int32_t *shape = (int32_t *)malloc(ndim * sizeof(int32_t));
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < ndim;i++){
        shape[i] = tensor->shape[i];
    }
    
        double_t *data = (double_t *)malloc(tensor->size * sizeof(double_t));
        if (data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_tensor_pow_scalar(tensor,exponent,data);
        return create_tensor(data,shape,ndim);
    
}
Tensor *scalar_pow_tensor(double_t base,Tensor *tensor){
    int32_t ndim = tensor->dimension;
    int32_t *shape = (int32_t *)malloc(ndim * sizeof(int32_t));
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < ndim;i++){
        shape[i] = tensor->shape[i];
    }
    
        double_t *data = (double_t *)malloc(tensor->size * sizeof(double_t));
        if (data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_scalar_pow_tensor(base,tensor,data);
        return create_tensor(data,shape,ndim);
    
}
Tensor *tensor_ln(Tensor *tensor){
    int32_t ndim = tensor->dimension;
    int32_t *shape = (int32_t *)malloc(ndim * sizeof(int32_t));
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < ndim;i++){
        shape[i] = tensor->shape[i];
    }
    
        double_t *data = (double_t *)malloc(tensor->size * sizeof(double_t));
        if (data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_tensor_ln(tensor,data);
        return create_tensor(data,shape,tensor->dimension);
    
}
Tensor *tensor_equal(Tensor *tensor_one,Tensor *tensor_two){
    if(tensor_one->dimension != tensor_two->dimension){
        fprintf(stderr,"Tensors must have the same number of dimensions %d and %d for addition\n",tensor_one->dimension,tensor_two->dimension);
        exit(1);
    }
    
    int32_t ndim = tensor_one->dimension;
    int32_t *shape = (int32_t *)malloc(ndim * sizeof(int32_t));
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < ndim;i++){
        if(tensor_one->shape[i] != tensor_two->shape[i]){
            fprintf(stderr,"[ERROR]:Tensors must have the same shape[%d] and [%d] at index %d for addition\n",tensor_one->shape[i],tensor_two->shape[i],i);
            exit(1);
        }
        shape[i] = tensor_one->shape[i];
    }
   
    double_t *data = (double_t *)malloc(tensor_one->size * sizeof(double_t));
    if (data == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    cpu_tensor_equal(tensor_one,tensor_two,data);
    return create_tensor(data,shape,ndim);
   
}
Tensor *tensor_equal_broadcasted(Tensor *tensor_one,Tensor *tensor_two){
    
    int32_t max_ndim = tensor_one->dimension > tensor_two->dimension ? tensor_one->dimension : tensor_two->dimension;
    int32_t *broadcasted_shape = (int32_t *)malloc(max_ndim * sizeof(int32_t));
    if(broadcasted_shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < max_ndim;i++){
        int32_t dim_one = i < tensor_one->dimension ? tensor_one->shape[tensor_one->dimension - 1 - i] : 1;
        int32_t dim_two = i < tensor_two->dimension ? tensor_two->shape[tensor_two->dimension - 1 - i] : 1;
        if(dim_one != dim_two && dim_one != 1 && dim_two != 1){
            fprintf(stderr,"[ERROR]:Shapes are not compatible for broadcasting\n");
            exit(1);
        }
        broadcasted_shape[max_ndim - 1 - i] = dim_one > dim_two ? dim_one : dim_two;
    }
    int32_t broadcasted_size = 1;
    for(int32_t i = 0; i < max_ndim;i++){
        broadcasted_size *= broadcasted_shape[i];
    }
   
        double_t *data = (double_t *)malloc(broadcasted_size * sizeof(double_t));//data
        if(data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_tensor_broadcasted_equal(tensor_one,tensor_two,data,broadcasted_shape,broadcasted_size);
        return create_tensor(data,broadcasted_shape,max_ndim);
    
}
Tensor *tensor_reshape(Tensor *tensor,int32_t *new_shape,int32_t new_ndim){
    int32_t size = 1;
    int32_t *shape= (int32_t *)malloc(tensor->dimension * sizeof(int32_t));
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < new_ndim;i++){
        size *= new_shape[i];
    }
    if(size != tensor->size){
        fprintf(stderr,"[ERROR]:Cannot reshape tensor. Total number of elements in new shape does not match the current size of the tensor.\n");
        exit(1);
    }
    
    double_t *data = (double_t *)malloc(tensor->size * sizeof(double_t));
    if(data == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    cpu_tensor_assign(tensor,data);
    return create_tensor(data,new_shape,new_ndim);
    
}
Tensor *ones_like(Tensor *tensor){
    int32_t * shape= (int32_t *)malloc(tensor->dimension * sizeof(int32_t));
    int ndim = tensor->dimension;
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < tensor->dimension;i++){
        shape[i] = tensor->shape[i];
    }
    
        double_t *data = (double_t *)malloc(tensor->size * sizeof(double_t));
        if(data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_tensor_ones_like(tensor,data);
        return create_tensor(data,shape,ndim);
    
}
Tensor *zeros_like(Tensor *tensor){
    int32_t *shape= (int32_t *)malloc(tensor->dimension * sizeof(int32_t));
    int32_t ndim = tensor->dimension;
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < tensor->dimension;i++){
        shape[i] = tensor->shape[i];
    }
    
        double_t *data = (double_t *)malloc(tensor->size * sizeof(double_t));
        if(data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_tensor_zeros_like(tensor,data);
        return create_tensor(data,shape,ndim);
   
}
Tensor *tensor_sin(Tensor *tensor){
    int32_t ndim = tensor->dimension;
    int32_t *shape= (int32_t *)malloc(ndim * sizeof(int32_t));
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < ndim;i++){
        shape[i] = tensor->shape[i];
    }
    
        double_t *data = (double_t *)malloc(tensor->size * sizeof(double_t));
        if (data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_tensor_sin(tensor,data);
        return create_tensor(data,shape,ndim);
    
}
Tensor *tensor_cos(Tensor *tensor){
    int32_t ndim = tensor->dimension;
    int32_t * shape= (int32_t *)malloc(ndim * sizeof(int32_t));
    if(shape == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < ndim;i++){
        shape[i] = tensor->shape[i];
    }
    
        double_t *data = (double_t *)malloc(tensor->size * sizeof(double_t));
        if (data == NULL){
            fprintf(stderr,"[ERROR]:Memory allocation failed\n");
            exit(1);
        }
        cpu_tensor_cos(tensor,data);
        return create_tensor(data,shape,ndim);
   
}
Tensor *tensor_axes_transpose(Tensor *tensor, int32_t *axis,int32_t axis_length){
    int32_t *shape_transpose = (int32_t *)malloc(tensor->dimension * sizeof(int32_t));
    if(shape_transpose == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    for(int32_t i = 0;i < tensor->dimension;i++){
        shape_transpose[i] = tensor->shape[axis[i]];//转换后轴的顺序
    }
    int32_t ndim = tensor->dimension;
    if(axis_length != tensor->dimension){
        fprintf(stderr,"[ERROR]:You need to keep the axis length the same as the tensor shape length\n");
        exit(1);
    }
    double_t *data = (double_t *)malloc(tensor->size * sizeof(double_t));
    if(data == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    cpu_tensor_transpose_axes(tensor,data,axis,shape_transpose);
    return create_tensor(data,shape_transpose,ndim);
    
}
void make_contiguous(Tensor *tensor){
    int32_t *new_strides = (int32_t *)malloc(tensor->dimension * sizeof(int32_t));
    if(new_strides == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    int32_t *new_shape = (int32_t *)malloc(tensor->dimension * sizeof(int32_t));

    int32_t stride = 1;
    for(int32_t i = 0;i < tensor->dimension;i++){
        new_shape[i] = tensor->shape[i];
        new_strides[i] = stride;
        stride *= tensor->shape[i];
    }

    double_t *data = (double_t *)malloc(tensor->size * sizeof(double_t));
    if(data == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    cpu_tensor_make_contiguous(tensor,data,new_strides);
}
Tensor *tensor_sum(Tensor *tensor,int32_t axis,bool keepdims){
    int32_t ndim;
    int32_t *shape;
    if(axis > tensor->dimension - 1){
        fprintf(stderr,"[ERROR]:axis argument %d must be smaller than tensor dimension %d", axis, tensor->dimension);
    }
    if (axis == -1){
        shape = (int32_t *)malloc(sizeof(int32_t));
        shape[0] = 1;
        ndim = 1; 
    }else{
        shape = (int32_t *)malloc((tensor->dimension - 1) * sizeof(int32_t));//sum后维度减1
        for(int32_t i = 0,j = 0;i < tensor->dimension;++i){
            if(i != axis){//如果i不等于轴的索引
                shape[j++] = tensor->shape[i];//新的形状则等于后一位
            }
        }
        ndim = tensor->dimension - 1; 
    }
    int32_t axis_size = 1;
    for(int32_t i = 0; i < ndim;i++){
        axis_size *= shape[i];//新形状的元素总数
    }

    double_t *data = (double_t *)calloc(axis_size,sizeof(double_t));
    if(data == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    cpu_tensor_sum(tensor,data,axis_size,shape,axis);
    if(keepdims){
        if(axis == -1){
                // ndim
                
            ndim = tensor->dimension;
            shape = (int32_t *)malloc(tensor->dimension * sizeof(int32_t));
            for(int32_t i = 0;i < tensor->dimension;i++){
                shape[i] = 1;
            }
                
        }else{
            shape = (int32_t *)malloc(tensor->dimension * sizeof(int32_t));
            for(int32_t i = 0;i < tensor->dimension;i++){
                shape[i] = tensor->shape[i];
            }
            shape[axis] = 1;
            ndim = tensor->dimension;
        }
    }
    return create_tensor(data,shape,ndim);
    
}
// Tensor *tensor_matmul_batched_broadcasted(Tensor *tensor_one,Tensor *tensor_two){
//     if(tensor_one->shape[1] != tensor_two->shape[1]){
         
//     }
// }
// Tensor *tensor_matmul_batched(Tensor *tensor_one,Tensor *tensor_two){

// }
