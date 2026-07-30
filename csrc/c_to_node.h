#ifndef C_TO_NODE_H
#define C_TO_NODE_H

#include <node_api.h>
#include <tensor.h>

napi_value create_typedarray(napi_env env,void *array,int length);

void *get_node_array(napi_env env,napi_value argv);

Tensor *get_node_tensors(napi_env env,napi_value argv);

napi_value push_node_tensor(napi_env env,Tensor *tensor,napi_value constructor);

napi_value T_add(napi_env env,napi_callback_info info);//Patameter 3 (tensor_one,tensor_two,constructor)

napi_value T_sub(napi_env env,napi_callback_info info);//Patameter 3 (tensor_one,tensor_two,constructor)

napi_value T_min(napi_env env,napi_callback_info info);//Parameter 4 (obj,axis,keepdims,constructor)

napi_value T_max(napi_env env,napi_callback_info info);//Parameter 4 (obj,axis,keepdims,constructor)

napi_value Init(napi_env env,napi_value exports);

void init_function(napi_env env,napi_value exports,napi_callback cfuntion,char *function_name);

napi_value node_reset_tensor_pool(napi_env env,napi_callback_info info);//Parameter 0

napi_value node_destory_tensor_pool(napi_env env,napi_callback_info info);//Parameter 0

napi_value node_create_tensor_pool(napi_env env,napi_callback_info info);//Parameter 0

napi_value T_div_S(napi_env env,napi_callback_info info);//Patameter 3 (tensor,scalar,constructor)

napi_value T_mul_T_ele(napi_env env,napi_callback_info info);//Parameter 3 (tensor_one,tensor_two,constructor)

napi_value T_sub_broadcasted(napi_env env,napi_callback_info info);//Parameter 3 (tensor_one,tensor_two,constructor)

napi_value T_add_broadcasted(napi_env env,napi_callback_info info);//Parameter 3 (tensor_one,tensor_two,constructor)

napi_value S_mul_T(napi_env env,napi_callback_info info);//Patameter 3 (tensor,scalar,constructor)

napi_value S_div_T(napi_env env,napi_callback_info info);//Patameter 3 (tensor,scalar,constructor)

napi_value T_div_T(napi_env env,napi_callback_info info);//Parameter 3 (tensor_one,tensor_two,constructor)

napi_value reshape(napi_env env,napi_callback_info info);//Parameter 4 (obj,new_shape,new_dim,constructor)

napi_value T_matul_T(napi_env env,napi_callback_info info);//Parameter 3 (tensor_one,tensor_two,constructor)

napi_value T_pow(napi_env env,napi_callback_info info);//Patameter 3 (tensor,scalar,constructor)

napi_value S_pow_T(napi_env env,napi_callback_info info);//Patameter 3 (tensor,scalar,constructor)

napi_value ln(napi_env env,napi_callback_info info);//Parameter 2 (tensor,constructor)

napi_value equal(napi_env env,napi_callback_info info);//Parameter 3 (tensor_one,tensor_two,constructor)

napi_value T_equal_broadcasted(napi_env env,napi_callback_info info);//Parameter 3 (tensor_one,tensor_two,constructor)

napi_value ones(napi_env env,napi_callback_info info);//Parameter 2 (tensor,constructor)

napi_value zeros(napi_env env,napi_callback_info info);//Parameter 2 (tensor,constructor)

napi_value sins(napi_env env,napi_callback_info info);//Parameter 2 (tensor,constructor)

napi_value coss(napi_env env,napi_callback_info info);//Parameter 2 (tensor,constructor)

napi_value transpose(napi_env env,napi_callback_info info);//Parameter 4 (obj,axis,constructor)

napi_value contiguous(napi_env env,napi_callback_info info);//Parameter 2 (tensor,constructor)
#endif