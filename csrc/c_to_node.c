#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "tensor.h"
#include "node_api.h"
#include "MemoryPool.h"


typedef struct array_with_length{
    void *data;
    size_t length;
}awl;


void free_c_variable(node_api_basic_env env,void* finalize_data,void* finalize_hint){
    free(finalize_data);
    finalize_data = NULL;
    finalize_hint = NULL;
}

void catchError(napi_env env){
    napi_extended_error_info *ErrorInfo;
    napi_get_last_error_info(env,&ErrorInfo);
    fprintf(stderr,"[ERROR]:%s\n",ErrorInfo->error_message);
}

napi_value create_typedarray(napi_env env,void *data,uint32_t length,napi_typedarray_type type){
    napi_status status;
    size_t byte_legth;//in

    switch (type){
        case napi_float64_array:
            byte_legth = sizeof(double_t) * length;
            break;
        case napi_int32_array:
            byte_legth = sizeof(int32_t) * length;
            break;
        default:
            fprintf(stderr,"[ERROR]:There is no such type\n");
            goto send_err;
    }

    napi_value arraybuffer;//out
    napi_value typedarray;

    status = napi_create_external_arraybuffer(env,data,byte_legth,&free_c_variable,data,&arraybuffer);
    if(status != napi_ok){
        fprintf(stderr,"[ERROR]:Cannot create arraybuffer\n");
        goto send_err;
    }

    status = napi_create_typedarray(env,type,length,arraybuffer,0,&typedarray);
    if(status != napi_ok){
        fprintf(stderr,"[ERROR]:Cannot create arraybuffer\n");
        goto send_err;
    }

    return typedarray;
    send_err:
        fprintf(stderr,"[ERROR]:Fail to call create_typearray\n");
        exit(1);
}

awl get_node_array(napi_env env,napi_value argv,bool return_length){
    void *data;
    size_t byte_length;
    size_t offest,length;//偏移,数组长度
    napi_status status;

    napi_value input_array = argv;
    napi_typedarray_type type;//参数类型
    napi_value input_buffer = NULL;//元数据
    awl node_Array;

    status = napi_get_typedarray_info(env,input_array,&type,&length,NULL,&input_buffer,&offest);
    if(status != napi_ok){
        goto send_err;
    }
    status = napi_get_arraybuffer_info(env,input_buffer,&data,&byte_length);
    if(status != napi_ok){
        goto send_err;
    }

    node_Array.data = data;
    node_Array.length = 0;
    if(return_length){
        node_Array.length = length;
        return node_Array;
    }
    return node_Array;

    send_err:
        node_Array .data = NULL;
        node_Array.length = -1;
        return node_Array;
}

Tensor *get_node_tensors(napi_env env,napi_value argv){
    napi_handle_scope scope;
    napi_open_handle_scope(env,&scope);

    napi_value Typed_data;
    napi_status proper_data = napi_get_named_property(env,argv,"data",&Typed_data);
    napi_value shape;
    napi_status proper_shape = napi_get_named_property(env,argv,"shape",&shape);
    if(proper_shape != napi_ok){
        fprintf(stderr,"[ERROR]:Cannot get the shape property\n");
        exit(1);
    }
    if(proper_data != napi_ok){
        fprintf(stderr,"[ERROR]:Cannot get the data property\n");
        exit(1);
    }
    //Array
    awl data = get_node_array(env,Typed_data,false);
    //shape
    awl shape_data = get_node_array(env,shape,true);

    Tensor *tensor = create_tensor((double_t *)data.data,(int32_t *)shape_data.data,shape_data.length);
    
    napi_close_handle_scope(env,scope);
    return tensor;
}

//con_argc 参数数量
//con_argv 构造函数参数
//constructor //构造函数
napi_value push_node_tensor(napi_env env,Tensor *tensor,napi_value constructor){
    napi_value tensor_obj;

    napi_value init_tensor[2];

    napi_value data = create_typedarray(env,(void *)tensor->data,tensor->size,napi_float64_array);
    napi_value shape = create_typedarray(env,(void *)tensor->shape,tensor->dimension,napi_int32_array);
    init_tensor[0] = data;
    init_tensor[1] = shape;

    napi_status instance_status = napi_new_instance(env,constructor,2,init_tensor,&tensor_obj);
    if(instance_status != napi_ok){
        fprintf(stderr,"[ERROR]:Cannot new instance\n");
        catchError(env);
        exit(1);
    }
    return tensor_obj;
}

napi_value T_add(napi_env env,napi_callback_info info){

    size_t argc = 2;
    napi_value argv[2];
    napi_value constructor;
    napi_status status;
    napi_status cb_status = napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);

    if(cb_status != napi_ok){
        fprintf(stderr,"[ERROR]:Cannot get the info\n");
        exit(1);
    }
    if(argv != NULL){;

        Tensor *tensor_one;
        napi_unwrap(env,argv[0],&tensor_one);//obj_one

        Tensor *tensor_two;
        napi_unwrap(env,argv[1],&tensor_two);//obj_two

        Tensor *new_tensor = tensor_add(tensor_one,tensor_two);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}


napi_value T_min(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_value constructor;

    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor;
        napi_unwrap(env,argv[0],&tensor);

        int32_t axis;
        napi_get_value_int32(env,argv[1],&axis);//轴
        bool keepdims;
        napi_get_value_bool(env,argv[2],&keepdims);//保持维度

        Tensor *new_tensor = tensor_min(tensor,axis,keepdims);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value T_max(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor;
        napi_unwrap(env,argv[0],&tensor);

        int32_t axis;
        napi_status axis_status = napi_get_value_int32(env,argv[1],&axis);//轴
        bool keepdims;
        napi_status bool_status = napi_get_value_bool(env,argv[2],&keepdims);//保持维度
        if(axis_status != napi_ok||bool_status != napi_ok){
            fprintf(stderr,"[ERROR]:Cannot get the bool || aixs\n");
            exit(1);
        }
        Tensor *new_tensor = tensor_max(tensor,axis,keepdims);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value T_sub(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor_one;
        napi_unwrap(env,argv[0],&tensor_one);//obj_one

        Tensor *tensor_two;
        napi_unwrap(env,argv[1],&tensor_two);//obj_two
    
        Tensor *new_tensor = tensor_sub(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value T_div_S(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor;
        napi_unwrap(env,argv[0],&tensor);
        
        double _scalar;
        napi_get_value_double(env,argv[1],&_scalar);
        Tensor *new_tensor = tensor_div_scalar(tensor,_scalar);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}//!

napi_value T_sub_broadcasted(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor_one;
        napi_unwrap(env,argv[0],&tensor_one);//obj_one

        Tensor *tensor_two;
        napi_unwrap(env,argv[1],&tensor_two);//obj_two
    
        Tensor *new_tensor = tensor_sub_broadcasted(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value T_mul_T_ele(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor_one;
        napi_unwrap(env,argv[0],&tensor_one);//obj_one

        Tensor *tensor_two;
        napi_unwrap(env,argv[1],&tensor_two);//obj_two
        Tensor *new_tensor = tensor_mul_elementwise(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value dot(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor_one;
        napi_unwrap(env,argv[0],&tensor_one);//obj_one

        Tensor *tensor_two;
        napi_unwrap(env,argv[1],&tensor_two);//obj_two
        Tensor *new_tensor = tensor_dot(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value T_add_broadcasted(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor_one;
        napi_unwrap(env,argv[0],&tensor_one);//obj_one

        Tensor *tensor_two;
        napi_unwrap(env,argv[1],&tensor_two);//obj_two
    
        Tensor *new_tensor = tensor_add_broadcasted(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value S_mul_T(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor;
        napi_unwrap(env,argv[0],&tensor);
        
        double _scalar;
        napi_get_value_double(env,argv[1],&_scalar);
        Tensor *new_tensor = scalar_mul_tensor(tensor,_scalar);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value S_div_T(napi_env env,napi_callback_info info){
    size_t argc= 2;
    napi_value argv[2];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor;
        napi_unwrap(env,argv[0],&tensor);
        
        double _scalar;
        napi_get_value_double(env,argv[1],&_scalar);
        Tensor *new_tensor = scalar_div_tensor(tensor,_scalar);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value T_div_T(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor_one;
        napi_unwrap(env,argv[0],&tensor_one);//obj_one

        Tensor *tensor_two;
        napi_unwrap(env,argv[1],&tensor_two);//obj_two
    
        Tensor *new_tensor = tensor_div_tensor(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value reshape(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor;
        napi_unwrap(env,argv[0],&tensor);

        awl new_shape = get_node_array(env,argv[1],true);
        
        Tensor *new_tensor = tensor_reshape(tensor,(int32_t *)new_shape.data,new_shape.length);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value T_pow(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor;
        napi_unwrap(env,argv[0],&tensor);
        
        double _scalar;
        napi_get_value_double(env,argv[1],&_scalar);
        Tensor *new_tensor = tensor_pow(tensor,_scalar);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value S_pow_T(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor;
        napi_unwrap(env,argv[0],&tensor);
        
        double _scalar;
        napi_get_value_double(env,argv[1],&_scalar);
        Tensor *new_tensor = scalar_pow_tensor(_scalar,tensor);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value ln(napi_env env,napi_callback_info info){
    size_t argc = 1;
    napi_value argv[1];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor;
        napi_unwrap(env,argv[0],&tensor);
        
        Tensor *new_tensor = tensor_ln(tensor);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value equal(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor_one;
        napi_unwrap(env,argv[0],&tensor_one);//obj_one

        Tensor *tensor_two;
        napi_unwrap(env,argv[1],&tensor_two);//obj_two
    
        Tensor *new_tensor = tensor_equal(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}


napi_value T_equal_broadcasted(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor_one;
        napi_unwrap(env,argv[0],&tensor_one);//obj_one

        Tensor *tensor_two;
        napi_unwrap(env,argv[1],&tensor_two);//obj_two
    
        Tensor *new_tensor = tensor_equal_broadcasted(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value ones(napi_env env,napi_callback_info info){
    size_t argc = 1;
    napi_value argv[1];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor;
        napi_unwrap(env,argv[0],&tensor);
        
        Tensor *new_tensor = ones_like(tensor);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value zeros(napi_env env,napi_callback_info info){
    size_t argc = 1;
    napi_value argv[1];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor;
        napi_unwrap(env,argv[0],&tensor);
        
        Tensor *new_tensor = zeros_like(tensor);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value sins(napi_env env,napi_callback_info info){
    size_t argc = 1;
    napi_value argv[1];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor;
        napi_unwrap(env,argv[0],&tensor);
        
        Tensor *new_tensor = tensor_sin(tensor);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value coss(napi_env env,napi_callback_info info){
    size_t argc = 1;
    napi_value argv[1];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor;
        napi_unwrap(env,argv[0],&tensor);//对象1
        
        Tensor *new_tensor = tensor_cos(tensor);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value transpose(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor;
        napi_unwrap(env,argv[0],&tensor);
        awl new_axis = get_node_array(env,argv[1],true);
        Tensor *new_tensor = tensor_axes_transpose(tensor,(int32_t *)new_axis.data,new_axis.length);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);


        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}



napi_value node_destory_tensor_pool(napi_env env,napi_callback_info info){
    extern MemoryPool *tensor_pool;
    destory_pool(tensor_pool);
    return NULL;
}

napi_value node_reset_tensor_pool(napi_env env,napi_callback_info info){
    extern MemoryPool *tensor_pool;
    reset_pool(tensor_pool);
    return NULL;
}

napi_value T_sum(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor;
        napi_unwrap(env,argv[0],&tensor);

        int32_t axis;
        napi_status axis_status = napi_get_value_int32(env,argv[1],&axis);//轴
        bool keepdims;
        napi_status bool_status = napi_get_value_bool(env,argv[2],&keepdims);//保持维度
        if(axis_status != napi_ok||bool_status != napi_ok){
            fprintf(stderr,"[ERROR]:Cannot get the bool || axe\n");
            exit(1);
        }
        Tensor *new_tensor = tensor_sum(tensor,axis,keepdims);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,constructor);

        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}


napi_value slice(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_value constructor;
    napi_get_cb_info(env,info,&argc,argv,&constructor,NULL);
    if(argv != NULL){
        Tensor *tensor;
        napi_unwrap(env,argv[0],&tensor);

        int32_t axe;
        napi_status axe_status = napi_get_value_int32(env,argv[1],&axe);//轴

        int32_t num;
        napi_status num_status = napi_get_value_int32(env,argv[2],&num);//数量
        if(axe_status != napi_ok || num_status != napi_ok){
            fprintf(stderr,"[ERROR]:Cannot get the axe of num\n");
            exit(1);
        }

        Tensor **new_tensor = tensor_slice(tensor,num,axe);
        napi_value tensor_list;
        napi_status napi_array = napi_create_array(env,&tensor_list);
        if(napi_array != napi_ok){
            fprintf(stderr,"[ERROR]:Cannot create array\n");
            exit(1);
        }

        for(int32_t i = 0; i < num; i++){
            napi_value tensor_obj =  push_node_tensor(env,new_tensor[i],constructor);

            napi_status napi_set_ele = napi_set_element(env,tensor_list,i,tensor_obj);
            if(napi_set_ele != napi_ok){
                fprintf(stdout,"[ERROR]:Cannot set element\n");
            }
        }

        free(new_tensor);
        return tensor_list;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value node_create_pool(napi_env env,napi_callback_info info){
    fprintf(stdout,"[TIP]:Using MemoryPool\n");
    extern MemoryPool *tensor_pool;
    tensor_pool = create_pool(sizeof(Tensor),50);
    return NULL;
}