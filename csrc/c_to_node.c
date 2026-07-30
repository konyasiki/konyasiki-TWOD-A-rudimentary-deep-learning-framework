#include "tensor.h"
#include "node_api.h"
#include <stdlib.h>
#include <stdio.h>
#include "MemoryPool.h"

typedef struct array_with_length{
    void *data;
    size_t length;
}awl;

napi_value create_typedarray(napi_env env,void *array,int32_t length){
    size_t byte_legth = sizeof(int32_t) * length;
    napi_value arraybuffer;
    napi_value typedarray = NULL;
    napi_status status = napi_create_arraybuffer(env,byte_legth,&array,&arraybuffer);
    napi_status statu2 = napi_create_typedarray(env,napi_float64_array,length,arraybuffer,byte_legth,&typedarray);
    return typedarray;
}

awl get_node_array(napi_env env,napi_value argv,bool return_length){
    void *data;
    size_t byte_length;
    size_t offest,length;//偏移,数组长度

    napi_value input_array = argv;
    napi_typedarray_type type;//参数类型
    napi_value input_buffer = NULL;//元数据
    napi_status typedarray_status = napi_get_typedarray_info(env,input_array,&type,&length,NULL,&input_buffer,&offest);
    napi_status buffer_status = napi_get_arraybuffer_info(env,input_buffer,&data,&byte_length);
    if(buffer_status != napi_ok  || typedarray_status != napi_ok){
        fprintf(stderr,"[ERROR]:cannot get buffer_info && typedarray_info\n");
    }
    awl node_Array;
    node_Array.data = data;
    node_Array.length = 0;
    if(return_length){
        node_Array.length = length;
        return node_Array;
    }
    return node_Array;
}

Tensor *get_node_tensors(napi_env env,napi_value argv){
    napi_handle_scope scope;
    napi_open_handle_scope(env,&scope);

    napi_value Typed_data;
    napi_status proper_data = napi_get_named_property(env,argv,"data",&Typed_data);
    napi_value shape;
    napi_status proper_shape = napi_get_named_property(env,argv,"shape",&shape);
    if(proper_data != napi_ok || proper_shape != napi_ok){
        fprintf(stderr,"[ERROR]:Cannot get the property\n");
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
    size_t con_argc = 1;
    napi_value con_argv;
    napi_status array_status =  napi_create_array(env,&con_argv);
    for(int32_t i = 0;i < tensor->size;i++){
        napi_value data;
        napi_create_double(env,tensor->data[i],&data);
        napi_set_element(env,con_argv,i,data);
    }
    napi_status instance_status = napi_new_instance(env,constructor,con_argc,&con_argv,&tensor_obj);
    if(array_status != napi_ok || instance_status != napi_ok){
        fprintf(stderr,"[ERROR]:Cannot new instance\n");
    }
    return tensor_obj;
}

napi_value T_add(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_status cb_status = napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(cb_status != napi_ok){
        fprintf(stderr,"[ERROR]:Cannot get the info\n");
    }
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        Tensor *tensor_two = get_node_tensors(env,argv[1]);//对象2
        Tensor *new_tensor = tensor_add(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(tensor_two);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}


napi_value T_min(napi_env env,napi_callback_info info){
    size_t argc = 4;
    napi_value argv[4];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象

        int32_t *axis = 0;
        napi_get_value_int32(env,argv[1],axis);//轴
        bool keepdims;
        napi_get_value_bool(env,argv[2],&keepdims);//保持维度

        Tensor *new_tensor = tensor_min(tensor_one,*axis,keepdims);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[3]);

        delete_tensor(tensor_one);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value T_max(napi_env env,napi_callback_info info){
    size_t argc = 4;
    napi_value argv[4];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象

        int32_t *axis = 0;
        napi_get_value_int32(env,argv[1],axis);//轴
        bool keepdims;
        napi_get_value_bool(env,argv[2],&keepdims);//保持维度

        Tensor *new_tensor = tensor_max(tensor_one,*axis,keepdims);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[3]);

        delete_tensor(tensor_one);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value T_sub(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        Tensor *tensor_two = get_node_tensors(env,argv[1]);//对象2
    
        Tensor *new_tensor = tensor_sub(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(tensor_two);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value T_div_S(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        
        double _scalar;
        napi_get_value_double(env,argv[1],&_scalar);
        Tensor *new_tensor = tensor_div_scalar(tensor_one,_scalar);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value T_sub_broadcasted(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        Tensor *tensor_two = get_node_tensors(env,argv[1]);//对象2
    
        Tensor *new_tensor = tensor_sub_broadcasted(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(tensor_two);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value T_mul_T_ele(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        Tensor *tensor_two = get_node_tensors(env,argv[1]);//对象2
    
        Tensor *new_tensor = tensor_mul_elementwise(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(tensor_two);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value T_add_broadcasted(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        Tensor *tensor_two = get_node_tensors(env,argv[1]);//对象2
    
        Tensor *new_tensor = tensor_add_broadcasted(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(tensor_two);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value S_mul_T(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        
        double _scalar;
        napi_get_value_double(env,argv[1],&_scalar);
        Tensor *new_tensor = scalar_mul_tensor(tensor_one,_scalar);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value S_div_T(napi_env env,napi_callback_info info){
    size_t argc= 3;
    napi_value argv[3];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        
        double _scalar;
        napi_get_value_double(env,argv[1],&_scalar);
        Tensor *new_tensor = scalar_div_tensor(tensor_one,_scalar);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value T_div_T(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        Tensor *tensor_two = get_node_tensors(env,argv[1]);//对象2
    
        Tensor *new_tensor = tensor_div_tensor(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(tensor_two);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value reshape(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1

        awl new_shape = get_node_array(env,argv[1],true);
        
        Tensor *new_tensor = tensor_reshape(tensor_one,(int32_t *)new_shape.data,new_shape.length);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value T_matul_T(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        Tensor *tensor_two = get_node_tensors(env,argv[1]);//对象2
    
        Tensor *new_tensor = tensor_matmul(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(tensor_two);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value T_pow(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        
        double _scalar;
        napi_get_value_double(env,argv[1],&_scalar);
        Tensor *new_tensor = tensor_pow(tensor_one,_scalar);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value S_pow_T(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        
        double _scalar;
        napi_get_value_double(env,argv[1],&_scalar);
        Tensor *new_tensor = scalar_pow_tensor(_scalar,tensor_one);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value ln(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        
        Tensor *new_tensor = tensor_ln(tensor_one);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[1]);
        delete_tensor(tensor_one);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value equal(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        Tensor *tensor_two = get_node_tensors(env,argv[1]);//对象2
    
        Tensor *new_tensor = tensor_equal(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(tensor_two);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}


napi_value T_equal_broadcasted(napi_env env,napi_callback_info info){
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        Tensor *tensor_two = get_node_tensors(env,argv[1]);//对象2
    
        Tensor *new_tensor = tensor_equal_broadcasted(tensor_one,tensor_two);
        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(tensor_two);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value ones(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        
        Tensor *new_tensor = ones_like(tensor_one);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value zeros(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        
        Tensor *new_tensor = zeros_like(tensor_one);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value sins(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        
        Tensor *new_tensor = tensor_sin(tensor_one);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value coss(napi_env env,napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        
        Tensor *new_tensor = tensor_cos(tensor_one);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[2]);
        delete_tensor(tensor_one);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value transpose(napi_env env,napi_callback_info info){
    size_t argc = 4;
    napi_value argv[4];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1

        awl new_axis = get_node_array(env,argv[1],true);
        
        Tensor *new_tensor = tensor_reshape(tensor_one,(int32_t *)new_axis.data,new_axis.length);

        napi_value tensor_obj =  push_node_tensor(env,new_tensor,argv[4]);
        delete_tensor(tensor_one);
        delete_tensor(new_tensor);
        return tensor_obj;
    }else{
        fprintf(stderr,"[ERROR]:Missing parameter\n");
        exit(1);
    }
}

napi_value contiguous(napi_env env,napi_callback_info info){
    size_t argc = 1;
    napi_value argv[1];
    napi_get_cb_info(env,info,&argc,argv,NULL,NULL);
    if(argv != NULL){
        Tensor *tensor_one = get_node_tensors(env,argv[0]);//对象1
        
        make_contiguous(tensor_one);
        return NULL;
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

napi_value node_create_pool(napi_env env,napi_callback_info info){
    fprintf(stdout,"[TIP]:Using MemoryPool\n");
    extern MemoryPool *tensor_pool;
    tensor_pool = create_pool(sizeof(Tensor),50);
    return NULL;
}

void init_function(napi_env env,napi_value exports,napi_callback cfuntion,char *function_name){
    napi_value function;
    napi_create_function(env, NULL, 0,cfuntion,NULL, &function);
    napi_set_named_property(env,exports,function_name,function);
}



napi_value Init(napi_env env,napi_value exports){//我知道我这一段写得很屎但我没招了

    init_function(env,exports,T_add,"add");

    init_function(env,exports,T_sub,"sub");
    
    init_function(env,exports,T_max,"max");
    
    init_function(env,exports,T_min,"min");

    init_function(env,exports,node_destory_tensor_pool,"destory_tensor_pool");

    init_function(env,exports,node_reset_tensor_pool,"reset_tensor_pool");

    init_function(env,exports,T_div_S,"div_scalar");

    init_function(env,exports,T_sub_broadcasted,"sub_broadcasted");

    init_function(env,exports,T_mul_T_ele,"mul_elementwise");

    init_function(env,exports,T_add,"add_broadcasted");

    init_function(env,exports,S_mul_T,"mul_scalar");

    init_function(env,exports,S_div_T,"scalar_div_tensor");

    init_function(env,exports,T_div_T,"tensor_div_tensor");

    init_function(env,exports,reshape,"reshape");

    init_function(env,exports,T_matul_T,"mat");

    init_function(env,exports,T_pow,"pow_scalar");

    init_function(env,exports,S_pow_T,"pow_tensor");

    init_function(env,exports,ln,"ln");

    init_function(env,exports,equal,"equal");

    init_function(env,exports,T_equal_broadcasted,"equal_broadcasted");

    init_function(env,exports,ones,"ones_like");

    init_function(env,exports,zeros,"zeros_like");

    init_function(env,exports,sins,"sin");

    init_function(env,exports,coss,"cos");

    init_function(env,exports,transpose,"transpose");

    init_function(env,exports,contiguous,"make_contiguous");

    init_function(env,exports,node_create_pool,"create_pool");
    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)