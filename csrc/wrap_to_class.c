#include <stdio.h>
#include "node_api.h"
#include <stdlib.h>
#include <stdbool.h>
#include "c_to_node.h"

napi_value constructor(napi_env env,napi_callback_info info){

    napi_status status;
    size_t argc = 2;
    napi_value argv[2];
    napi_value this;
    bool instance_of_array;
    
    status = napi_get_cb_info(env,info,&argc,argv,&this,NULL);
    if(status != napi_ok)
        

        goto send_err;
    
    status = napi_is_array(env,argv[0],&instance_of_array);
    if(status != napi_ok)
        goto send_err;
    
    awl data;
    awl shape;
    
    napi_value node_data;
    napi_value node_shape;

    if(instance_of_array){
        napi_value T_obj;
        napi_value flatten;

        status = napi_get_named_property(env,this,"flatten",&flatten);
        if(status != napi_ok)
            goto send_err;
        
        status = napi_call_function(env,this,flatten,1,argv,&T_obj);
        if(status != napi_ok)
            goto send_err;
        
        napi_value T_data;
        napi_value T_shape;
        
        status = napi_get_named_property(env,T_obj,"Tdata",&T_data);
        if(status != napi_ok)
            goto send_err;
        
        status = napi_get_named_property(env,T_obj,"Tshape",&T_shape);
        if(status != napi_ok)
            goto send_err;
        
        data = get_node_array(env,T_data,true);
        shape = get_node_array(env,T_shape,true);
        
        node_shape = T_shape;
        node_data = T_data;
    }else{

        data = get_node_array(env,argv[0],true);
        shape = get_node_array(env,argv[1],true);
        node_data = argv[0];
        node_shape = argv[1];
    }

    Tensor *tensor = create_tensor((double_t *)data.data,(int32_t *)shape.data,shape.length);

    napi_value strides = create_typedarray(env,tensor->strides,tensor->dimension,napi_int32_array);

    napi_value ndim;

    status = napi_create_int32(env,shape.length,&ndim);
    if(status != napi_ok)
        goto send_err;

    napi_value size;

    status = napi_create_int32(env,data.length,&size);
    if(status != napi_ok)
        goto send_err;

    napi_property_descriptor properties[] = {
        {"#data",NULL,NULL,NULL,NULL,node_data,napi_default_jsproperty,NULL},
        {"#ndim",NULL,NULL,NULL,NULL,ndim,napi_default_jsproperty,NULL},
        {"#strides",NULL,NULL,NULL,NULL,strides,napi_default_jsproperty,NULL},
        {"#shape",NULL,NULL,NULL,NULL,node_shape,napi_default_jsproperty,NULL},
        {"#size",NULL,NULL,NULL,NULL,size,napi_default_jsproperty,NULL}
    };
    status = napi_wrap(env,this,tensor,&free_c_variable,tensor,NULL);
    if(status != napi_ok)
        goto send_err;
    
    status = napi_define_properties(env,this,5,properties);

    if(status != napi_ok)
        goto send_err;

    return this;
    send_err:
        fprintf(stderr,"[ERROR]:Fail to call constructor\n");
        catchError(env);
        exit(1);
}
napi_value Getdata(napi_env env,napi_callback_info info){

    napi_value this;
    napi_get_cb_info(env,info,NULL,NULL,&this,NULL);

    napi_value out;
    napi_get_named_property(env,this,"#data",&out);
    return out;
}

napi_value Getshape(napi_env env,napi_callback_info info){

    napi_value this;
    napi_get_cb_info(env,info,NULL,NULL,&this,NULL);

    napi_value out;
    napi_get_named_property(env,this,"#shape",&out);
    return out;
}

napi_value Getndim(napi_env env,napi_callback_info info){

    napi_value this;
    napi_get_cb_info(env,info,NULL,NULL,&this,NULL);

    napi_value out;
    napi_get_named_property(env,this,"#ndim",&out);
    return out;
}

napi_value Getsize(napi_env env,napi_callback_info info){

    napi_value this;
    napi_get_cb_info(env,info,NULL,NULL,&this,NULL);

    napi_value out;
    napi_get_named_property(env,this,"#size",&out);
    return out;
}
napi_value Getstrides(napi_env env,napi_callback_info info){

    napi_value this;
    napi_get_cb_info(env,info,NULL,NULL,&this,NULL);

    napi_value out;
    napi_get_named_property(env,this,"#strides",&out);
    return out;
}
napi_value Init(napi_env env,napi_value exports){
    napi_status status;
    napi_property_descriptor properties[] = {
        {"data",NULL,NULL,Getdata,NULL,NULL,napi_default,NULL},
        {"shape",NULL,NULL,Getshape,NULL,NULL,napi_default,NULL},
        {"ndim",NULL,NULL,Getndim,NULL,NULL,napi_default,NULL},
        {"size",NULL,NULL,Getsize,NULL,NULL,napi_default,NULL},
        {"strides",NULL,NULL,Getstrides,NULL,NULL,napi_default,NULL},        //静态方法
        {"add",NULL,T_add,NULL,NULL,NULL,napi_static,NULL},
        {"sub",NULL,T_sub,NULL,NULL,NULL,napi_static,NULL},
        {"min",NULL,T_min,NULL,NULL,NULL,napi_static,NULL},
        {"max",NULL,T_max,NULL,NULL,NULL,napi_static,NULL},
        {"div",NULL,T_div_S,NULL,NULL,NULL,napi_static,NULL},//
        {"mul",NULL,T_mul_T_ele,NULL,NULL,NULL,napi_static,NULL},//
        {"add_broadcast",NULL,T_add_broadcasted,NULL,NULL,NULL,napi_static,NULL},
        {"sub_broadcast",NULL,T_sub_broadcasted,NULL,NULL,NULL,napi_static,NULL},
        {"reshape",NULL,reshape,NULL,NULL,NULL,napi_static,NULL},
        {"dot",NULL,dot,NULL,NULL,NULL,napi_static,NULL},// {"pow",NULL,pow_static,NULL,NULL,NULL,napi_static,NULL},
        {"ln",NULL,ln,NULL,NULL,NULL,napi_static,NULL},
        {"equal",NULL,equal,NULL,NULL,NULL,napi_static,NULL},
        {"equal_broadcast",NULL,T_equal_broadcasted,NULL,NULL,NULL,napi_static,NULL},
        {"ones_like",NULL,ones,NULL,NULL,NULL,napi_static,NULL},
        {"zeros_like",NULL,zeros,NULL,NULL,NULL,napi_static,NULL},
        {"transpose",NULL,transpose,NULL,NULL,NULL,napi_static,NULL},
        {"sin",NULL,sins,NULL,NULL,NULL,napi_static,NULL},
        {"cos",NULL,coss,NULL,NULL,NULL,napi_static,NULL},
        {"sum",NULL,T_sum,NULL,NULL,NULL,napi_static,NULL},//少了S_sum_T
        {"slice",NULL,slice,NULL,NULL,NULL,napi_static,NULL}//少了flatten
    };

    napi_value node_constructor;
    status = napi_define_class(env,"Tensorninher",NAPI_AUTO_LENGTH,constructor,NULL,25,properties,&node_constructor);
    if(status != napi_ok){
        fprintf(stderr,"[ERROR]:Failed to define class\n");
        goto send_err;
    }

    status = napi_set_named_property(env,exports,"Tensorninher",node_constructor);
    if(status != napi_ok){
        fprintf(stderr,"[ERROR]:Failed to set property\n");
        goto send_err;
    }

    return exports;

    send_err:
        fprintf(stderr,"[ERROR]:Fail to init\n");
        catchError(env);
        exit(1);
}
NAPI_MODULE(NODE_GYP_MODULE_NAME,Init);