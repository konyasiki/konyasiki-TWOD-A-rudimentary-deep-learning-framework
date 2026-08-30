import Tensor from "../tensor.ts";
import type { Flatten } from "../types/tensorType.ts";
export interface tensor_c_type{
    add:(tensor_one: Tensor, tensor_two: Tensor,constructor:new (data:Flatten)=>Tensor)=>Tensor,
    sub:(tensor_one: Tensor, tensor_two: Tensor,constructor:new (data:Flatten)=>Tensor)=> Tensor,
    max:(tensor: Tensor, axis: number, keepdims: boolean,constructor:new (data:Flatten)=>Tensor)=> Tensor,
    min:(tensor: Tensor, axis: number, keepdims: boolean,constructor:new (data:Flatten)=>Tensor)=> Tensor,
    reset_tensor_pool:()=>null,
    destory_tensor_pool:()=>null,
    create_pool:()=>null,
    scalar_div_tensor:(tensor:Tensor,scalar:number,constructor:new (data:Flatten)=>Tensor)=>Tensor,
    div_scalar:(tensor: Tensor, scalar:number,constructor:new (data:Flatten)=>Tensor)=> Tensor,
    tensor_div_tensor:(tensor_one: Tensor, tensor_two:Tensor,constructor:new (data:Flatten)=>Tensor)=> Tensor,
    mul_scalar:(tensor_one: Tensor, scalar: number,constructor:new (data:Flatten)=>Tensor)=> Tensor,
    mat:(tensor_one: Tensor, tensor_two: Tensor,constructor:new (data:Flatten)=>Tensor)=> Tensor,
    mul_elementwise:(tensor_one:Tensor,tensor_two:Tensor,constructor:new (data:Flatten)=>Tensor)=> Tensor,
    add_broadcasted:(tensor_one: Tensor, tensor_two: Tensor,constructor:new (data:Flatten)=>Tensor)=> Tensor,
    sub_broadcasted:(tensor_one: Tensor, tensor_two: Tensor,constructor:new (data:Flatten)=>Tensor)=> Tensor,
    reshape:(tensor: Tensor, new_shape: Int32Array,constructor:new (data:Flatten)=>Tensor)=> Tensor,
    pow_scalar:(tensor:Tensor, scalar: number,constructor:new (data:Flatten)=>Tensor)=>Tensor,
    pow_tensor:(tensor: Tensor,scalar: number,constructor:new (data:Flatten)=>Tensor)=>Tensor,
    ln:(tensor: Tensor,constructor:new (data:Flatten)=>Tensor)=> Tensor,
    equal:(tensor_one: Tensor, tensor_two: Tensor,constructor:new (data:Flatten)=>Tensor)=>Tensor,
    equal_broadcasted:(tensor_one: Tensor, tensor_two: Tensor,constructor:new (data:Flatten)=>Tensor)=> Tensor,
    ones_like:(tensor: Tensor,constructor:new (data:Flatten)=>Tensor)=>Tensor,
    zeros_like:(tensor: Tensor,constructor:new (data:Flatten)=>Tensor)=>Tensor,
    sins:(tensor: Tensor,constructor:new (data:Flatten)=>Tensor)=>Tensor,
    coss:(tensor: Tensor,constructor:new (data:Flatten)=>Tensor)=>Tensor,
    transpose:(tensor: Tensor, axis: Int32Array,constructor:new (data:Flatten)=>Tensor)=>Tensor,
    make_contiguous:(tensor: Tensor)=>null,//
    sum:(tensor: Tensor, axis: number, keepdims: boolean,constructor:new (data:Flatten)=>Tensor)=>Tensor
}