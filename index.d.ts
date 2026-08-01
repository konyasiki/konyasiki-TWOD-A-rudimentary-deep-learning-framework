import type { Flatten } from "../../../ts_script/@types/tensorType.ts";
import Tensor from "../../../ts_script/tensor.ts"
const tensor_c = {
    add:(tensor_one: Tensor, tensor_two: Tensor,constructor:new (data:Flatten)=>Tensor)=>new Tensor,
    sub:(tensor_one: Tensor, tensor_two: Tensor,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    max:(tensor: Tensor, axis: number, keepdims: boolean,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    min:(tensor: Tensor, axis: number, keepdims: boolean,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    reset_tensor_pool:()=>null,
    destory_tensor_pool:()=>null,
    create_pool:()=>null,
    scalar_div_tensor:(tensor:Tensor,scalar:number,constructor:new (data:Flatten)=>Tensor)=> newTensor,
    div_scalar:(tensor: Tensor, scalar:number,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    tensor_div_tensor:(tensor_one: Tensor, tensor_two:Tensor,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    mul_scalar:(tensor_one: Tensor, scalar: number,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    mat:(tensor_one: Tensor, tensor_two: Tensor,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    mul_elementwise:(tensor_one:Tensor,tensor_two:Tensor,constructor:new (data:Flatten)=>Tensor)=> newTensor,
    add_broadcasted:(tensor_one: Tensor, tensor_two: Tensor,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    sub_broadcasted:(tensor_one: Tensor, tensor_two: Tensor,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    reshape:(tensor: Tensor, new_shape: Int32Array,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    pow_scalar:(tensor:Tensor, scalar: number,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    pow_tensor:(tensor: Tensor,scalar: number,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    ln:(tensor: Tensor,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    equal:(tensor_one: Tensor, tensor_two: Tensor,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    equal_broadcasted:(tensor_one: Tensor, tensor_two: Tensor,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    ones_like:(tensor: Tensor,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    zeros_like:(tensor: Tensor,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    sins:(tensor: Tensor,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    cos:(tensor: Tensor,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    transpose:(tensor: Tensor, axis: Int32Array,constructor:new (data:Flatten)=>Tensor)=> new Tensor,
    make_contiguous:(tensor: Tensor)=> new null,//!
}
export default tensor_c;
