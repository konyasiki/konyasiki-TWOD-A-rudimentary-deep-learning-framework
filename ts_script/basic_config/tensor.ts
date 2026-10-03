import type { Data_Shape,Flatten} from "../types/tensorType.ts"
import tensor_c from "../tensor_c/index.js"
export default class Tensor extends tensor_c.Tensorninher{
    flatten(data:Flatten):Data_Shape{
        
        let Tdata:Flatten|number[] = [];
        let shape:number[] = [];
        Tdata = data;//root
        let length:number|null = Tdata.length;
        shape.push(Tdata.length);
        while(Tdata.length > 0){
            length--;
            if(Tdata[0] instanceof Array){
                let Data = Tdata.shift();
                if(Data instanceof Array){
                    Data.forEach(value=>{
                        Tdata.push(value);
                        if(length == 0){
                            shape.push(Data.length);
                            length = Data[0] instanceof Array ? Data.length : null;
                        }
                    })
                }
            }else{
                break;
            }
        }
        return {Tdata:new Float64Array(Tdata as number[]),Tshape:new Int32Array(shape)};
    }
    add(tensor:Tensor){
        return Tensor.add(this,tensor);
    }
    sub(tensor:Tensor){
        return Tensor.sub(this,tensor);;
    }
    max(axis:number,keepdims:boolean){//
        return Tensor.max(this,axis,keepdims);
    }
    min(axis:number,keepdims:boolean){//
        return Tensor.min(this,axis,keepdims);
    }
    sum(axis:number,keepdims:boolean){//
        return Tensor.sum(this,axis,keepdims);
    }
    div(parameter:Tensor|number,S_div_T:boolean = false):Tensor|never{
        if(typeof parameter == "number"){
            return Tensor.div_scalar(this,parameter);
        }
        if(parameter instanceof Tensor){
            return Tensor.tensor_div_tensor(this,parameter);
        }
        if(S_div_T && typeof parameter == "number"){
            return Tensor.scalar_div_tensor(this,parameter);
        }else{
            throw new Error("[ERROR]:The parameter cannot  be 'Tensor' when S_div_T equal false\n");
        }
    }
    mul(parameter:Tensor|number){
        if(typeof parameter == "number"){
            return Tensor.mul_scalar(this,parameter);
        }else{
            return Tensor.mul_elementwise(this,parameter);
        }
    }
    add_broadcasted(tensor:Tensor){
        return Tensor.add_broadcasted(this,tensor);
    }
    sub_broadcasted(tensor:Tensor){
        return Tensor.sub_broadcasted(this,tensor);
    }
    reshape(new_shape:number[]){
        let shape = new Int32Array(new_shape);
        return Tensor.reshape(this,shape);
    }
    pow(parameter:number,T_pow_S:boolean = true){//
        if(T_pow_S){
            return Tensor.pow_scalar(this,parameter);
        }else{
            return Tensor.pow_tensor(this,parameter);
        }
    }
    ln(){
        return Tensor.ln(this);
    }
    equal(tensor:Tensor){
        return Tensor.equal(this,tensor);
    }
    equal_broadcasted(tensor:Tensor){
        return Tensor.equal_broadcasted(this,tensor);
    }
    ones_like(){
        return Tensor.ones_like(this);
    }
    zeros_like(){
        return Tensor.zeros_like(this);
    }
    sin(){
        return Tensor.sin(this);
    }
    cos(){
        return Tensor.cos(this);
    }
    transpose(axis:number[]){
        let _axis = new Int32Array(axis);
        return Tensor.transpose(this,_axis);
    }
}
