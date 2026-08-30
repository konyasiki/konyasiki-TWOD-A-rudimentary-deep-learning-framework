import { transcode } from "buffer";
import td from "../TwoD.ts";
import config from "./Config.ts";
import Tensor from "./tensor.ts";
import { transpose } from "numpy-ts";
export abstract class Functions{
    constructor(){
    }
    call(...input:td[]){
        this.input = input;
        // this.generation = Math.max(input[0].generation,input[1].generation);

        let inputs:Tensor[] = [];//gc
        input.forEach(val=>{
            inputs.push(val.data);
        });

        let tdlist:td[]= [];
        let forward_output = this.forward(inputs);//gc
        forward_output.forEach(val=>{
            tdlist.push(new td(val));
        })
        if(!config.enable_backward){

            let max_generation = 0;
            input.forEach(val=>{
                max_generation = val.generation >= max_generation ? val.generation : max_generation;
            })
            
            this.generation = max_generation;
            tdlist.forEach(val=>{
                val.setCreator(this);
            })
        }
        
        tdlist.forEach(val=>{
            this.output.push(new WeakRef(val));
        })
        return tdlist.length > 1 ? tdlist : tdlist[0]!;
    }

    abstract forward(inputs:Tensor[]):Tensor[];
    abstract backward(grad:Tensor[]):Tensor[];

    input!:td[];
    output:WeakRef<td>[] = [];
    generation:number = 0;
}

class Square extends Functions{
    forward(input:Tensor[]){
        return [Tensor.pow(input[0]!,2)];
    }
    backward(grad:Tensor[]){
        console.log(`grad:${grad[0]}`);
        // console.log(`input:${this.input[0]!.data}`)
        return [Tensor.mul((this.input[0]!).data,Tensor.mul(grad[0]!,2))];
    }
}
class Exp extends Functions{
    forward(tensor:Tensor[]){
        return [Tensor.pow(Math.E,tensor[0]!)];
    }
    backward(grad:Tensor[]){
        return [Tensor.mul(Tensor.pow(Math.E,(this.input[0])!.data),grad[0]!)];
    }
}
class Add extends Functions{
    forward(Tensors:Tensor[]){
        return [Tensor.add(Tensors[0]!,Tensors[1]!)];
    }
    backward(grad: Tensor[]){
        return [grad[0]!,grad[0]!];
    }
}
export function square(input:td){
    let out = new Square();
    return (out.call(input) as td);
}
export function exp(input:td){
    let out = new Exp();
    return (out.call(input) as td);
}
export function add(input_one:td,input_two:td){
    let out = new Add();
    return (out.call(input_one,input_two) as td);
}
export function sub(){

}
export function max(){

}
export function min(){

}
export function div(){

}
export function mul(){

}
export function mat(){

}
export function reshape(){

}
export function pow(){

}
export function ln(){

}
export function equal(){

}
export function ones_like(){

}
export function zeros_like(){

}
export function sin(){

}
export function cos(){

}
export function transpose(){

}
export function sum(){
    
}
//断言搁置
// class SquareTest extends Assert{

// }