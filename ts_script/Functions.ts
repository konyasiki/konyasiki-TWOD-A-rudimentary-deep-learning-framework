import {TwoD} from "./TwoD.ts";
import Tensor from "./tensor.ts";
import assert, { Assert } from "assert";
export abstract class Functions{
    constructor(){

    }
    call(input:TwoD){
        this.input = input;
        let output = new TwoD(this.forward(input.data));
        output.setCreator(this);
        this.output = output;
        return output;
    }
    abstract forward(tensor:Tensor):Tensor

    input!:TwoD

    output!:TwoD

    abstract backward(grad:Tensor):Tensor
}

export class Square extends Functions{
    forward(input:Tensor):Tensor{
        return Tensor.pow(input,2);
    }
    backward(grad:Tensor):Tensor{
        return Tensor.mul(this.input.data,Tensor.mul(grad,2));
    }
}
export class Exp extends Functions{
    forward(tensor:Tensor){
        return Tensor.pow(Math.E,tensor);
    }
    backward(grad:Tensor){
        return Tensor.mul(Tensor.pow(Math.E,this.input.data),grad);
    }
}
let S1 = new Square();
let S2 = new Square();
let E = new Exp();
let test = new TwoD([0.5]);
let a = S1.call(test);
let b = E.call(a);
let c = S2.call(b);
c.backward();
console.log(test.grad);