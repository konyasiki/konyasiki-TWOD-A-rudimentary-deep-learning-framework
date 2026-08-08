import TD from "./TD.ts";
import Tensor from "./tensor.ts";
export abstract class Functions{
    constructor(){

    }
    call(input:TD){
        this.input = input;
        let output = new TD(this.forward(input.data));
        output.setCreator(this);
        this.output = output;
        return output;
    }
    abstract forward(tensor:Tensor):Tensor

    input!:TD

    output!:TD

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
let test = new TD([0.5]);
let aF = new Square();
let bF = new Exp();
let cF = new Square();
let a = aF.call(test);

let b = bF.call(a);

let c = cF.call(b);

c.grad = new Tensor([1]);
b.grad = cF.backward(c.grad);
a.grad = bF.backward(b.grad);
test.grad = aF.backward(a.grad);
console.log(test.grad);