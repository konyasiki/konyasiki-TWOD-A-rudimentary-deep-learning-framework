import type {Flatten } from "./@types/tensorType.ts";
import { Functions } from "./Functions.ts";
import  Tensor  from "./tensor.ts";
export class TwoD{
    constructor(data:Flatten|Tensor){
        if(data instanceof Tensor){
            this.data = data;
        }else{
            this.data = new Tensor(data);
        }
    }
    data:Tensor;
    grad!:Tensor;
    creator!:Functions;
    setCreator(creator:Functions){
        this.creator = creator;
    }
    backward(){
        // if(this.creator != undefined){
        //     let input = this.creator.input;
        //     input.grad = this.creator.backward(this.grad);
        //     input.backward();//递归
        // }
        if(!this.grad){
            this.grad = this.data.ones_like();
        }
        let functionsList:Functions[] = [this.creator];
        while(functionsList.length != 0){
            let functions = functionsList.pop();
            let output = functions!.output;
            let input = functions!.input;
            input.grad = functions!.backward(output.grad);
            if(input.creator != undefined)
                functionsList.push(input.creator);//循环
        }
    }
}
