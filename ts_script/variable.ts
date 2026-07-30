import type {Flatten } from "./@types/tensorType.ts";
import { Tensor } from "./tensor.ts";
export class Variable extends Tensor{//默认cpu float32
    constructor(data:Flatten){
        super(data);
    }
    grad!:number;
    backward(){

    }
    forward(){

    }
}