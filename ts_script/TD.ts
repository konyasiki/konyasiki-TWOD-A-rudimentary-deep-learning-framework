import type {Flatten } from "./@types/tensorType.ts";
import { Functions } from "./Functions.ts";
import  Tensor  from "./tensor.ts";
export default class TD{
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
}