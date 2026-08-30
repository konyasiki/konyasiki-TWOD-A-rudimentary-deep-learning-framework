
import type {Flatten } from "./types/tensorType.ts";
import { add, Functions, square} from "./basic_config/Functions.ts";
import  Tensor  from "./basic_config/tensor.ts";
export default class TwoD{
    constructor(data:Flatten|Tensor,name?:string){
        this.name = name;
        if(data instanceof Tensor){
            this.data = data;
        }else{
            this.data = new Tensor(data);
        }
    }
    generation:number = 0;
    name:string|undefined;
    data:Tensor;
    grad:Tensor|undefined;
    creator!:Functions;
    setCreator(creator:Functions){
        this.generation = creator.generation + 1;
        this.creator = creator;
    }
    backward(retain_grad = true){
        if(this.grad == undefined){//初始化
            this.grad = this.data.ones_like();
        }
        
        let functionsList:Functions[] = [];//gc
        let FuncSet = new WeakSet();//gc
        function addFun(functions:Functions){
            if(!FuncSet.has(functions)){
                functionsList.push(functions);
                FuncSet.add(functions);
                functionsList.sort((a,b)=>{
                    return a.generation - b.generation;
                })
            }
        }
        addFun(this.creator);
        while(functionsList.length != 0){

            let functions = functionsList.pop();
            let output:WeakRef<TwoD>[] = functions!.output;
            let input = functions!.input;
            
            let output_grad:Tensor[] = [];//遍历输出梯度 //gc
            output.forEach(val=>{
                output_grad.push(val.deref()!.grad!);
            })

            let inputs_grad:Tensor[] = functions!.backward(output_grad);//计算输入梯度
            inputs_grad.forEach((val,index)=>{
                input[index]!.grad = !input[index]?.grad ? val : Tensor.add(val,input[index]!.grad!);//使用同一变量
                if(input[index]?.creator){
                    addFun(input[index]?.creator);

                }
            })

            if(!retain_grad){
                output.forEach(val=>{
                    val.deref()!.grad = undefined;
                })
            }
        }
    }
    cleanGrad(){
        this.grad = undefined;
    }
}