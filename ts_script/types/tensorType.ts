// import Tensor from "../tensor.ts";
// import td from "../TwoD.ts"
export interface init_tensor{
    data:Float64Array;
    shape:Int32Array;
}
export type Data_Shape = {
    Tshape:number[],
    Tdata:number[];
}
export type Flatten = Array<number | Flatten>;
// export type TdList = [td,td]
// export interface forwardPtwo{
//     (parameter_one:Tensor,parameter_two:Tensor):Tensor
// }
// export interface forwardPone{
//     (parameter:Tensor):Tensor
// }