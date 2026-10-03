import {Data_Shape, Flatten} from "../types/tensorType.ts"
declare class Tensorninher{
        private _data:Float64Array;
        private _strides:Int32Array;
        private _shape:Int32Array;
        private _ndim:number;
        private _size:number;

        get data():Float16Array;
        get shape():Int32Array;
        get strides():Int32Array;
        get size():number;
        get ndim():number;

        constructor(data:Flatten);
        constructor(data:Float64Array,shape:Int32Array);

        flatten(data:Flatten):Data_Shape

        static add< T extends Tensorninher>(tensor_one:T, tensor_two: T):T
        static sub<T extends Tensorninher>(tensor_one: T, tensor_two: T): T
        static max<T extends Tensorninher>(tensor: T, axis: number, keepdims: boolean): T
        static min<T extends Tensorninher>(tensor: T, axis: number, keepdims: boolean): T
        static reset_tensor_pool():void
        static destory_tensor_pool():void
        static create_pool():void
        static scalar_div_tensor<T extends Tensorninher>(tensor:T,scalar:number):T
        static div_scalar<T extends Tensorninher>(tensor: T, scalar:number): T
        static tensor_div_tensor<T extends Tensorninher>(tensor_one: T, tensor_two:T): T
        static mul_scalar<T extends Tensorninher>(tensor_one: T, scalar: number): T
        static mat<T extends Tensorninher>(tensor_one: T, tensor_two: T): T
        static mul_elementwise<T extends Tensorninher>(tensor_one:T,tensor_two:T): T
        static add_broadcasted<T extends Tensorninher>(tensor_one: T, tensor_two: T): T
        static sub_broadcasted<T extends Tensorninher>(tensor_one: T, tensor_two: T): T
        static reshape<T extends Tensorninher>(tensor: T, new_shape: Int32Array): T
        static pow_scalar<T extends Tensorninher>(tensor:T, scalar: number):T
        static pow_tensor<T extends Tensorninher>(tensor: T,scalar: number):T
        static ln<T extends Tensorninher>(tensor: T): T
        static equal<T extends Tensorninher>(tensor_one: T, tensor_two: T):T
        static equal_broadcasted<T extends Tensorninher>(tensor_one: T, tensor_two: T): T
        static ones_like<T extends Tensorninher>(tensor: T):T
        static zeros_like<T extends Tensorninher>(tensor: T):T
        static sin<T extends Tensorninher>(tensor: T):T
        static cos<T extends Tensorninher>(tensor: T):T
        static transpose<T extends Tensorninher>(tensor: T, axis: Int32Array):T
        static make_contiguous<T extends Tensorninher>(tensor: T):void//
        static dot<T extends Tensorninher>(tensor_one:T,tensor_two:T): T
        static sum<T extends Tensorninher>(tensor: T, axis: number, keepdims: boolean):T
        static slice<T extends Tensorninher>(tensor:T,axe:number,num:number):T
}