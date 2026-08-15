import type { Data_Shape,Flatten} from "./@types/tensorType.ts"
import tensor_c from "tensor_c"

export default class Tensor{
    private _data:Float64Array;
    private _strides!:Int32Array;
    private _shape:Int32Array;
    private _ndim:number;
    private _size:number;
    constructor(data:Flatten){
        let {Tdata,Tshape} = this.flatten(data);
        this._shape = new Int32Array(Tshape);
        this._data = new Float64Array(Tdata);
        this._ndim = Tshape.length;
        this._size = this.data.length;
        let stride = 1;
        let _stride:number[] = [];
        for(let i = this._ndim - 1;i >= 0;i--){
            _stride[i] = stride;
            stride *= this._shape[i]!;
        }
        this._strides = new Int32Array(_stride);
    }
    private flatten(data:Flatten):Data_Shape{
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
        return {Tdata:Tdata as number[],Tshape:shape};
    }
    get data():Float64Array{
        return this._data;
    }
    get ndim():number{
        return this._shape.length;
    }
    get size():number{
        return this._data.length;
    }
    get strides():Int32Array{
        return this._strides;
    }
    get shape():Int32Array{
        return this._shape;
    }
    static add(tensor_one:Tensor,tensor_two:Tensor){
        return tensor_c.add(tensor_one,tensor_two,Tensor);
    }
    static sub(tensor_one:Tensor,tensor_two:Tensor){
        return tensor_c.sub(tensor_one,tensor_two,Tensor);
    }
    static max(tensor:Tensor,axis:number[],keepdims:boolean){
        return tensor_c.max(tensor,axis,keepdims,Tensor);
    }
    static min(tensor:Tensor,axis:number[],keepdims:boolean){
        return tensor_c.min(tensor,axis,keepdims,Tensor);
    }
    static reset_pool():void{
        tensor_c.reset_tensor_pool();
    }
    static destory_pool():void{
        tensor_c.destory_tensor_pool();
    }
    static create_pool():void{
        tensor_c.create_pool();
    }
    static div(parameter_one:Tensor|number,parameter_two:Tensor|number):Tensor|never{
        if(parameter_one instanceof Tensor && typeof parameter_two == "number"){
            return tensor_c.div_scalar(parameter_one,parameter_two,Tensor);
        }
        if(parameter_one instanceof Tensor && parameter_two instanceof Tensor){
            return tensor_c.tensor_div_tensor(parameter_one,parameter_two,Tensor);
        }
        if(typeof parameter_one == "number" && parameter_two instanceof Tensor){
            return tensor_c.scalar_div_tensor(parameter_two,parameter_one,Tensor);
        }else{
            throw new Error("[ERROR]:The two parameters cannot both be 'number'\n");
        }
    }
    static mul(parameter_one:Tensor,parameter_two:Tensor|number){
        if(typeof parameter_two == "number"){
            return tensor_c.mul_scalar(parameter_one,parameter_two,Tensor);
        }else{
            return tensor_c.mul_elementwise(parameter_one,parameter_two,Tensor);
        }
    }
    static Matrix_mul(tensor_one:Tensor,tensor_two:Tensor){
        return tensor_c.mat(tensor_one,tensor_two,Tensor);
    }
    static add_broadcasted(tensor_one:Tensor,tensor_two:Tensor){
        return tensor_c.add_broadcasted(tensor_one,tensor_two,Tensor);
    }
    static sub_broadcasted(tensor_one:Tensor,tensor_two:Tensor){
        return tensor_c.sub_broadcasted(tensor_one,tensor_two,Tensor);
    }
    static reshape(tensor:Tensor,new_shape:number[]){
        let shape = new Int32Array(new_shape);
        return tensor_c.reshape(tensor,shape,Tensor);
    }
    static pow(parameter_one:Tensor|number,parameter_two:Tensor|number){
        if(parameter_one instanceof Tensor && typeof parameter_two == "number"){
            return tensor_c.pow_scalar(parameter_one,parameter_two,Tensor);
        }
        if(typeof parameter_one == "number" && parameter_two instanceof Tensor){
            return tensor_c.pow_tensor(parameter_two,parameter_one,Tensor);
        }else{
            throw new Error("[ERROR]:The parameter can't be both a 'Tensor' and a 'number' at the same time\n");
        }
    }
    static ln(tensor:Tensor){
        return tensor_c.ln(tensor,Tensor);
    }
    static equal(tensor_one:Tensor,tensor_two:Tensor){
        return tensor_c.equal(tensor_one,tensor_two,Tensor);
    }
    static euqal_broadcasted(tensor_one:Tensor,tensor_two:Tensor){
        return tensor_c.equal_broadcasted(tensor_one,tensor_two,Tensor);
    }
    static ones_like(tensor:Tensor){
        return tensor_c.ones_like(tensor,Tensor);
    }
    static zeros_like(tensor:Tensor){
        return tensor_c.zeros_like(tensor,Tensor);
    }
    static sin(tensor:Tensor){
        return tensor_c.sins(tensor,Tensor);
    }
    static cos(tensor:Tensor){
        return tensor_c.cos(tensor,Tensor);
    }
    static transpose(tensor:Tensor,axis:number[]){
        return tensor_c.transpose(tensor,axis,axis.length,Tensor);
    }
    static make_contiguous(tensor:Tensor){
        return tensor_c.make_contiguous(tensor);
    }
    add(tensor:Tensor){
        return tensor_c.add(this,tensor,Tensor);
    }
    sub(tensor:Tensor){
        return tensor_c.sub(this,tensor,Tensor);
    }
    max(axis:number[],keepdims:boolean){
        return tensor_c.max(this,axis,keepdims,Tensor);
    }
    min(axis:number[],keepdims:boolean){
        return tensor_c.min(this,axis,keepdims,Tensor);
    }
    div(parameter:Tensor|number,S_div_T:boolean = false):Tensor|never{
        if(typeof parameter == "number"){
            return tensor_c.div_scalar(this,parameter,Tensor);
        }
        if(parameter instanceof Tensor){
            return tensor_c.tensor_div_tensor(this,parameter,Tensor);
        }
        if(S_div_T && typeof parameter == "number"){
            return tensor_c.scalar_div_tensor(this,parameter,Tensor);
        }else{
            throw new Error("[ERROR]:The parameter cannot  be 'Tensor' when S_div_T equal false\n");
        }
    }
    mul(parameter:Tensor|number){
        if(typeof parameter == "number"){
            return tensor_c.mul_scalar(this,parameter,Tensor);
        }else{
            return tensor_c.mul_elementwise(this,parameter,Tensor);
        }
    }
    add_broadcasted(tensor:Tensor){
        return tensor_c.add_broadcasted(this,tensor,Tensor);
    }
    sub_broadcasted(tensor:Tensor){
        return tensor_c.sub_broadcasted(this,tensor,Tensor);
    }
    reshape(new_shape:number[]){
        let shape = new Int32Array(new_shape);
        return tensor_c.reshape(this,shape,Tensor);
    }
    pow(parameter:number,T_pow_S:boolean = false){
        if(T_pow_S){
            return tensor_c.pow_scalar(this,parameter,Tensor);
        }else{
            return tensor_c.pow_tensor(this,parameter,Tensor);
        }
    }
    ln(){
        return tensor_c.ln(this,Tensor);
    }
    equal(tensor:Tensor){
        return tensor_c.equal(this,tensor,Tensor);
    }
    equal_broadcasted(tensor:Tensor){
        return tensor_c.equal_broadcasted(this,tensor,Tensor);
    }
    ones_like(){
        return tensor_c.ones_like(this,Tensor);
    }
    zeros_like(){
        return tensor_c.zeros_like(this,Tensor);
    }
    sin(){
        return tensor_c.sins(this,Tensor);
    }
    cos(){
        return tensor_c.cos(this,Tensor);
    }
    transpose(axis:number[]){
        return tensor_c.transpose(this,axis,axis.length,Tensor);
    }
    make_contiguous(){
        return tensor_c.make_contiguous(this);
    }
}
Tensor.create_pool();
let a = new Tensor([[1.0,2.0],[0.3,3.0]]);
let b = new Tensor([[1.0,3.0],[2.0,3.0]]);
let c = Tensor.Matrix_mul(a,b);
console.log(c)
Tensor.destory_pool();
