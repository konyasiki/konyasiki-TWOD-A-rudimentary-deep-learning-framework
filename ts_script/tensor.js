"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Tensor = void 0;
// import { createRequire } from "node:module";
// const require = createRequire(import.meta.url);
// const tensor_c = require("tensor_c");
var tensor_c = require("tensor_c");
var Tensor = /** @class */ (function () {
    function Tensor(data) {
        var _a = this.flatten(data), Tdata = _a.Tdata, Tshape = _a.Tshape;
        this._shape = new Int32Array(Tshape);
        this._data = new Float64Array(Tdata);
        this._ndim = Tshape.length;
        this._size = this.data.length;
        var stride = 1;
        var _stride = [];
        for (var i = this._ndim - 1; i >= 0; i--) {
            _stride[i] = stride;
            stride *= this._shape[i];
        }
        this._strides = new Int32Array(_stride);
    }
    Tensor.prototype.flatten = function (data) {
        var Tdata = [];
        var shape = [];
        Tdata = data; //root
        var length = Tdata.length;
        shape.push(Tdata.length);
        var _loop_1 = function () {
            length--;
            if (Tdata[0] instanceof Array) {
                var Data_1 = Tdata.shift();
                if (Data_1 instanceof Array) {
                    Data_1.forEach(function (value) {
                        Tdata.push(value);
                        if (length == 0) {
                            shape.push(Data_1.length);
                            length = Data_1[0] instanceof Array ? Data_1.length : null;
                        }
                    });
                }
            }
            else {
                return "break";
            }
        };
        while (Tdata.length > 0) {
            var state_1 = _loop_1();
            if (state_1 === "break")
                break;
        }
        return { Tdata: Tdata, Tshape: shape };
    };
    Object.defineProperty(Tensor.prototype, "data", {
        get: function () {
            return this._data;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Tensor.prototype, "ndim", {
        get: function () {
            return this._shape.length;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Tensor.prototype, "size", {
        get: function () {
            return this._data.length;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Tensor.prototype, "strides", {
        get: function () {
            return this._strides;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Tensor.prototype, "shape", {
        get: function () {
            return this._shape;
        },
        enumerable: false,
        configurable: true
    });
    Tensor.add = function (tensor_one, tensor_two) {
        return tensor_c.T_add(tensor_one, tensor_two, Tensor);
    };
    Tensor.sub = function (tensor_one, tensor_two) {
        return tensor_c.T_sub(tensor_one, tensor_two, Tensor);
    };
    Tensor.max = function (tensor, axis, keepdims) {
        return tensor_c.T_max(tensor, axis, keepdims, Tensor);
    };
    Tensor.min = function (tensor, axis, keepdims) {
        return tensor_c.T_min(tensor, axis, keepdims, Tensor);
    };
    Tensor.reset_pool = function () {
        return tensor_c.node_reset_tensor_pool();
    };
    Tensor.destory_pool = function () {
        return tensor_c.node_destory_tensor_pool();
    };
    Tensor.div = function (parameter_one, parameter_two) {
        if (parameter_one instanceof Tensor && typeof parameter_two == "number") {
            return tensor_c.T_div_S(parameter_one, parameter_two, Tensor);
        }
        if (parameter_one instanceof Tensor && parameter_two instanceof Tensor) {
            return tensor_c.T_div_T(parameter_one, parameter_two, Tensor);
        }
        if (typeof parameter_one == "number" && parameter_two instanceof Tensor) {
            return tensor_c.S_div_T(parameter_two, parameter_one, Tensor);
        }
        else {
            throw new Error("[ERROR]:The two parameters cannot both be 'number'\n");
        }
    };
    Tensor.mul = function (parameter_one, parameter_two) {
        if (typeof parameter_two == "number") {
            return tensor_c.S_mul_T(parameter_one, parameter_two, Tensor);
        }
        else {
            return tensor_c.T_mul_T_ele(parameter_one, parameter_one, Tensor);
        }
    };
    Tensor.Matrix_mul = function (tensor_one, tensor_two) {
        return tensor_c.T_matul_T(tensor_one, tensor_two, Tensor);
    };
    Tensor.add_broadcasted = function (tensor_one, tensor_two) {
        return tensor_c.T_add_broadcasted(tensor_one, tensor_two, Tensor);
    };
    Tensor.sub_broadcasted = function (tensor_one, tensor_two) {
        return tensor_c.T_sub_broadcasted(tensor_one, tensor_two, Tensor);
    };
    Tensor.reshape = function (tensor, new_shape, new_dim) {
        return tensor_c.reshape(tensor, new_shape, new_dim, Tensor);
    };
    Tensor.pow = function (parameter_one, parameter_two) {
        if (parameter_one instanceof Tensor && typeof parameter_two == "number") {
            return tensor_c.T_pow(parameter_one, parameter_two, Tensor);
        }
        if (typeof parameter_one == "number" && parameter_two instanceof Tensor) {
            return tensor_c.S_pow_T(parameter_two, parameter_one, Tensor);
        }
        else {
            throw new Error("[ERROR]:The parameter can't be both a 'Tensor' and a 'number' at the same time\n");
        }
    };
    Tensor.ln = function (tensor) {
        return tensor_c.ln(tensor, Tensor);
    };
    Tensor.equal = function (tensor_one, tensor_two) {
        return tensor_c.equal(tensor_one, tensor_two, Tensor);
    };
    Tensor.euqal_broadcasted = function (tensor_one, tensor_two) {
        return tensor_c.T_equal_broadcasted(tensor_one, tensor_two, Tensor);
    };
    Tensor.ones_like = function (tensor) {
        return tensor_c.ones(tensor, Tensor);
    };
    Tensor.zeros_like = function (tensor) {
        return tensor_c.zeros(tensor, Tensor);
    };
    Tensor.sin = function (tensor) {
        return tensor_c.sinss(tensor, Tensor);
    };
    Tensor.cos = function (tensor) {
        return tensor_c.coss(tensor, Tensor);
    };
    Tensor.transpose = function (tensor, axis) {
        return tensor_c.transpose(tensor, axis, axis.length, Tensor);
    };
    Tensor.make_contiguous = function (tensor) {
        return tensor_c.contiguous(tensor);
    };
    Tensor.prototype.add = function (tensor) {
        return tensor_c.T_add(this, tensor, Tensor);
    };
    Tensor.prototype.sub = function (tensor) {
        return tensor_c.T_sub(this, tensor, Tensor);
    };
    Tensor.prototype.max = function (axis, keepdims) {
        return tensor_c.T_max(this, axis, keepdims, Tensor);
    };
    Tensor.prototype.min = function (axis, keepdims) {
        return tensor_c.T_min(this, axis, keepdims, Tensor);
    };
    Tensor.prototype.div = function (parameter, S_div_T) {
        if (S_div_T === void 0) { S_div_T = false; }
        if (typeof parameter == "number") {
            return tensor_c.T_div_S(this, parameter, Tensor);
        }
        if (parameter instanceof Tensor) {
            return tensor_c.T_div_T(this, parameter, Tensor);
        }
        if (S_div_T && typeof parameter == "number") {
            return tensor_c.S_div_T(this, parameter, Tensor);
        }
        else {
            throw new Error("[ERROR]:The parameter cannot  be 'Tensor' when S_div_T equal false\n");
        }
    };
    Tensor.prototype.mul = function (parameter) {
        if (typeof parameter == "number") {
            return tensor_c.S_mul_T(this, parameter, Tensor);
        }
        else {
            return tensor_c.T_mul_T_ele(this, parameter, Tensor);
        }
    };
    Tensor.prototype.add_broadcasted = function (tensor) {
        return tensor_c.T_add_broadcasted(this, tensor, Tensor);
    };
    Tensor.prototype.sub_broadcasted = function (tensor) {
        return tensor_c.T_sub_broadcasted(this, tensor, Tensor);
    };
    Tensor.prototype.reshape = function (new_shape, new_dim) {
        return tensor_c.reshape(this, new_shape, new_dim, Tensor);
    };
    Tensor.prototype.pow = function (parameter, S_pow_T) {
        if (S_pow_T === void 0) { S_pow_T = false; }
        if (S_pow_T) {
            return tensor_c.T_pow(this, parameter, Tensor);
        }
        else {
            return tensor_c.S_pow_T(this, parameter, Tensor);
        }
    };
    Tensor.prototype.ln = function () {
        return tensor_c.ln(this, Tensor);
    };
    Tensor.prototype.equal = function (tensor) {
        return tensor_c.equal(this, tensor, Tensor);
    };
    Tensor.prototype.equal_broadcasted = function (tensor) {
        return tensor_c.T_equal_broadcasted(this, tensor, Tensor);
    };
    Tensor.prototype.ones_like = function () {
        return tensor_c.ones(this, Tensor);
    };
    Tensor.prototype.zeros_like = function () {
        return tensor_c.zeros(this, Tensor);
    };
    Tensor.prototype.sin = function () {
        return tensor_c.sinss(this, Tensor);
    };
    Tensor.prototype.cos = function () {
        return tensor_c.sinss(this, Tensor);
    };
    Tensor.prototype.transpose = function (axis) {
        return tensor_c.transpose(this, axis, axis.length, Tensor);
    };
    Tensor.prototype.make_contiguous = function () {
        return tensor_c.contiguous(this);
    };
    return Tensor;
}());
exports.Tensor = Tensor;
