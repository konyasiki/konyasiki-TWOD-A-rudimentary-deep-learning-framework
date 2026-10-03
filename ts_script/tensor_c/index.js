import { createRequire } from "module";
const node_require = createRequire(import.meta.url);
const tensor_c  = node_require("../../csrc/build/Release/tensor.node");
export default tensor_c; 