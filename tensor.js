import { createRequire } from "module";
const require = createRequire(import.meta.url);
const tensor_c = require("./build/Release/tensor.node");
export default tensor_c;