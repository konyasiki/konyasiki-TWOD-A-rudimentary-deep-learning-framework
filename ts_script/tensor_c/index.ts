import { createRequire } from "module";
import type { tensor_c_type } from "./tensorType.ts";
const require = createRequire(import.meta.url);
const tensor_c  = require("./build/Release/tensor.node");
export default tensor_c as tensor_c_type;