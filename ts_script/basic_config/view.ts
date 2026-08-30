import http from "http";
http.createServer((req,res)=>{
    let reqURL = req.url;
    res.write("1");
}).listen(1209);