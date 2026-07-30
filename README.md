# TWOD-A-rudimentary-deep-learning-framework

## How to initialize a tensor?
```ts
let tensor = new Tensor([[1.0,2.0],[0.3,3.0]]);//The default data type is Float64Array
```
#### tip:
You can call 👇 before initializing the tensor.This way you can use a memory pool to avoid allocating memory frequently.
```ts
Tensor.create_pool();
```
But after creating the memory pool, please destroy it using 👇 to prevent memory leaks.
```ts
Tensor.destory_pool();
```

## The algorithm ideas for some tensor operations in this project are included inProject Ideas & Insights.md

### 张量广播
```C
void cpu_tensor_broadcasted_add(Tensor *tensor_one,Tensor *tensor_two,float *data,int *broadcasted_shape,int *broadcasted_size);
```
```
例如 数组a[] = [[1],
                [2],
                [3]](3,1)
        b[] = [[0,1]](1,2)
- 无法逐元素计算

- 张量形状相同时可进行逐元素计算，但不同时可以通过广播执行
a + b形状不同 => a广播 => a(3,2) [[1,1],
                                 [2,2],
                                 [3,3]]
同理 b广播 => b(3,2)[[0,1],
                    [0,1],
                    [0,1]]
a + b(broadcasted) =  [[1,2],
                       [2,3],
                       [3,4]]
```
![广播条件](./tensor_broadcasted.png)
```
从最右边的维度开始比较，若是相等则可以直接匹配
其中一个为1也可以=>直接拓展到最大维度
例:
    a(3,1,5) => 逐维度比较 => 5 = 5  √ a => broadcasted (3,4,5)
    b(1,4,5)                 1 != 4 √ b => broadcasted (3,4,5)
                             3 != 1 √
    c(1,2,3) => 逐维度比较 3 != 2 × 不可广播
    d(3,2,2)               
    c[[[1,3,4],
        [2,3,5]]]
    d[[[2,3],[3,5]],[[4,4],[6,5]],
      [[4,3],[3,5]],[[4,4],[0,5]]
      [[2,3],[5,5]],[[6,4],[6,5]]]
    主要看最后两位(2,3)的矩阵如何补都无法和(2,2)的矩阵相加
```
```c
void cpu_tensor_broadcasted_add(Tensor *tensor_one,Tensor *tensor_two,float *data,int *broadcasted_shape,int broadcasted_size){
    int max_ndim = tensor_one->dimension > tensor_two->dimension ? tensor_one->dimension : tensor_two->dimension;//计算最大的维度
    /**
     * tensor_one(4,3,5,6);
     * tensor_two(3,5,6) => (4,3,5,6)
     */
    int *strides_one = (int*)malloc(max_ndim * sizeof(int));
    int *strides_two = (int*)malloc(max_ndim * sizeof(int));//补齐维度
    if(strides_one == NULL || strides_two == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    int stride_one = 1;
    int stride_two = 1;
    for(int i = max_ndim - 1;i >= 0;i--){
        int dim_one = (i < tensor_one->dimension & tensor_one->dimension - max_ndim + i >= 0) ? tensor_one->shape[tensor_one->dimension - max_ndim + i] : 1;
        int dim_two = (i < tensor_two->dimension & tensor_two->dimension - max_ndim + i >= 0) ? tensor_two->shape[tensor_two->dimension - max_ndim + i] : 1;
        strides_one[i] = dim_one == broadcasted_shape[i] ? stride_one : 0;//index 3 2 1 0 //当形状为一时为了不增加内存则将步长设置为0
        strides_two[i] = dim_two == broadcasted_shape[i] ? stride_two : 0;
        stride_one *= (dim_one == broadcasted_shape[i] ? dim_one : 1);//接上 此维度步长为0则保持步长不变
        stride_two *= (dim_two == broadcasted_shape[i] ? dim_two : 1);
    }
    for(int i = 0;i < broadcasted_size;i++){//计算线性索引 
        int index_one = 0;
        int index_two = 0;
        int linear_index = i;// 0
        for(int j = max_ndim - 1;j >=0;j--){
            int pos = linear_index % broadcasted_shape[j];//各维度不能整除的偏移量以上一维度的步长相乘再相加即可得到线性索引
            linear_index /= broadcasted_shape[j];//各维度相对于上维度的偏移量
            if(strides_one[j] != 0) index_one += pos * strides_one[j];
            if(strides_two[j] != 0) index_two += pos * strides_two[j];
        }
        data[i] = tensor_one->data[index_one] + tensor_two->data[index_two];//
    }
    free(strides_one);
    free(strides_two);
}
```
