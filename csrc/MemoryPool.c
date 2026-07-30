#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include "MemoryPool.h"

MemoryPool *create_pool(size_t block_size,size_t block_count){
    if(block_count == 0||block_size == 0){
        return NULL;
    }

    size_t align_block_size = POOL_ALIGN(block_size,POOL_ALIGN_SIZE);
    if(align_block_size < sizeof(void *)){//块大小要大于等于指针域大小
        align_block_size = sizeof(void *);
    }

    MemoryPool *pool = (MemoryPool *)malloc(sizeof(MemoryPool));
    if(pool == NULL){
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }

    size_t total_size = align_block_size * block_count;
    pool->pool_address = (uint8_t *)malloc(total_size);
    if(pool->pool_address == NULL){
        free(pool);
        pool = NULL;
        fprintf(stderr,"[ERROR]:Memory allocation failed\n");
        exit(1);
    }
    
    pool->block_size = align_block_size;
    pool->block_count = block_count;
    pool->free_count = block_count;
    /**
     * 初始化空闲地址
     */
    pool->free_list = pool->pool_address;
    uint8_t *pointer = pool->pool_address;//首地址
    for (size_t i = 0; i < block_count - 1; i++) {
        *(void **)pointer = pointer + align_block_size;//转换为二级指针并指向原先后align_block_size字节的地址
        pointer += align_block_size;//下一个块的地址
    }
    /* 最后一个块的next置空 */
    *(void **)pointer = NULL;
    return pool;
}

void destory_pool(MemoryPool *pool){
    if(pool == NULL){
        fprintf(stderr,"[ERROR]:The parameter cannot be NULL\n");
        exit(1);
    }
    if(pool->pool_address != NULL){
        free(pool->pool_address);
        pool->pool_address = NULL;

    }
    free(pool);
}

void *alloc_pool(MemoryPool *pool){
    if(pool == NULL || pool->free_list == NULL){
        fprintf(stderr,"[ERROR]:The pool and free_list cannot be NULL\n");
        exit(1);
    }
    void *pointer = pool->free_list;
    pool->free_list = *(void **)pointer;
    pool->free_count--;

    memset(pointer,0,pool->block_size);//清空原先内存,避免污染数据

    return pointer;
}

void free_pool(MemoryPool *pool,void *block_ptr){
    if(pool == NULL || block_ptr == NULL){
        fprintf(stderr,"[ERROR]:The pool and block_ptr cannot be NULL\n");
        exit(1);
    }

    uint8_t *pointer = (uint8_t *)block_ptr;
    if(pointer < pool->pool_address || pointer > pool->pool_address + pool->block_size * pool->block_count){
        fprintf(stderr,"[ERROR]:The address of block_ptr is not within the range of the memory pool\n");
        exit(1);
    }

    *(void **)pointer = pool->free_list;
    pool->free_list = pointer;
    pool->free_count++;
}

size_t get_free_count(MemoryPool *pool){
    if(pool == NULL){
        fprintf(stderr,"[ERROR]:The parameter cannot be NULL\n");
        exit(1);
    }

    return pool->free_count;
}

void reset_pool(MemoryPool *pool){
    if(pool == NULL){
        fprintf(stderr,"[ERROR]:The parameter cannot be NULL\n");
        exit(1);
    }

    pool->free_count = pool->block_count;
    pool->free_list = pool->pool_address;
    uint8_t *pointer = pool->pool_address;

    for(size_t i = 0;i < pool->block_count - 1;i++){
        *(void **)pointer = pointer + pool->block_size;
        pointer += pool->block_size;
    }

    *(void **)pointer = NULL;
}