#ifndef MEMORYPOOL_H
#define MEMORYPOOL_H
#define POOL_ALIGN(size,align) (((size) + (align) - 1) & ~((align) - 1))
#define POOL_ALIGN_SIZE 8
#include <stdint.h>
#include <stddef.h>

typedef struct MemoryPool{
    uint8_t *pool_address;
    size_t block_size;
    size_t block_count;
    void *free_list;//空闲块链表头
    size_t free_count;//空闲块数量
}MemoryPool;
/**
 * brief 创建固定块内存池
 * block_size 单个内存块大小
 * block_count 总内存块数量
 *
 */
MemoryPool *create_pool(size_t block_size,size_t block_count);

void destory_pool(MemoryPool *pool);
//销毁内存池

void *alloc_pool(MemoryPool *pool);
//分配块内存

void free_pool(MemoryPool *pool,void *block_ptr);
//归还块内存

size_t get_free_count(MemoryPool *pool);

void reset_pool(MemoryPool *pool);
//重置内存池
#endif