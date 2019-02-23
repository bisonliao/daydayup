#include "buffer.h"
#include "chash.h"


static THashHead hash;



static unsigned int fHash(const unsigned char * keybuf, size_t keysize) 
{
	return *(uint32_t*)keybuf;
}
static int fKeyEqual(const unsigned char * key1, size_t key1size, const unsigned char * key2, size_t key2size)
{
	uint32_t t1, t2;
	t1 = *(uint32_t*)key1;
	t2 = *(uint32_t*)key2;

	return t1== t2;
}

void buffer_init(uint32_t buffer_start, uint32_t buffer_size)
{
	if (hash_init( &hash, 1024, sizeof(uint32_t), sizeof(TBuffer), 
					fHash, 
					fKeyEqual,
					NULL))
	{
		panic("%s:%d:hash_init() failed!", __FILE__, __LINE__);
	}
	if (hash_mem_attach(&hash, (void*)buffer_start, buffer_size, 1))
	{
		panic("%s:%d:hash_mem_attach() failed!", __FILE__, __LINE__);
	}
}
