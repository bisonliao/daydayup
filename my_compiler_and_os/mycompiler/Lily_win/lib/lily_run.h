#if !defined(__LILY_RUN_H_INCLUDED__)
#define __LILY_RUN_H_INCLUDED__

#if defined(_MAKE_DLL)
	#define DLL_SPEC  __declspec(dllexport) 
#else
	#define DLL_SPEC  __declspec(dllimport) 
#endif

extern "C" DLL_SPEC int ExecuteFile(const char * filename, 
									const char* InputBuffer, unsigned int InputBufferSize,
									char* OutputBuffer, unsigned int * OutputBufferSize,
									char * errmsg);
#endif
