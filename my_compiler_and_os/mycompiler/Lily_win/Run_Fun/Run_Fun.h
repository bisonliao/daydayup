#if !defined(__RUN_FUN_H_INCLUDED__)
#define  __RUN_FUN_H_INCLUDED__

#ifdef _MAKE_DLL
#define DLL_SPEC __declspec(dllexport) 
#else
#define DLL_SPEC __declspec(dllimport) 
#endif

extern "C" DLL_SPEC int Run_Fun(int funnum, int argc, Variable* argv);
#endif
