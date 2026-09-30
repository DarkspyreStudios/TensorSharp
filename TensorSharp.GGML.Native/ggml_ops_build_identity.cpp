// Build identity of this GgmlOps binary.
//
// CMakeLists.txt passes the values as compile definitions. The managed native
// loader (TensorSharp.GGML.GgmlNativeLoader) reads the string after it loads a
// candidate and compares the TensorSharp build and RID with the candidate it
// asked for. The string is "key=value" pairs separated by ';' and uses only
// printable ASCII. Keys: format, tensorsharp, source, ggml, variant, rid, cpu.
#include "ggml_ops_internal.h"

#ifndef TSG_BUILD_TENSORSHARP_VERSION
#define TSG_BUILD_TENSORSHARP_VERSION "unknown"
#endif
#ifndef TSG_BUILD_SOURCE_COMMIT
#define TSG_BUILD_SOURCE_COMMIT "unknown"
#endif
#ifndef TSG_BUILD_GGML_COMMIT
#define TSG_BUILD_GGML_COMMIT "unknown"
#endif
#ifndef TSG_BUILD_VARIANT
#define TSG_BUILD_VARIANT "dev"
#endif
#ifndef TSG_BUILD_RID
#define TSG_BUILD_RID ""
#endif
#ifndef TSG_BUILD_CPU_PROFILE
#define TSG_BUILD_CPU_PROFILE "native"
#endif

TSG_EXPORT const char* TSGgml_GetBuildIdentity()
{
    return "format=1"
           ";tensorsharp=" TSG_BUILD_TENSORSHARP_VERSION
           ";source=" TSG_BUILD_SOURCE_COMMIT
           ";ggml=" TSG_BUILD_GGML_COMMIT
           ";variant=" TSG_BUILD_VARIANT
           ";rid=" TSG_BUILD_RID
           ";cpu=" TSG_BUILD_CPU_PROFILE;
}
