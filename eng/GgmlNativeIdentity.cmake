# Both this function and GgmlNativeIdentity.targets hash the normalized source
# manifest. The identity covers bridge implementations, managed interop and the
# pinned upstream revision, independent of Git checkout and display version.
function(tensorsharp_native_abi ROOT OUTPUT)
    file(GLOB INPUTS RELATIVE "${ROOT}"
        "${ROOT}/TensorSharp.GGML.Native/*.cpp"
        "${ROOT}/TensorSharp.GGML.Native/*.h"
        "${ROOT}/TensorSharp.GGML.Native/*.hpp"
        "${ROOT}/TensorSharp.GGML.Native/*.inc"
        "${ROOT}/TensorSharp.GGML.Native/*.cu"
        "${ROOT}/TensorSharp.GGML.Native/*.cuh"
        "${ROOT}/TensorSharp.Backends.GGML/*.cs")
    list(APPEND INPUTS "eng/ggml-revision")
    list(SORT INPUTS)
    set(MANIFEST "")
    foreach(INPUT IN LISTS INPUTS)
        file(READ "${ROOT}/${INPUT}" CONTENT)
        string(REPLACE "\r\n" "\n" CONTENT "${CONTENT}")
        string(SHA256 HASH "${CONTENT}")
        string(APPEND MANIFEST "${INPUT}=${HASH}\n")
        if(NOT CMAKE_SCRIPT_MODE_FILE)
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ROOT}/${INPUT}")
        endif()
    endforeach()
    string(SHA256 IDENTITY "${MANIFEST}")
    set(${OUTPUT} "${IDENTITY}" PARENT_SCOPE)
endfunction()
