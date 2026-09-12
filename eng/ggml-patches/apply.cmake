# Small upstream correctness fixes needed by TensorSharp. Keep these as normal
# unified diffs so they can be reviewed, submitted upstream, and removed once
# the fetched revision contains the fix. Never silently build a partial patch.
find_package(Git REQUIRED)
set(_tsg_ggml_source "${CMAKE_CURRENT_LIST_DIR}/../../ExternalProjects/ggml")
set(_tsg_precision_patch "${CMAKE_CURRENT_LIST_DIR}/0001-cuda-honor-f32-matmul-precision.patch")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_tsg_precision_patch}")
file(LOCK "${_tsg_ggml_source}/../.ggml-patches.lock" GUARD FILE TIMEOUT 60)
execute_process(COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${_tsg_precision_patch}"
    WORKING_DIRECTORY "${_tsg_ggml_source}" RESULT_VARIABLE _tsg_applied
    OUTPUT_QUIET ERROR_QUIET)
if (NOT _tsg_applied EQUAL 0)
    execute_process(COMMAND "${GIT_EXECUTABLE}" apply --check "${_tsg_precision_patch}"
        WORKING_DIRECTORY "${_tsg_ggml_source}" RESULT_VARIABLE _tsg_can_apply
        OUTPUT_QUIET ERROR_VARIABLE _tsg_patch_error)
    if (NOT _tsg_can_apply EQUAL 0)
        message(FATAL_ERROR "ggml CUDA precision fix conflicts with the fetched revision: ${_tsg_patch_error}")
    endif()
    execute_process(COMMAND "${GIT_EXECUTABLE}" apply "${_tsg_precision_patch}"
        WORKING_DIRECTORY "${_tsg_ggml_source}" RESULT_VARIABLE _tsg_patch_result
        ERROR_VARIABLE _tsg_patch_error)
    if (NOT _tsg_patch_result EQUAL 0)
        message(FATAL_ERROR "Cannot apply ggml CUDA precision fix: ${_tsg_patch_error}")
    endif()
endif()
