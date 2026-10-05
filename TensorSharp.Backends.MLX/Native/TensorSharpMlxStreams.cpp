#include <exception>

#include "mlx/c/error.h"
#include "mlx/stream.h"

#if defined(_WIN32)
#define TENSORSHARP_MLX_EXPORT __declspec(dllexport)
#else
#define TENSORSHARP_MLX_EXPORT __attribute__((visibility("default")))
#endif

extern "C" TENSORSHARP_MLX_EXPORT int tensorsharp_mlx_synchronize_all_streams() {
  try {
    const auto streams = mlx::core::get_streams();
    for (const auto& stream : streams) {
      mlx::core::synchronize(stream);
    }
    return 0;
  } catch (const std::exception& error) {
    mlx_error("%s", error.what());
  } catch (...) {
    mlx_error("Unknown native failure while synchronizing MLX streams.");
  }
  return 1;
}
