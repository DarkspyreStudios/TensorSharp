// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Deliberately exports no cuDNN functions. The regression places this DLL and
// sibling-name copies in an isolated build fixture to exercise loader failure.
extern "C" __declspec(dllexport) int TensorSharpMissingCudnnFixture() { return 1; }
