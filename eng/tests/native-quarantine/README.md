# Managed Native Quarantine Authority

This harness loads actual compiled Core and fixture assemblies in distinct collectible
ALCs. It proves managed admission, exact-cause identity, failed owner/storage retention,
healthy retirement, and process-shared lock/schema behavior. It does not load native
libraries or instantiate actual backend models. CUDA/MLX/GGML caller integrations are
INERT/unimplemented; these tests do not qualify native release or hardware safety.

Build `native-quarantine.csproj` with all three native hooks disabled, then run:

```bash
python3 eng/tests/run-native-quarantine.py --configuration Debug --output tmp/quarantine/debug
python3 eng/tests/run-native-quarantine.py --configuration Release --output tmp/quarantine/release
```

Build and execute each configuration sequentially because Core uses a shared output
directory. The runner supplies all native-skip properties, does not build, and runs
each terminal/schema case in a fresh process with a 20-second bound. No slot reset is
available. Protocol-corruption cases affect only their test process.

`healthy` uses outer-frame weak references after safe explicit completion. Promotion
cases verify actual foreign owner/storage/Core/ALC roots survive finalizer drainage
while an independent safely released generation collects. `constructor` invokes an
actual managed fixture constructor that reserves ownership before throwing an original
plus cleanup aggregate; it is not an actual TensorSharp model constructor. `race`
observes the contender's reserved BCL frame before publishing through the active lease,
then proves zero managed effect entry. `queued` checks again on the executing thread.
`lease-negatives` rejects owner/thread/nesting/disposed/scope/upgrade violations before
mutation. `mismatched-lease` rejects a token with another registration's actual active
frame. `cross-nesting` proves the same nesting rules across distinct real Core ALCs.
`roles` checks all six fixed resource roles. `lock-order` holds metadata and AppDomain
on separate threads/ALCs behind barriers and uses a bounded top-level assertion; on
failure the fresh process exits without pretending to unlock abandoned gates. No
numeric lease count is used as proof of freeing a native buffer.

Per-mode reports record actual fixture/Core hashes and MVID. Test public-method
reflection is used solely to execute the compiled foreign fixture; there is no
private-field reflection into shipping code or reflective Inference observation seam.
