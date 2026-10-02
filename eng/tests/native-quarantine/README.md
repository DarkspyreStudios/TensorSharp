# Managed Native Quarantine Authority

This harness loads actual compiled Core and fixture assemblies in distinct collectible
ALCs. It proves managed admission, exact-cause identity, failed owner/storage retention,
healthy retirement, and process-shared lock/schema behavior. It does not load native
libraries or instantiate actual backend models. CUDA contexts, streams and modules use the authority.
The other CUDA/MLX/GGML callers do not yet use it. These Core tests do not qualify
backend caller coverage, native release or hardware safety.

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

`resolved-cuda-nesting` admits known-device work under an already-held unresolved CUDA
write gate. It rejects device widening, write upgrades and unrelated MLX entry.
`queued-safe-release` distinguishes blocked reservations from executed effects.
`foreign-queued-safe-release` executes the queued registration through a second actual
Core generation, rejects its effect after safe owner removal, drains the BCL frames,
and collects both foreign generations. `safe-release-negatives` rejects nested release
before its controlled free callback and rejects wrong-owner, forged-gate, wrong-thread
and disposed leases. Safe release validates every frozen held gate before teardown;
blocked reservations do not imply that native work has executed.

`finalizer-origin` runs a genuine foreign fixture finalizer without a static fixture
root. The actual registration uses a resurrection-tracking long weak reference.
Publication retains the authenticated finalizing owner and its actual managed resource.
After finalizer drainage, a fixture-only accessor proves executing fixture/Core MVIDs
and collectible provenance. Short weak `RuntimeAssembly` wrapper collection does not
prove owner or loader retirement. `finalizer-healthy` explicitly completes safe release,
suppresses finalization, and collects actual owner/resource/ALC references. These cases
exercise managed authority behavior, not a device allocation or native free failure.

Per-mode reports record actual fixture/Core hashes and MVID. Test public-method
reflection is used solely to execute the compiled foreign fixture; there is no
private-field reflection into shipping code or reflective Inference observation seam.
