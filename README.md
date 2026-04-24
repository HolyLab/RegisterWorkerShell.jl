# RegisterWorkerShell

`RegisterWorkerShell` defines the abstract interface for image-registration worker
algorithms in the HolyLab registration pipeline. It is not a registration algorithm
itself; rather, it provides the scaffolding that concrete worker packages build on.

## Implementing a worker

Subtype `AbstractWorker` and add a `workertid` field (or override `workertid`):

```julia
using RegisterWorkerShell

mutable struct MyWorker <: AbstractWorker
    workertid::Int
    # algorithm parameters...
end
```

Implement `worker`, which is called once per time-point:

```julia
function RegisterWorkerShell.worker(alg::MyWorker, img, tindex, mon)
    slice = getindex_t(img, tindex)
    # ... perform registration on slice ...
    monitor!(mon, alg)   # copy monitored fields back to driver
    return result
end
```

Optionally specialize `init!` and `close!` for setup/teardown:

```julia
RegisterWorkerShell.init!(alg::MyWorker, args...) = # allocate buffers, etc.
RegisterWorkerShell.close!(alg::MyWorker, args...) = # release resources, etc.
```

All four functions (`worker`, `init!`, `close!`, `load_mm_package`) also accept a
`RemoteChannel` holding a worker, forwarding the call to the wrapped object on the
remote process.

## Monitoring results

The driver selects which fields to stream back from each worker using `monitor`:

```julia
mon = monitor(alg, (:shift, :mismatch))
```

After each call to `worker`, the results land in `mon`. Inside your `worker`
implementation, copy fields into `mon` with `monitor!`:

```julia
monitor!(mon, alg)             # copies all fields that appear in mon
monitor!(mon, :residual, val)  # copies a single computed quantity
```

Use `haskey(mon, :key)` to skip expensive computations when a field is not
being monitored.

## Shared arrays

`maybe_sharedarray` handles allocation of arrays that may need to be visible to
a remote worker process:

```julia
# Wrap an existing array — returns a SharedArray if pid is remote, else A itself
S = maybe_sharedarray(A, pid)

# Allocate by type and size
S = maybe_sharedarray(Float64, (256, 256, 64), pid)

# Allocate from an ArrayDecl (deferred array specification)
adcl = ArrayDecl(Array{Float32, 3}, (256, 256, 64))
S = maybe_sharedarray(adcl, pid)
```

Non-bits-type arrays and non-array objects are returned unchanged.
