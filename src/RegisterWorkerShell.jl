module RegisterWorkerShell

using SimpleTraits, ImageAxes, ImageMetadata, Distributed, SharedArrays
using AxisArrays: AxisArray, Axis

export AbstractWorker, AnyValue, ArrayDecl, close!, init!, maybe_sharedarray, monitor, monitor!, worker, workerpid, getindex_t
export load_mm_package

"""
An `AbstractWorker` type performs registration on a single
image. Aside from the "moving" image (see `worker` for how this is
specified), all inputs/parameters to the algorithm should be supplied
via fields of an object `algorithm` which is a subtype of
`AbstractWorker`.

See `RegisterWorkerShell` for an overview of the API supported by
`AbstractWorker` types.
"""
abstract type AbstractWorker end

# Not sure about this next type...
struct ArrayDecl{A<:AbstractArray,N}
    arraysize::NTuple{N,Int}
end
ArrayDecl(::Type{A}, sz) where {A<:AbstractArray} = ArrayDecl{A,ndims(A)}(sz)

Base.eltype(::ArrayDecl{A}) where {A} = eltype(A)

"""
# RegisterWorkerShell

This module defines the core operations for all `AbstractWorker`
subtypes.  The exported operations are:

  - `monitor` and `monitor!`: passing results from worker(s) to the driver
  - `init!` and `close!`: functions you may specialize if your algorithm
    needs to initialize or clean up resources
  - `worker`: perform registration on an image
  - `workerpid`: extract the process-id for a given worker
"""
RegisterWorkerShell

"""
`mon = monitor(algorithm, (:var1, :var2, ...))` turns on "monitoring"
(reporting) for fields named `:var`, `:var2`, ... in `algorithm`. This
causes results to be passed back to the driver algorithm, which will
then save the values to disk. If `algorithm` is a Vector of algorithm
objects, then one `mon` object is created for each.

`mon = monitor(algorithm, (:var1, :var2), Dict(:var3=>value3, ...))`
monitors additional "internal" variables in the algorithm, as long as
the worker algorithm has been set up to look for these entries in
`mon`. You can safely choose 0 as the value for all `var`s; however,
large arrays of bitstypes may benefit from being provided as a full
instance of the proper type and size (see below about SharedArrays).

The worker algorithm should call `monitor!(mon, algorithm)` to copy
the values into `mon`, and `monitor!(mon, :var3, var3)` for an
internal variable `var3` that is not taken from `algorithm`. See
`monitor!` for more detail.

An important detail is that if `workerpid(algorithm) ≠ myid()`, then any
requested `AbstractArray` fields in `algorithm` will be turned into
`SharedArray`s for `mon`. This reduces the cost of communication
between the worker and driver processes.
"""
function monitor(algorithm::AbstractWorker, fields::Union{NTuple{N,Symbol},Vector{Symbol}}, morevars::Dict{Symbol} = Dict{Symbol,Any}()) where N
    pid = workerpid(algorithm)
    mon = Dict{Symbol,Any}()
    for f in fields
        isdefined(algorithm, f) || continue
        mon[f] = maybe_sharedarray(getfield(algorithm, f), pid)
    end
    for (k,v) in morevars
        mon[k] = maybe_sharedarray(v, pid)
    end
    mon
end

monitor(algorithm::Vector{W}, fields, morevars::Dict{Symbol} = Dict{Symbol,Any}()) where {W<:AbstractWorker} = map(alg->monitor(alg, fields, morevars), algorithm)

"""
`monitor!(mon, algorithm)` updates `mon` with the current values of
the fields in `algorithm`.  Workers should call this after all
computations have finished.  See `monitor` for more information.

`monitor!(mon, :parameter, algorithm)` copies just the value of
`algorithm.parameter`, after first checking `haskey(mon, :parameter)`.

One can check whether certain parameters are being request using
`haskey(mon, :parameter)`. This might be wise if computation of
`parameter` is non-essential and time consuming.
"""
function monitor!(mon::Dict{Symbol}, algorithm::AbstractWorker)
    for f in fieldnames(typeof(algorithm))
        monitor!(mon, f, getfield(algorithm, f))
    end
    mon
end

function monitor!(mon, fn::Symbol, v::AbstractArray)
    if haskey(mon, fn)
        if isa(mon[fn], AbstractArray) && size(mon[fn]) == size(v)
            copyto!(mon[fn], v)
        else
            mon[fn] = v
        end
    end
    mon
end

function monitor!(mon, fn::Symbol, v)
    if haskey(mon, fn)
        mon[fn] = v
    end
    mon
end

"""
`init!(algorithm)` performs any necessary initialization prior to
beginning a registration sequence using algorithm `algorithm`. The
default action is to return `nothing`. If you require initialization,
specialize this function for your `AbstractWorker` subtype.
"""
init!(algorithm::AbstractWorker, args...) = nothing

init!(rr::RemoteChannel, args...) = init!(fetch(rr), args...)

"""
`close!(algorithm)` performs any necessary cleanup after a
registration sequence using algorithm `algorithm`. The
default action is to return `nothing`. If you require cleanup,
specialize this function for your `AbstractWorker` subtype.
"""
close!(algorithm::AbstractWorker, args...) = nothing

close!(rr::RemoteChannel, args...) = close!(fetch(rr), args...)

"""
`worker(algorithm, img, tindex, mon)` causes registration to be performed
using the algorithm and parameters defined by `algorithm`, a subtype of
`AbstractWorker`.  Registration is performed on `img["t", tindex]`.
`mon` should be a `Dict(sym=>value)` that chooses the
outputs/variables to be monitored; see `monitor` for details.

You must define this function for your `AbstractWorker` subtype.
"""
worker(algorithm::AbstractWorker, img, tindex, mon) = error("Worker modules must define `worker`")

worker(rr::RemoteChannel, img, tindex, mon) = worker(fetch(rr), img, tindex, mon)

"""
`workerpid(algorithm)` extracts the `pid` associated with the worker
that will be assigned tasks for `algorithm`.  All `AbstractWorker`
subtypes should include a `workerpid` field, or overload this function
to return myid().
"""
workerpid(w::AbstractWorker) = w.workerpid

"""
`load_mm_package(dev)` loads appropriate mismatch module conditioned on
using cuda device.
"""
load_mm_package(dev, args...) = nothing

load_mm_package(rr::RemoteChannel, args...) = load_mm_package(fetch(rr), args...)


## Utility functions
function maybe_sharedarray(A::AbstractArray, pid::Int=myid())
    if pid != myid() && isbitstype(eltype(A))
        S = SharedArray{eltype(A)}(size(A), pids=union(myid(), pid))
        copyto!(S, A)
    else
        S = A
    end
    S
end

function maybe_sharedarray(::Type{T}, sz::Dims, pid=myid()) where T
    if isbitstype(T)
        S = SharedArray{T}(sz, pids=union(myid(), pid))
    else
        S = Array{T}(undef, sz)
    end
    S
end

maybe_sharedarray(adcl::ArrayDecl, pid::Int=myid()) =
    maybe_sharedarray(eltype(adcl), adcl.arraysize, pid)

maybe_sharedarray(obj, pid::Int = myid()) = obj

"""
    getindex_t(img, tindex)

Take a time-slice of `img` at time `tindex`. If `img` doesn't have a
`:time` axis, this just returns `img`.
"""
getindex_t(img, tindex) = _getindex_t(img, timeaxis(img), tindex)
_getindex_t(img, ::Nothing, tindex) = img
_getindex_t(img, tax::Axis, tindex) = view(img, tax(tindex))

end
