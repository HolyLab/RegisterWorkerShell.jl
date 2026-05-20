using RegisterWorkerShell, Test, Aqua, ExplicitImports
using ImageAxes, ImageMetadata, AxisArrays
using SharedArrays
using Distributed

# Concrete AbstractWorker subtype for testing
mutable struct TestWorker <: AbstractWorker
    workertid::Int
    param_scalar::Float64
    param_array::Vector{Int}
    param_string::String
end

@testset "RegisterWorkerShell" begin
    Aqua.test_all(RegisterWorkerShell)
    @test_nowarn test_explicit_imports(RegisterWorkerShell)


    @testset "ArrayDecl" begin
        adcl = ArrayDecl(Array{Float64, 2}, (3, 4))
        @test adcl.arraysize == (3, 4)
        @test eltype(adcl) == Float64

        adcl3 = ArrayDecl(Array{Int32, 3}, (2, 3, 4))
        @test adcl3.arraysize == (2, 3, 4)
        @test eltype(adcl3) == Int32
    end

    @testset "monitor" begin
        w = TestWorker(1, 3.14, [1, 2, 3], "hello")

        mon = monitor(w, (:param_scalar, :param_array))
        @test mon[:param_scalar] == 3.14
        @test mon[:param_array] == [1, 2, 3]
        @test !haskey(mon, :param_string)

        # monitor with extra variables
        mon2 = monitor(w, (:param_scalar,), Dict{Symbol, Any}(:extra => 42))
        @test mon2[:param_scalar] == 3.14
        @test mon2[:extra] == 42

        # field that doesn't exist should be silently skipped
        mon3 = monitor(w, (:nonexistent,))
        @test isempty(mon3)

        # Vector of workers
        workers = [TestWorker(i, Float64(i), [i], "w$i") for i in 1:3]
        mons = monitor(workers, (:param_scalar,))
        @test length(mons) == 3
        for i in 1:3
            @test mons[i][:param_scalar] == Float64(i)
        end

        # Regression: per-worker mons must not alias morevars values. Sharing the
        # same buffer caused races under multi-threaded drivers where every frame's
        # output collapsed to the last writer.
        shared_buf = [0, 0, 0]
        mons_v = monitor(workers, (), Dict{Symbol, Any}(:buf => shared_buf))
        @test mons_v[1][:buf] !== mons_v[2][:buf]
        @test mons_v[2][:buf] !== mons_v[3][:buf]
        @test mons_v[1][:buf] !== shared_buf
        @test mons_v[1][:buf] == shared_buf
    end

    @testset "monitor!" begin
        w = TestWorker(1, 2.71, [4, 5], "world")

        # monitor! updates all monitored fields
        mon = Dict{Symbol, Any}(:param_scalar => 0.0, :param_array => [0, 0])
        monitor!(mon, w)
        @test mon[:param_scalar] == 2.71
        @test mon[:param_array] == [4, 5]

        # monitor! with symbol + array of same size: should copyto! in place
        original_ref = [0, 0]
        mon[:param_array] = original_ref
        monitor!(mon, :param_array, [10, 20])
        @test mon[:param_array] == [10, 20]
        @test mon[:param_array] === original_ref  # same object, mutated

        # monitor! with symbol + array of different size: should replace
        monitor!(mon, :param_array, [1, 2, 3])
        @test mon[:param_array] == [1, 2, 3]

        # monitor! with scalar value
        monitor!(mon, :param_scalar, 9.99)
        @test mon[:param_scalar] == 9.99

        # monitor! with key not in mon should NOT add it
        monitor!(mon, :not_monitored, 100)
        @test !haskey(mon, :not_monitored)
    end

    @testset "init! and close!" begin
        w = TestWorker(1, 1.0, [1], "test")

        @test init!(w) === nothing
        @test close!(w) === nothing

        # extra args are accepted
        @test init!(w, 1, :a) === nothing
        @test close!(w, 1, :a) === nothing
    end

    @testset "worker (unimplemented)" begin
        w = TestWorker(1, 1.0, [1], "test")
        @test_throws ErrorException worker(w, nothing, 1, Dict{Symbol, Any}())
    end

    @testset "workertid" begin
        w = TestWorker(7, 1.0, [1], "test")
        @test workertid(w) == 7
    end

    @testset "load_mm_package" begin
        @test load_mm_package(:cpu) === nothing
        @test load_mm_package(:gpu, 1, 2) === nothing
    end

    @testset "maybe_sharedarray" begin
        # Array on same process: returned as-is
        A = [1.0, 2.0, 3.0]
        @test maybe_sharedarray(A) === A

        # Bits type + size: creates SharedArray
        S = maybe_sharedarray(Float64, (3, 4))
        @test S isa SharedArray{Float64}
        @test size(S) == (3, 4)

        # Non-bits type + size: creates regular Array
        R = maybe_sharedarray(String, (2, 3))
        @test R isa Array{String}
        @test size(R) == (2, 3)

        # ArrayDecl with bits eltype: creates SharedArray
        adcl = ArrayDecl(Array{Float32, 2}, (5, 6))
        Sa = maybe_sharedarray(adcl)
        @test Sa isa SharedArray{Float32}
        @test size(Sa) == (5, 6)

        # Non-array scalar passthrough
        @test maybe_sharedarray(42) === 42
        @test maybe_sharedarray("hello") === "hello"
    end

    @testset "getindex_t" begin
        # Image without time axis: return img itself
        img_no_t = rand(3, 4)
        @test getindex_t(img_no_t, 1) === img_no_t

        # AxisArray with time axis: return view at tindex
        data = rand(3, 4, 5)
        img_t = AxisArray(
            data,
            Axis{:x}(1:3),
            Axis{:y}(1:4),
            Axis{:time}(1:5)
        )
        slice = getindex_t(img_t, 3)
        @test size(slice) == (3, 4)
        @test slice == view(data, :, :, 3)
    end

    @testset "RemoteChannel dispatch" begin
        rr = RemoteChannel(() -> Channel{AbstractWorker}(1))
        put!(rr, TestWorker(1, 1.0, [1], "test"))

        @test init!(rr) === nothing
        @test close!(rr) === nothing
        @test_throws ErrorException worker(rr, nothing, 1, Dict{Symbol,Any}())
        @test load_mm_package(rr) === nothing
    end

    @testset "maybe_sharedarray remote pid" begin
        pids = addprocs(1)
        try
            @everywhere pids using SharedArrays
            A = [1.0, 2.0, 3.0]
            S = maybe_sharedarray(A, pids[1])
            @test S isa SharedArray{Float64}
            @test S == A
        finally
            rmprocs(pids)
        end
    end

end
