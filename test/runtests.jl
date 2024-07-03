using Test

function test_modules(dir::AbstractString)
    result = Dict{String, Vector{String}}()
    for (root, dirs, files) in walkdir(dir)
        for file in joinpath.(root, filter(f -> occursin(r"test_(.)+\.jl", f), files))
            main_case = splitpath(file)[end-2]
            if !haskey(result, main_case)
                result[main_case] = String[]
            end
            push!(result[main_case], file)
        end
    end
    return result
end

@testset "Tests" begin
    for (main_case, files) in test_modules(@__DIR__)
        @testset "$main_case" begin
            for file in files
                @testset "$(basename(dirname(file)))" begin
                    include(file)
                end
            end
        end
    end
end