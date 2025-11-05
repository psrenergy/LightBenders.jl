using LightBenders
using JobQueueMPI
using JobQueueMPI.MPI

project_dir = dirname(Base.active_project())
parallel_script = joinpath(@__DIR__, "script_advanced_problems_jobqueue.jl")

mpiexec(exe -> run(`$exe -n 3 $(Base.julia_cmd()) --project=$(project_dir) $(parallel_script)`))
