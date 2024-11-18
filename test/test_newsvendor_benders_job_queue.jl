using LightBenders
using JobQueueMPI
using JobQueueMPI.MPI

project_dir = dirname(Base.active_project())
parallel_script = joinpath(@__DIR__, "script_newsvendor_benders_job_queue.jl")

mpiexec(exe -> run(`$exe -n 2 $(Base.julia_cmd()) --project=$(project_dir) $(parallel_script)`))
