"""
    Results{T <: Real, N}

A custom structure to store the results of the Benders algorithm. The structure is defined by:
 * `output_id::String`: The name of the output
 * `stage::Int`: The stage of the output
 * `scenario::Int`: The scenario of the output
 * `df::DataFrame`: The DataFrame with the results.

The dataframe with results can store multiple dimensions in a tabular way.
If a given output varies by block the first column od the dataframe should be the block id.
If a given output varies by block and segment, the two first columns should be the block and segment ids.
"""
mutable struct Results
    # Some things might be missing and that should go into 
    # the metadata of the Arrow file:
    # - The stage type of the output
    output_id::String
    stage::Int
    scenario::Int
    df::DataFrame
end

function Results(
    output_id::String,
    stage::Int,
    scenario::Int,
    dimension_names::Vector{String},
    dimension_values::VecOrMat{TD},
    agent_names::Vector{String},
    agent_values::VecOrMat{TA}
) where {TD<:Real,TA<:Real}

    num_errors = 0
    if length(dimension_names) != size(dimension_values, 2)
        @error(
            "The number of dimension names ($(length_dimension_names)) must be equal " *
            "to the number of columns in dimension_values ($(size(dimension_values, 2)))."
        )
        num_errors += 1
    end
    if length(agent_names) != size(agent_values, 2)
        @error(
            "The number of agent names ($(length(agent_names))) must be equal " *
            "to the number of columns in agent_values ($(size(agent_values, 2)))."
        )
        num_errors += 1
    end
    if num_errors > 0
        error("result $output_id of stage $stage scenario $scenario has $(num_errors) validation errors.")
    end

    df = DataFrame()
    for (i, name) in enumerate(dimension_names)
        df[!, Symbol(name)] = dimension_values[:, i]
    end
    for (i, name) in enumerate(agent_names)
        df[!, Symbol(name)] = agent_values[:, i]
    end
    return Results(output_id, stage, scenario, df)
end

"""
    serialize_results(results::Results, outputs_path::AbstractString)

Serialize the results of the Benders algorithm to a file in the outputs_path directory.
This function should be called for every result of the Benders algorithm that we wish to save.
"""
function serialize_results(results::Results, outputs_path::AbstractString)
    output_id = results.output_id
    output_dir = joinpath(outputs_path, output_id)
    if !ispath(output_dir)
        mkdir(output_dir)
    end
    path_of_result = joinpath(output_dir, "$(output_id)_stage_$(results.stage)_scenario_$(results.scenario).jls")
    Serialization.serialize(path_of_result, results)
    return nothing
end

# TODO we could have a serial and a parallel version of this function
"""
    results_gatherer(
        stages::Int,
        forwards::Vector{Int},
        outputs_path::AbstractString
    )::Nothing

Gathers the results of the Benders algorithm from the outputs_path directory into Arrow files.
This function should be called after all the results of the Benders algorithm have been saved.
"""
function results_gatherer(
    stages::Int,
    # This is a vector if intgers because we can
    # choose to simulate only a few forward scenarios.
    forwards::Vector{Int},
    outputs_path::AbstractString
)::Nothing
    # Query all available output ids
    output_path_content = readdir(outputs_path)

    # Filter files by output_id
    # Save all output ids in a Set
    output_ids = Set{String}()
    for content in output_path_content
        if isdir(joinpath(outputs_path, content))
            push!(output_ids, content)
        end
    end

    # Gather results in an Arrow Stream by output
    # This scheme can be paralelized on the output_ids
    for output_id in output_ids
        # Gather results
        output_dir = joinpath(outputs_path, output_id)
        open(Arrow.Writer, joinpath(outputs_path, "$(output_id).arrow")) do writer
            for t in 1:stages, s in forwards
                path_of_result = joinpath(output_dir, "$(output_id)_stage_$(t)_scenario_$(s).jls")
                if !isfile(path_of_result)
                    error("Missing result for $(output_id) stage $(t) scenario $(s).")
                end
                # deserialize the results
                results = Serialization.deserialize(path_of_result)
                # Add the stage and forward scenario to the DataFrame
                DataFrames.insertcols!(results.df, 1, :stage => t, :scenario => s)
                # Write the results to the Arrow file
                Arrow.write(writer, results.df)
            end
        end
        # Remove all the files
        rm(output_dir; recursive=true)
    end
    return nothing
end

"""
    read_benders_result(file_path::AbstractString)

Reads the result of an Benders optimization from an Arrow file into a DataFrame.
"""
function read_benders_result(file_path::AbstractString)
    return DataFrame(Arrow.Table(file_path))
end