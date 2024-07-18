function deterministic_equivalent(;
    state_variables_builder::Function,
    first_stage_builder::Function,
    second_stage_builder::Function,
    second_stage_modifier::Function,
    inputs=nothing,
    num_scenarios::Int,
)
    DeterministicEquivalentLog(num_scenarios)

    model = state_variables_builder(inputs)
    first_stage_builder(model, inputs)
    subproblem = state_variables_builder(inputs)
    second_stage_builder(subproblem, inputs)
    for scen in 1:num_scenarios
        second_stage_modifier(subproblem, inputs, scen)
        push_model!(
            model,
            subproblem,
            scen,
            num_scenarios,
        )
    end
    JuMP.optimize!(model)
    treat_termination_status(model, 0, 0)
    results = save_deterministic_results(model, num_scenarios)
    results["objective", 0] = JuMP.objective_value(model)
    return results
end

function copy_and_replace_variables(
    src::Vector,
    map::Dict{JuMP.VariableRef, JuMP.VariableRef},
)
    return copy_and_replace_variables.(src, Ref(map))
end

function copy_and_replace_variables(
    src::Real,
    ::Dict{JuMP.VariableRef, JuMP.VariableRef},
)
    return src
end

function copy_and_replace_variables(
    src::JuMP.VariableRef,
    src_to_dest_variable::Dict{JuMP.VariableRef, JuMP.VariableRef},
)
    return src_to_dest_variable[src]
end

function copy_and_replace_variables(
    src::JuMP.GenericAffExpr,
    src_to_dest_variable::Dict{JuMP.VariableRef, JuMP.VariableRef},
)
    return JuMP.GenericAffExpr(
        src.constant,
        Pair{VariableRef,Float64}[
            src_to_dest_variable[key] => val for (key, val) in src.terms
        ],
    )
end

function copy_and_replace_variables(
    src::JuMP.GenericQuadExpr,
    src_to_dest_variable::Dict{JuMP.VariableRef, JuMP.VariableRef},
)
    return JuMP.GenericQuadExpr(
        copy_and_replace_variables(src.aff, src_to_dest_variable),
        Pair{UnorderedPair{VariableRef},Float64}[
            UnorderedPair{VariableRef}(
                src_to_dest_variable[pair.a],
                src_to_dest_variable[pair.b],
            ) => coef for (pair, coef) in src.terms
        ],
    )
end

function is_state_variable(var, model::JuMP.Model)
    state = model.ext[:state]
    for idx in eachindex(state.variables)
        if var == state.variables[idx]
            return true
        end
    end
    return false
end

function num_state_variables(model::JuMP.Model)
    return length(model.ext[:state].variables)
end

function all_variables_but_state(model::JuMP.Model)
    return filter(var -> !is_state_variable(var, model), vcat(JuMP.all_variables(model)))
end

function push_model!(
    model::JuMP.Model,
    subproblem::JuMP.Model,
    scenario::Int,
    num_scenarios::Int
)
    # push variables
    src_variables = all_variables_but_state(subproblem)
    x = @variable(model, [i in 1:length(src_variables)])
    var_src_to_dest = Dict{JuMP.VariableRef, JuMP.VariableRef}()
    for (src, dest) in zip(src_variables, x)
        var_src_to_dest[src] = dest
        name = JuMP.name(src)
        if isempty(name)
            name = string("_[", index(src).value, "]")
        end
        # append scenario to original variable index
        JuMP.set_name(dest, string(name, "_scen#", scenario))
    end
    # push state variables
    model_state = model.ext[:state]
    subproblem_state = subproblem.ext[:state]
    for idx in eachindex(model_state.variables)
        # since this is a two stage model
        # state variables do not change
        # with the scenario
        var_src_to_dest[subproblem_state.variables[idx]] = model_state.variables[idx]
    end
    # push constraints
    for (F, S) in JuMP.list_of_constraint_types(subproblem)
        for con in JuMP.all_constraints(subproblem, F, S)
            obj = JuMP.constraint_object(con)
            if is_state_variable(obj.func, subproblem)
                # skip state variable bounds
                # they are already in the model
                # due to the state_variables_builder
                continue
            end
            new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
            @constraint(model, new_func in obj.set)
        end
    end
    # push objective
    current = JuMP.objective_function(model)
    subproblem_objective =
        copy_and_replace_variables(JuMP.objective_function(subproblem), var_src_to_dest)
    JuMP.set_objective_function(
        model,
        current + (1 / num_scenarios) * subproblem_objective,
    )
    return nothing
end
