# For best performance with large numbers of scenarios (100+), consider using
# JuMP.direct_model(optimizer) in your state_variables_builder instead of
# JuMP.Model(optimizer). This bypasses the caching layer for significant speedup.
Base.@kwdef mutable struct DeterministicEquivalentOptions
    num_scenarios::Int
    debugging_options::DebuggingOptions = DebuggingOptions()
    risk_measure::AbstractRiskMeasure = RiskNeutral()
    set_names::Bool = false  # Set to true to name variables (useful for debugging, but slower)
    skip_results::Bool = false  # Set to true to skip results extraction (useful when only writing LP files)
end

# ============================================================================
# Parameter Handling
# ============================================================================

struct ParameterInfo
    vars::Vector{JuMP.VariableRef}
    values::Vector{Float64}
end

function identify_parameters(model::JuMP.Model)
    parameter_vars = JuMP.VariableRef[]
    moi_model = JuMP.backend(model)

    for var in JuMP.all_variables(model)
        var_index = JuMP.index(var)
        ci_type = MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{Float64}}
        ci = ci_type(var_index.value)

        if MOI.is_valid(moi_model, ci)
            push!(parameter_vars, var)
        end
    end

    return parameter_vars
end

function extract_parameter_values(model::JuMP.Model, parameter_vars::Vector{JuMP.VariableRef})
    moi_model = JuMP.backend(model)
    values = map(parameter_vars) do var
        var_index = JuMP.index(var)
        ci_type = MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{Float64}}
        ci = ci_type(var_index.value)
        param_set = MOI.get(moi_model, MOI.ConstraintSet(), ci)
        param_set.value
    end
    return values
end

function get_parameter_info(subproblem::JuMP.Model, scenario::Int, inputs, second_stage_modifier::Function)
    # Call modifier to update parameters for this scenario
    second_stage_modifier(subproblem, inputs, scenario)

    # Extract current parameter values
    parameter_vars = identify_parameters(subproblem)
    parameter_values = extract_parameter_values(subproblem, parameter_vars)

    return ParameterInfo(parameter_vars, parameter_values)
end

# ============================================================================
# Constraint Caching
# ============================================================================

struct ConstraintCache
    constraint_types::Vector{Tuple{DataType, DataType}}
    constraints::Vector{Vector{Any}}
end

function cache_constraints(subproblem::JuMP.Model)
    constraint_types = JuMP.list_of_constraint_types(subproblem)
    constraints = [collect(JuMP.all_constraints(subproblem, F, S)) for (F, S) in constraint_types]
    return ConstraintCache(constraint_types, constraints)
end

function add_cached_constraints!(
    model::JuMP.Model,
    cache::ConstraintCache,
    var_mapping::Dict{JuMP.VariableRef, Any},
    state_vars_set::Set{JuMP.VariableRef}
)
    for (type_idx, (F, S)) in enumerate(cache.constraint_types)
        # Skip MOI.Parameter constraints (not real constraints)
        S <: MOI.Parameter && continue

        constraint_list = cache.constraints[type_idx]

        for con in constraint_list
            obj = JuMP.constraint_object(con)

            # Skip state variable bounds (already in model)
            if obj.func isa JuMP.VariableRef && obj.func in state_vars_set
                continue
            end

            new_func = substitute_variables(obj.func, var_mapping)
            JuMP.add_constraint(model, JuMP.build_constraint(error, new_func, obj.set))
        end
    end

    return nothing
end

# ============================================================================
# Scenario Processing
# ============================================================================

struct ScenarioData
    subproblem_vars::Vector{JuMP.VariableRef}
    parameter_vars::Vector{JuMP.VariableRef}
    state_vars_set::Set{JuMP.VariableRef}
    constraint_cache::ConstraintCache
end

function prepare_subproblem_template(subproblem::JuMP.Model)
    # Identify parameters and cache constraints
    parameter_vars = identify_parameters(subproblem)
    constraint_cache = cache_constraints(subproblem)

    # Get non-state, non-parameter variables
    state_vars = subproblem.ext[:first_stage_state].variables
    all_vars = JuMP.all_variables(subproblem)
    subproblem_vars = setdiff(all_vars, union(Set(state_vars), Set(parameter_vars)))

    # Pre-build state variable set for O(1) lookup
    state_vars_set = Set(state_vars)

    return ScenarioData(subproblem_vars, parameter_vars, state_vars_set, constraint_cache)
end

function build_variable_mapping(
    model::JuMP.Model,
    subproblem::JuMP.Model,
    scenario_vars::Vector{JuMP.VariableRef},
    param_info::ParameterInfo,
    scenario_data::ScenarioData
)
    var_mapping = Dict{JuMP.VariableRef, Any}()
    sizehint!(var_mapping, length(scenario_vars) + length(model.ext[:first_stage_state].variables) + length(param_info.vars))

    # Map subproblem variables to scenario-specific variables
    for (src, dest) in zip(scenario_data.subproblem_vars, scenario_vars)
        var_mapping[src] = dest
    end

    # Map state variables (first stage -> second stage)
    model_state = model.ext[:first_stage_state]
    subproblem_state = subproblem.ext[:first_stage_state]
    for (model_var, subproblem_var) in zip(model_state.variables, subproblem_state.variables)
        var_mapping[subproblem_var] = model_var
    end

    # Map parameters to their constant values for this scenario
    for (param_var, param_value) in zip(param_info.vars, param_info.values)
        var_mapping[param_var] = param_value
    end

    return var_mapping
end

function add_scenario_to_model!(
    model::JuMP.Model,
    subproblem::JuMP.Model,
    scenario::Int,
    inputs,
    second_stage_modifier::Function,
    scenario_data::ScenarioData,
    options::DeterministicEquivalentOptions,
    var_scenario_map::Union{Dict, Nothing}
)
    # Get parameter values for this scenario
    param_info = get_parameter_info(subproblem, scenario, inputs, second_stage_modifier)

    # Create scenario-specific variables
    num_vars = length(scenario_data.subproblem_vars)
    scenario_vars = @variable(model, [i in 1:num_vars])

    # Build variable mapping
    var_mapping = build_variable_mapping(model, subproblem, scenario_vars, param_info, scenario_data)

    # Track variable names for results extraction (if needed)
    if !options.skip_results && !isnothing(var_scenario_map)
        for (src, dest) in zip(scenario_data.subproblem_vars, scenario_vars)
            src_name = JuMP.name(src)
            if !isempty(src_name)
                var_scenario_map[dest] = (src_name, scenario)
            end
        end
    end

    # Set variable names for debugging (if requested)
    if options.set_names
        for (src, dest) in zip(scenario_data.subproblem_vars, scenario_vars)
            src_name = JuMP.name(src)
            if !isempty(src_name)
                JuMP.set_name(dest, "$(src_name)_scen$(scenario)")
            end
        end
    end

    # Add constraints for this scenario
    add_cached_constraints!(model, scenario_data.constraint_cache, var_mapping, scenario_data.state_vars_set)

    # Extract and return objective for this scenario
    subproblem_objective = substitute_variables(JuMP.objective_function(subproblem), var_mapping)
    return subproblem_objective
end

# ============================================================================
# Objective Building
# ============================================================================

function build_risk_neutral_objective(
    first_stage_objective,
    scenario_objectives::Vector,
    num_scenarios::Int
)
    weighted_scenarios = [(1 / num_scenarios) * obj for obj in scenario_objectives]
    return first_stage_objective + sum(weighted_scenarios)
end

function build_cvar_objective(
    model::JuMP.Model,
    first_stage_objective,
    scenario_objectives::Vector,
    num_scenarios::Int,
    cvar::CVaR,
    set_names::Bool
)
    # CVaR formulation: auxiliary variables z and delta[scen] >= 0
    # Constraints: delta[scen] >= scenario_objective[scen] - z
    # Objective: first_stage + (1-λ) * mean(scenarios) + λ * (z + 1/(1-α) * mean(deltas))

    @variable(model, z_cvar)
    @variable(model, delta_cvar[1:num_scenarios] >= 0)

    # Set names for debugging (if requested)
    if set_names
        JuMP.set_name(z_cvar, "z_cvar")
        for scen in 1:num_scenarios
            JuMP.set_name(delta_cvar[scen], "delta_cvar_scen$(scen)")
        end
    end

    # Add CVaR constraints
    for scen in 1:num_scenarios
        con = @constraint(model, delta_cvar[scen] >= scenario_objectives[scen] - z_cvar)
        set_names && JuMP.set_name(con, "cvar_constraint_scen$(scen)")
    end

    # Build complete objective
    alpha, lambda = cvar.alpha, cvar.lambda
    expected_value = sum((1 - lambda) / num_scenarios * scenario_objectives[scen] for scen in 1:num_scenarios)
    cvar_term = lambda * (z_cvar + sum(delta_cvar[scen] / ((1 - alpha) * num_scenarios) for scen in 1:num_scenarios))

    return first_stage_objective + expected_value + cvar_term
end

function build_objective!(
    model::JuMP.Model,
    first_stage_objective,
    scenario_objectives::Vector,
    options::DeterministicEquivalentOptions
)
    objective = if options.risk_measure isa RiskNeutral
        build_risk_neutral_objective(first_stage_objective, scenario_objectives, options.num_scenarios)
    elseif options.risk_measure isa CVaR
        build_cvar_objective(model, first_stage_objective, scenario_objectives,
                           options.num_scenarios, options.risk_measure, options.set_names)
    else
        error("Unsupported risk measure: $(typeof(options.risk_measure))")
    end

    JuMP.set_objective_function(model, objective)
    return nothing
end

# ============================================================================
# Main Entry Point
# ============================================================================

function deterministic_equivalent(;
    state_variables_builder::Function,
    first_stage_builder::Function,
    second_stage_builder::Function,
    second_stage_modifier::Function,
    inputs = nothing,
    options::DeterministicEquivalentOptions,
)
    DeterministicEquivalentLog(options.num_scenarios)
    @info "Building deterministic equivalent with $(options.num_scenarios) scenarios..."

    # Build first-stage model
    @info "  [1/5] Building first stage model..."
    model = build_first_stage_model(state_variables_builder, first_stage_builder, inputs)
    first_stage_objective = JuMP.objective_function(model)

    # Build and cache second-stage template
    @info "  [2/5] Building subproblem template..."
    subproblem = build_second_stage_template(state_variables_builder, second_stage_builder, inputs)

    @info "  [3/5] Caching constraint structure..."
    scenario_data = prepare_subproblem_template(subproblem)

    # Pre-allocate variable mapping for results (if needed)
    var_scenario_map = initialize_variable_map(scenario_data, options)

    # Add all scenarios to the model
    @info "  [4/5] Adding scenarios to model..."
    scenario_objectives = add_all_scenarios!(
        model, subproblem, inputs, second_stage_modifier,
        scenario_data, options, var_scenario_map
    )

    # Build and set the objective function
    @info "  [5/5] Building objective and optimizing..."
    build_objective!(model, first_stage_objective, scenario_objectives, options)

    # Solve and extract results
    return solve_and_extract_results(model, options, var_scenario_map)
end

# ============================================================================
# Helper Functions for Main Entry Point
# ============================================================================

function build_first_stage_model(state_variables_builder::Function, first_stage_builder::Function, inputs)
    model = state_variables_builder(inputs, 1)
    first_stage_builder(model, inputs)
    return model
end

function build_second_stage_template(state_variables_builder::Function, second_stage_builder::Function, inputs)
    subproblem = state_variables_builder(inputs, 1)
    second_stage_builder(subproblem, inputs)
    return subproblem
end

function initialize_variable_map(scenario_data::ScenarioData, options::DeterministicEquivalentOptions)
    if options.skip_results
        return nothing
    else
        num_vars = length(scenario_data.subproblem_vars)
        var_map = Dict{JuMP.VariableRef, Tuple{String, Int}}()
        sizehint!(var_map, num_vars * options.num_scenarios)
        return var_map
    end
end

function add_all_scenarios!(
    model::JuMP.Model,
    subproblem::JuMP.Model,
    inputs,
    second_stage_modifier::Function,
    scenario_data::ScenarioData,
    options::DeterministicEquivalentOptions,
    var_scenario_map::Union{Dict, Nothing}
)
    scenario_objectives = Vector{Any}(undef, options.num_scenarios)
    scenario_start_time = time()

    for scen in 1:options.num_scenarios
        # Show progress periodically
        log_scenario_progress(scen, options.num_scenarios, scenario_start_time)

        # Add this scenario to the model
        scenario_objectives[scen] = add_scenario_to_model!(
            model, subproblem, scen, inputs, second_stage_modifier,
            scenario_data, options, var_scenario_map
        )
    end

    return scenario_objectives
end

function log_scenario_progress(scenario::Int, num_scenarios::Int, start_time::Float64)
    # Log at start, every 10%, and at end
    if scenario == 1 || scenario % max(1, num_scenarios ÷ 10) == 0 || scenario == num_scenarios
        elapsed = time() - start_time
        @info "    Processing scenario $scenario/$num_scenarios ($(round(elapsed, digits=1))s elapsed)"
    end
end

function solve_and_extract_results(
    model::JuMP.Model,
    options::DeterministicEquivalentOptions,
    var_scenario_map::Union{Dict, Nothing}
)
    # Store variable mapping for results extraction (if needed)
    !options.skip_results && (model.ext[:var_scenario_map] = var_scenario_map)

    # Solve the model
    optimize_start = time()
    @info "    Calling solver..."
    JuMP.optimize!(model)
    @info "    Solver finished in $(round(time() - optimize_start, digits=2))s"

    treat_termination_status(model, options)

    # Extract and return results
    results = options.skip_results ? Dict{Tuple{String, Int}, Any}() : save_deterministic_results(model, options.num_scenarios)
    results["objective", 0] = JuMP.objective_value(model)
    return results
end

# ============================================================================
# Variable Substitution
# ============================================================================

# Substitute variables in expressions, handling parameters (replaced with constants)
substitute_variables(src::Vector, map::Dict) = substitute_variables.(src, Ref(map))
substitute_variables(src::Real, ::Dict) = src
substitute_variables(src::JuMP.VariableRef, map::Dict) = map[src]

function substitute_variables(src::JuMP.GenericAffExpr, var_map::Dict{JuMP.VariableRef, Any})
    new_constant = src.constant
    new_terms = Pair{VariableRef, Float64}[]

    for (var, coef) in src.terms
        replacement = var_map[var]

        if replacement isa Real
            # Parameter -> constant: add to constant term
            new_constant += coef * replacement
        else
            # Variable -> variable: add to terms
            push!(new_terms, replacement => coef)
        end
    end

    return JuMP.GenericAffExpr(new_constant, new_terms)
end

function substitute_variables(src::JuMP.GenericQuadExpr, var_map::Dict{JuMP.VariableRef, Any})
    new_aff = substitute_variables(src.aff, var_map)
    new_quad_terms = Pair{UnorderedPair{VariableRef}, Float64}[]

    for (pair, coef) in src.terms
        replacement_a = var_map[pair.a]
        replacement_b = var_map[pair.b]

        if replacement_a isa Real && replacement_b isa Real
            # Both parameters: becomes constant
            new_aff += coef * replacement_a * replacement_b
        elseif replacement_a isa Real
            # One parameter: becomes linear
            new_aff += coef * replacement_a * replacement_b
        elseif replacement_b isa Real
            # One parameter: becomes linear
            new_aff += coef * replacement_b * replacement_a
        else
            # Both variables: stays quadratic
            push!(new_quad_terms, UnorderedPair{VariableRef}(replacement_a, replacement_b) => coef)
        end
    end

    return JuMP.GenericQuadExpr(new_aff, new_quad_terms)
end

