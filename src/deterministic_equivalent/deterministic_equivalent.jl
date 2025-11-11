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

function deterministic_equivalent(;
    state_variables_builder::Function,
    first_stage_builder::Function,
    second_stage_builder::Function,
    second_stage_modifier::Function,
    inputs = nothing,
    options::DeterministicEquivalentOptions,
)
    num_scenarios = options.num_scenarios
    DeterministicEquivalentLog(num_scenarios)

    @info "Building deterministic equivalent with $num_scenarios scenarios..."

    stage = 1
    @info "  [1/5] Building first stage model..."
    model = state_variables_builder(inputs, stage)
    first_stage_builder(model, inputs)

    # Extract first stage objective once
    first_stage_objective = JuMP.objective_function(model)

    @info "  [2/5] Building subproblem template..."
    # Create subproblem template and extract structure ONCE
    subproblem = state_variables_builder(inputs, stage)
    second_stage_builder(subproblem, inputs)

    @info "  [3/5] Caching constraint structure..."
    # Cache constraint structure to avoid repeated model inspection
    cached_constraint_info = cache_constraint_structure(subproblem)

    # Identify parameter variables (variables in MOI.Parameter set)
    parameter_vars = identify_parameter_variables(subproblem)

    # Cache subproblem variables (they don't change between scenarios)
    # Exclude both state variables AND parameter variables
    cached_subproblem_vars = all_variables_but_state_and_parameters(subproblem, parameter_vars)
    num_vars = length(cached_subproblem_vars)

    # Pre-build Set of state variables for O(1) lookup (CRITICAL for performance)
    state_vars_set = Set(subproblem.ext[:first_stage_state].variables)

    # Pre-allocate objective terms array
    objective_terms = Vector{Any}(undef, num_scenarios)
    scenario_objectives = Vector{Any}(undef, num_scenarios)  # Store for CVaR weight calculation

    # Track variable mapping for results extraction (only if needed)
    # Maps: destination_var => (source_var_name, scenario_number)
    var_scenario_map = if !options.skip_results
        d = Dict{JuMP.VariableRef, Tuple{String, Int}}()
        sizehint!(d, num_vars * num_scenarios)
        d
    else
        nothing
    end

    @info "  [4/5] Adding scenarios to model..."
    scenario_start_time = time()
    # Process each scenario
    for scen in 1:num_scenarios
        # Show progress every 10% of scenarios
        if scen == 1 || scen % max(1, num_scenarios ÷ 10) == 0 || scen == num_scenarios
            elapsed = time() - scenario_start_time
            @info "    Processing scenario $scen/$num_scenarios ($(round(elapsed, digits=1))s elapsed)"
        end

        # Call modifier to set parameter values for this scenario
        second_stage_modifier(subproblem, inputs, scen)

        # Extract parameter values for this scenario
        parameter_values = extract_parameter_values(subproblem, parameter_vars)

        # Add variables for this scenario
        scenario_vars = @variable(model, [i in 1:num_vars])

        # Build variable mapping with pre-allocated dictionary
        # Now includes parameter substitution (parameters -> constants)
        var_src_to_dest = Dict{JuMP.VariableRef, Any}()
        sizehint!(var_src_to_dest, num_vars + num_state_variables(model) + length(parameter_vars))

        # Map regular variables to scenario-specific variables
        for (src, dest) in zip(cached_subproblem_vars, scenario_vars)
            var_src_to_dest[src] = dest

            # Store mapping for results extraction (only if needed)
            if !options.skip_results
                src_name = JuMP.name(src)
                if !isempty(src_name)
                    var_scenario_map[dest] = (src_name, scen)
                end
            end

            # Optionally set names for debugging
            if options.set_names
                src_name = JuMP.name(src)
                if !isempty(src_name)
                    JuMP.set_name(dest, "$(src_name)_scen$(scen)")
                end
            end
        end

        # Map state variables
        model_state = model.ext[:first_stage_state]
        subproblem_state = subproblem.ext[:first_stage_state]
        for idx in eachindex(model_state.variables)
            var_src_to_dest[subproblem_state.variables[idx]] = model_state.variables[idx]
        end

        # Map parameter variables to their fixed values (constants) for this scenario
        for (param_var, param_value) in zip(parameter_vars, parameter_values)
            var_src_to_dest[param_var] = param_value
        end

        # Add constraints using cached structure
        add_constraints_from_cache!(model, cached_constraint_info, var_src_to_dest, state_vars_set)

        # Extract and store objective term
        subproblem_objective = copy_and_replace_variables(
            JuMP.objective_function(subproblem),
            var_src_to_dest
        )
        scenario_objectives[scen] = subproblem_objective
    end

    @info "  [5/5] Building objective and optimizing..."
    # Build complete objective with appropriate risk measure
    if options.risk_measure isa RiskNeutral
        # Equal weights for risk-neutral case
        for scen in 1:num_scenarios
            objective_terms[scen] = (1 / num_scenarios) * scenario_objectives[scen]
        end
        JuMP.set_objective_function(model, first_stage_objective + sum(objective_terms))
    elseif options.risk_measure isa CVaR
        # For CVaR, we need to solve first to get scenario objective values,
        # then compute weights. Instead, we'll use the explicit CVaR formulation.
        # This adds auxiliary variables z and delta[scen] >= 0 for each scenario
        # such that delta[scen] >= scenario_objectives[scen] - z
        # The objective becomes: first_stage + (1-λ)*mean(scenarios) + λ*(z + 1/(1-α)*mean(deltas))

        @variable(model, z_cvar)
        @variable(model, delta_cvar[1:num_scenarios] >= 0)

        # Optionally set names for CVaR variables
        if options.set_names
            JuMP.set_name(z_cvar, "z_cvar")
            for scen in 1:num_scenarios
                JuMP.set_name(delta_cvar[scen], "delta_cvar_scen$(scen)")
            end
        end

        # Add constraints: delta[scen] >= scenario_objective[scen] - z
        for scen in 1:num_scenarios
            con = @constraint(model, delta_cvar[scen] >= scenario_objectives[scen] - z_cvar)
            if options.set_names
                JuMP.set_name(con, "cvar_constraint_scen$(scen)")
            end
        end

        alpha = options.risk_measure.alpha
        lambda = options.risk_measure.lambda

        # Objective: first_stage + (1-λ) * (1/N) * sum(scenarios) + λ * (z + 1/(1-α) * (1/N) * sum(deltas))
        expected_value_term = sum((1 - lambda) / num_scenarios * scenario_objectives[scen] for scen in 1:num_scenarios)
        cvar_term = lambda * (z_cvar + sum(delta_cvar[scen] / ((1 - alpha) * num_scenarios) for scen in 1:num_scenarios))

        JuMP.set_objective_function(model, first_stage_objective + expected_value_term + cvar_term)
    else
        error("Unsupported risk measure: $(typeof(options.risk_measure))")
    end

    # Store the variable mapping in the model for fast results extraction (if needed)
    if !options.skip_results
        model.ext[:var_scenario_map] = var_scenario_map
    end

    optimize_start = time()
    @info "    Calling solver..."
    JuMP.optimize!(model)
    optimize_time = time() - optimize_start
    @info "    Solver finished in $(round(optimize_time, digits=2))s"

    treat_termination_status(model, options)

    if options.skip_results
        # Return minimal results - just objective value
        results = Dict{Tuple{String, Int}, Any}()
        results["objective", 0] = JuMP.objective_value(model)
        return results
    else
        # Full results extraction
        results = save_deterministic_results(model, num_scenarios)
        results["objective", 0] = JuMP.objective_value(model)
        return results
    end
end

function copy_and_replace_variables(
    src::Vector,
    map::Dict{JuMP.VariableRef, Any},
)
    return copy_and_replace_variables.(src, Ref(map))
end

function copy_and_replace_variables(
    src::Real,
    ::Dict{JuMP.VariableRef, Any},
)
    return src
end

function copy_and_replace_variables(
    src::JuMP.VariableRef,
    src_to_dest_variable::Dict{JuMP.VariableRef, Any},
)
    return src_to_dest_variable[src]
end

function copy_and_replace_variables(
    src::JuMP.GenericAffExpr,
    src_to_dest_variable::Dict{JuMP.VariableRef, Any},
)
    # Build the new affine expression
    # Start with the constant term
    new_constant = src.constant
    new_terms = Pair{VariableRef, Float64}[]

    for (key, val) in src.terms
        replacement = src_to_dest_variable[key]
        if replacement isa Real
            # Parameter variable replaced with constant - add to constant term
            new_constant += val * replacement
        else
            # Regular variable replacement
            push!(new_terms, replacement => val)
        end
    end

    return JuMP.GenericAffExpr(new_constant, new_terms)
end

function copy_and_replace_variables(
    src::JuMP.GenericQuadExpr,
    src_to_dest_variable::Dict{JuMP.VariableRef, Any},
)
    # Handle quadratic expressions with parameter substitution
    new_aff = copy_and_replace_variables(src.aff, src_to_dest_variable)
    new_quad_terms = Pair{UnorderedPair{VariableRef}, Float64}[]

    for (pair, coef) in src.terms
        replacement_a = src_to_dest_variable[pair.a]
        replacement_b = src_to_dest_variable[pair.b]

        if replacement_a isa Real && replacement_b isa Real
            # Both are parameters - add constant to affine part
            new_aff += coef * replacement_a * replacement_b
        elseif replacement_a isa Real
            # First is parameter, second is variable - becomes linear term
            new_aff += coef * replacement_a * replacement_b
        elseif replacement_b isa Real
            # Second is parameter, first is variable - becomes linear term
            new_aff += coef * replacement_b * replacement_a
        else
            # Both are variables - keep as quadratic term
            push!(new_quad_terms, UnorderedPair{VariableRef}(replacement_a, replacement_b) => coef)
        end
    end

    return JuMP.GenericQuadExpr(new_aff, new_quad_terms)
end

function is_state_variable(var, model::JuMP.Model)
    return var isa JuMP.VariableRef &&
           var in model.ext[:first_stage_state].variables
end

function num_state_variables(model::JuMP.Model)
    return length(model.ext[:first_stage_state].variables)
end

function all_variables_but_state(model::JuMP.Model)
    return setdiff(
        JuMP.all_variables(model),
        model.ext[:first_stage_state].variables,
    )
end

function all_variables_but_state_and_parameters(model::JuMP.Model, parameter_vars::Vector{JuMP.VariableRef})
    all_vars = JuMP.all_variables(model)
    state_vars = model.ext[:first_stage_state].variables
    # Remove both state and parameter variables
    return setdiff(all_vars, union(Set(state_vars), Set(parameter_vars)))
end

function identify_parameter_variables(model::JuMP.Model)
    parameter_vars = JuMP.VariableRef[]
    for var in JuMP.all_variables(model)
        # Check if this variable has a MOI.Parameter constraint
        var_index = JuMP.index(var)
        moi_model = JuMP.backend(model)
        # Try to get the constraint type - parameters have VariableIndex-in-Parameter constraints
        ci_type = MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{Float64}}
        ci = ci_type(var_index.value)
        if MOI.is_valid(moi_model, ci)
            push!(parameter_vars, var)
        end
    end
    return parameter_vars
end

function extract_parameter_values(model::JuMP.Model, parameter_vars::Vector{JuMP.VariableRef})
    values = Float64[]
    for var in parameter_vars
        # Get the current parameter value
        var_index = JuMP.index(var)
        moi_model = JuMP.backend(model)
        ci_type = MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{Float64}}
        ci = ci_type(var_index.value)
        param_set = MOI.get(moi_model, MOI.ConstraintSet(), ci)
        push!(values, param_set.value)
    end
    return values
end

# Cache constraint structure to avoid repeated model inspection
struct ConstraintCache
    constraint_types::Vector{Tuple{DataType, DataType}}
    constraints::Vector{Vector{Any}}  # Stores constraint references for each type
end

function cache_constraint_structure(subproblem::JuMP.Model)
    constraint_types = JuMP.list_of_constraint_types(subproblem)
    constraints = Vector{Vector{Any}}(undef, length(constraint_types))

    for (i, (F, S)) in enumerate(constraint_types)
        constraints[i] = collect(JuMP.all_constraints(subproblem, F, S))
    end

    return ConstraintCache(constraint_types, constraints)
end

function add_constraints_from_cache!(
    model::JuMP.Model,
    cache::ConstraintCache,
    var_src_to_dest::Dict{JuMP.VariableRef, Any},
    state_vars_set::Set{JuMP.VariableRef}
)
    for (type_idx, (F, S)) in enumerate(cache.constraint_types)
        # Skip MOI.Parameter constraints - they're not real constraints
        # and direct_model doesn't support them
        if S <: MOI.Parameter
            continue
        end

        # Collect all valid constraints for this type (skip state variable bounds)
        constraint_list = cache.constraints[type_idx]

        # Pre-allocate arrays for vectorized constraint addition
        funcs = Vector{Any}()
        sets = Vector{Any}()
        sizehint!(funcs, length(constraint_list))
        sizehint!(sets, length(constraint_list))

        for con in constraint_list
            obj = JuMP.constraint_object(con)
            # Use Set for O(1) lookup instead of O(N) linear search
            if obj.func isa JuMP.VariableRef && obj.func in state_vars_set
                # Skip state variable bounds - already in model
                continue
            end
            new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
            push!(funcs, new_func)
            push!(sets, obj.set)
        end

        # Add all constraints of this type at once
        if !isempty(funcs)
            # Add constraints one by one, normalizing if needed
            # (JuMP.@constraint automatically normalizes, so we use it)
            for (func, set) in zip(funcs, sets)
                JuMP.add_constraint(model, JuMP.build_constraint(error, func, set))
            end
        end
    end
    return nothing
end

