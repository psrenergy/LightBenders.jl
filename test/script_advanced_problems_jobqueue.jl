module AdvancedProblemsJobQueue

using Test
using LightBenders
using JuMP
using HiGHS

"""
Advanced test problems to thoroughly validate job queue implementation.
Includes multi-dimensional states, integer variables, and complex constraints.
"""

# ============================================================================
# TEST 1: Multi-Product Inventory Problem
# ============================================================================

Base.@kwdef mutable struct MultiProductInputs
    num_products::Int
    buy_cost::Vector{Float64}      # Cost to purchase each product
    sell_price::Vector{Float64}    # Revenue from selling each product
    return_value::Vector{Float64}  # Value of returning unsold inventory
    storage_capacity::Float64      # Shared storage capacity
    storage_cost_per_unit::Float64 # Cost to hold inventory
    demand::Matrix{Float64}        # demand[product, scenario]
end

function mp_state_builder(inputs, stage)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    sp = LightBenders.SubproblemModel(model)

    if stage == 1
        @variable(sp, inventory[1:inputs.num_products] >= 0)
        LightBenders.set_first_stage_state(sp, :inventory, inventory)
    elseif stage == 2
        @variable(sp, inventory[1:inputs.num_products] in MOI.Parameter(0.0))
        LightBenders.set_second_stage_state(sp, :inventory, inventory)
    end
    return sp
end

function mp_first_stage(sp, inputs)
    inventory = sp[:inventory]

    # Capacity constraint
    @constraint(sp, sum(inventory) <= inputs.storage_capacity)

    # Purchase cost + storage cost
    @objective(sp, Min,
        sum(inputs.buy_cost[p] * inventory[p] for p in 1:inputs.num_products) +
        inputs.storage_cost_per_unit * sum(inventory)
    )

    return sp
end

function mp_second_stage(sp, inputs)
    inventory = sp[:inventory]

    @variable(sp, demand_param[1:inputs.num_products] in MOI.Parameter(0.0))
    @variable(sp, sold[1:inputs.num_products] >= 0)
    @variable(sp, returned[1:inputs.num_products] >= 0)

    # Sales cannot exceed demand
    @constraint(sp, [p = 1:inputs.num_products], sold[p] <= demand_param[p])

    # Balance constraint per product
    @constraint(sp, [p = 1:inputs.num_products],
        sold[p] + returned[p] <= inventory[p])

    # Revenue from sales and returns
    @objective(sp, Min,
        -sum(inputs.sell_price[p] * sold[p] for p in 1:inputs.num_products) -
        sum(inputs.return_value[p] * returned[p] for p in 1:inputs.num_products)
    )

    return sp
end

function mp_second_stage_modifier(sp, inputs, scenario)
    demand_param = sp[:demand_param]
    for p in 1:inputs.num_products
        JuMP.set_parameter_value(demand_param[p], inputs.demand[p, scenario])
    end
    return nothing
end

function test_multi_product()
    @testset "Multi-Product Inventory" begin
        inputs = MultiProductInputs(
            num_products = 4,
            buy_cost = [3.0, 5.0, 4.0, 6.0],
            sell_price = [8.0, 12.0, 10.0, 15.0],
            return_value = [1.0, 2.0, 1.5, 2.5],
            storage_capacity = 200.0,
            storage_cost_per_unit = 0.1,
            demand = [
                10 15 20 25 30 35;  # Product 1
                20 25 30 35 40 45;  # Product 2
                15 20 25 30 35 40;  # Product 3
                25 30 35 40 45 50   # Product 4
            ],
        )

        num_scenarios = size(inputs.demand, 2)

        # Deterministic Equivalent
        det_eq_options = LightBenders.DeterministicEquivalentOptions(; num_scenarios = num_scenarios)
        det_eq_results = LightBenders.deterministic_equivalent(;
            state_variables_builder = mp_state_builder,
            first_stage_builder = mp_first_stage,
            second_stage_builder = mp_second_stage,
            second_stage_modifier = mp_second_stage_modifier,
            inputs = inputs,
            options = det_eq_options,
        )
        det_eq_obj = det_eq_results["objective", 0]

        # Serial
        serial_policy = LightBenders.train(;
            state_variables_builder = mp_state_builder,
            first_stage_builder = mp_first_stage,
            second_stage_builder = mp_second_stage,
            second_stage_modifier = mp_second_stage_modifier,
            inputs = inputs,
            policy_training_options = LightBenders.PolicyTrainingOptions(;
                num_scenarios = num_scenarios,
                lower_bound = -1e6,
                implementation_strategy = LightBenders.SerialTraining(),
                stopping_rule = [LightBenders.GapWithMinimumNumberOfIterations(abstol = 1e-2, min_iterations = 3)],
                cut_strategy = LightBenders.CutStrategy.MultiCut,
                verbose = true,
            ),
        )

        # Job Queue
        jq_policy = LightBenders.train(;
            state_variables_builder = mp_state_builder,
            first_stage_builder = mp_first_stage,
            second_stage_builder = mp_second_stage,
            second_stage_modifier = mp_second_stage_modifier,
            inputs = inputs,
            policy_training_options = LightBenders.PolicyTrainingOptions(;
                num_scenarios = num_scenarios,
                lower_bound = -1e6,
                implementation_strategy = LightBenders.JobQueueTraining(),
                stopping_rule = [LightBenders.GapWithMinimumNumberOfIterations(abstol = 1e-2, min_iterations = 3)],
                cut_strategy = LightBenders.CutStrategy.MultiCut,
                verbose = true,
            ),
        )

        if jq_policy !== nothing
            println("\n=== Multi-Product Inventory (4 products, 6 scenarios) ===")
            println("Det. Eq.  - Obj: $(det_eq_obj)")
            println("Serial    - LB: $(LightBenders.lower_bound(serial_policy)), UB: $(LightBenders.upper_bound(serial_policy))")
            println("Job Queue - LB: $(LightBenders.lower_bound(jq_policy)), UB: $(LightBenders.upper_bound(jq_policy))")

            @test LightBenders.lower_bound(serial_policy) ≈ det_eq_obj atol = 1e-1
            @test LightBenders.upper_bound(serial_policy) ≈ det_eq_obj atol = 1e-1
            @test LightBenders.lower_bound(jq_policy) ≈ det_eq_obj atol = 1e-1
            @test LightBenders.upper_bound(jq_policy) ≈ det_eq_obj atol = 1e-1
        end
    end
end

# ============================================================================
# TEST 2: Capacity Expansion Problem (with binary variables)
# ============================================================================

Base.@kwdef mutable struct CapacityExpansionInputs
    num_generators::Int
    build_cost::Vector{Float64}      # Cost to build each generator
    max_capacity::Vector{Float64}    # Max capacity of each generator
    operating_cost::Vector{Float64}  # Variable operating cost
    demand::Vector{Float64}          # Demand per scenario
end

function ce_state_builder(inputs, stage)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    sp = LightBenders.SubproblemModel(model)

    if stage == 1
        @variable(sp, built[1:inputs.num_generators], Bin)
        LightBenders.set_first_stage_state(sp, :built, built)
    elseif stage == 2
        @variable(sp, built[1:inputs.num_generators])
        LightBenders.set_second_stage_state(sp, :built, built)
    end
    return sp
end

function ce_first_stage(sp, inputs)
    built = sp[:built]

    # Build cost
    @objective(sp, Min, sum(inputs.build_cost[g] * built[g] for g in 1:inputs.num_generators))

    return sp
end

function ce_second_stage(sp, inputs)
    built = sp[:built]

    @variable(sp, demand_param in MOI.Parameter(0.0))
    @variable(sp, generation[1:inputs.num_generators] >= 0)
    @variable(sp, unmet_demand >= 0)  # Recourse variable for infeasibility

    # Can only generate if built
    @constraint(sp, [g = 1:inputs.num_generators],
        generation[g] <= inputs.max_capacity[g] * built[g])

    # Must meet demand (with unmet demand as slack)
    @constraint(sp, sum(generation) + unmet_demand == demand_param)

    # Operating cost + penalty for unmet demand
    @objective(sp, Min,
        sum(inputs.operating_cost[g] * generation[g] for g in 1:inputs.num_generators) +
        1000.0 * unmet_demand  # High penalty for unmet demand
    )

    return sp
end

function ce_second_stage_modifier(sp, inputs, scenario)
    demand_param = sp[:demand_param]
    JuMP.set_parameter_value(demand_param, inputs.demand[scenario])
    return nothing
end

function test_capacity_expansion()
    @testset "Capacity Expansion (Binary Variables)" begin
        inputs = CapacityExpansionInputs(
            num_generators = 3,
            build_cost = [100.0, 150.0, 120.0],
            max_capacity = [50.0, 80.0, 60.0],
            operating_cost = [2.0, 1.5, 1.8],
            demand = [40, 60, 80, 100, 120, 140, 160],
        )

        num_scenarios = length(inputs.demand)

        # Deterministic Equivalent
        det_eq_options = LightBenders.DeterministicEquivalentOptions(; num_scenarios = num_scenarios)
        det_eq_results = LightBenders.deterministic_equivalent(;
            state_variables_builder = ce_state_builder,
            first_stage_builder = ce_first_stage,
            second_stage_builder = ce_second_stage,
            second_stage_modifier = ce_second_stage_modifier,
            inputs = inputs,
            options = det_eq_options,
        )
        det_eq_obj = det_eq_results["objective", 0]

        # Serial
        serial_policy = LightBenders.train(;
            state_variables_builder = ce_state_builder,
            first_stage_builder = ce_first_stage,
            second_stage_builder = ce_second_stage,
            second_stage_modifier = ce_second_stage_modifier,
            inputs = inputs,
            policy_training_options = LightBenders.PolicyTrainingOptions(;
                num_scenarios = num_scenarios,
                lower_bound = 0.0,
                implementation_strategy = LightBenders.SerialTraining(),
                stopping_rule = [LightBenders.IterationLimit(10)],
                cut_strategy = LightBenders.CutStrategy.MultiCut,
                mip_options = LightBenders.MIPOptions(run_mip_after_iteration = 3),
                verbose = true,
            ),
        )

        # Job Queue
        jq_policy = LightBenders.train(;
            state_variables_builder = ce_state_builder,
            first_stage_builder = ce_first_stage,
            second_stage_builder = ce_second_stage,
            second_stage_modifier = ce_second_stage_modifier,
            inputs = inputs,
            policy_training_options = LightBenders.PolicyTrainingOptions(;
                num_scenarios = num_scenarios,
                lower_bound = 0.0,
                implementation_strategy = LightBenders.JobQueueTraining(),
                stopping_rule = [LightBenders.IterationLimit(10)],
                cut_strategy = LightBenders.CutStrategy.MultiCut,
                mip_options = LightBenders.MIPOptions(run_mip_after_iteration = 3),
                verbose = true,
            ),
        )

        if jq_policy !== nothing
            println("\n=== Capacity Expansion (3 generators, 7 scenarios, binary vars) ===")
            println("Det. Eq.  - Obj: $(det_eq_obj)")
            println("Serial    - LB: $(LightBenders.lower_bound(serial_policy)), UB: $(LightBenders.upper_bound(serial_policy))")
            println("Job Queue - LB: $(LightBenders.lower_bound(jq_policy)), UB: $(LightBenders.upper_bound(jq_policy))")

            @test LightBenders.lower_bound(serial_policy) ≈ det_eq_obj atol = 1.0
            @test LightBenders.upper_bound(serial_policy) ≈ det_eq_obj atol = 1.0
            @test LightBenders.lower_bound(jq_policy) ≈ det_eq_obj atol = 1.0
            @test LightBenders.upper_bound(jq_policy) ≈ det_eq_obj atol = 1.0
        end
    end
end

# ============================================================================
# Run All Tests
# ============================================================================

function run_all_tests()
    @testset "Advanced Problems - Job Queue Validation" begin
        test_multi_product()
        test_capacity_expansion()
    end
end

run_all_tests()

end # module AdvancedProblemsJobQueue
