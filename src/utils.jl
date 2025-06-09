function truncate_small_numbers(x::Float64)
    if isapprox(x, 0.0, atol = 1e-6)
        return 0.0
    else
        return x
    end
end
