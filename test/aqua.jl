function test_aqua()
    @testset "LightBenders" begin
        Aqua.test_ambiguities(LightBenders, recursive = false)
    end
    Aqua.test_all(LightBenders, ambiguities = false)

    return nothing
end
