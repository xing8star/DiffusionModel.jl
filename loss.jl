using Statistics
function _check_sizes(ŷ::AbstractArray, y::AbstractArray)
    for d in 1:max(ndims(ŷ), ndims(y)) 
    size(ŷ,d) == size(y,d) || throw(DimensionMismatch(
        "loss function expects size(ŷ) = $(size(ŷ)) to match size(y) = $(size(y))"
    ))
    end
end
ofeltype(x, y) = convert(float(eltype(x)), y)
function huber_loss(ŷ, y; agg = mean, δ = ofeltype(ŷ, 1))
    _check_sizes(ŷ, y)
    abs_error = abs.(ŷ .- y)
    temp = Zygote.dropgrad(abs_error .<  δ)
    x = ofeltype(ŷ, 0.5)
    agg(((abs_error .^ 2) .* temp) .* x .+ δ * (abs_error .- x * δ) .* (1 .- temp))
end
function mse_loss(ŷ, y)
    mean(abs2, ŷ .- y)
end