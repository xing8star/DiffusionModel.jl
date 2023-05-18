using Images
display_image(data::Array{<:AbstractFloat, 3})=colorview(RGB,permutedims(data,(3,1,2)))
display_image(data::Array{<:AbstractFloat, 4})=display_image(vcat(unbatch(data)...))

save_image(data,path::String)=Images.save(path,display_image(data))
save_safe_image(data,path::String)=Images.save(path,map(clamp01nan,display_image(data)))
load_image(path::String)=Images.load(path) |>
                            channelview |>
                            x->permutedims(x,(2,3,1))
load_image(path::String,size::Tuple)=Images.load(path) |>
                            x->imresize(x,size)|>
                            channelview |>
                            x->permutedims(x,(2,3,1))
test_image(path::String)= load_image(path)|> x->unsqueeze(x,4)
function cite3channel(x::AbstractArray{<:Real,3})
    if size(x,3)==4
        return x[:,:,1:3]
    else
        return x
    end
end    