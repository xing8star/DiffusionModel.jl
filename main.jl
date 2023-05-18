using Zygote
using Random
using BSON
include("dataset.jl")
include("layers.jl")
include("config.jl")
include("sample.jl")
include("train.jl")


# x=load_image("avater.jpg",(28,28))
x=load_image("avater.jpg",(64,64))

x=unsqueeze(x,dims=4).|>Float32

x=device(x)
for epoch = 1:100
    t=rand(0:timesteps,(1,)) |>device
    
    (loss,st), back = Zygote.pullback(p->p_losses(model,p,st,x,t,huber_loss),ps) 
    gs = back((one(loss),nothing))[1]
    optimizer, ps = Optimisers.update(optimizer, ps, gs)
    println(loss)
end


BSON.@save "model.bson" ps=cpu(ps) st=cpu(st)
