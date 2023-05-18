using Optimisers


model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,8)
);



ps,st=Lux.setup(Lux.Random.GLOBAL_RNG,model);

# BSON.@load "model.bson" ps st


optimizer = Optimisers.setup(Adam(1f-3),ps);
Optimisers.ChainRulesCore.@non_differentiable p_sample_loop(a,b)
Optimisers.ChainRulesCore.@non_differentiable p_sample(a,b,c,d)


ps,st=(ps,st).|>device;
optimizer=device(optimizer);