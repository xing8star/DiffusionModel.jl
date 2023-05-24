
BSON.@load "model.bson" ps st


res=p_sample_loop(model,ps,st,(64,64,3,1))


display_image(res[1])


unbatch(res)
res=vcat(unbatch(res)...);
res=cat(unbatch(res)...;dims=2);
display_image(res)
init_dim=mod(28,3)*2
dims=[2,(1, 2, 4).*28...]
28*2