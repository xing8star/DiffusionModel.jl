
BSON.@load "model.bson" ps st

display_image(res1[1])

noi=[q_sample(x,i) for i in [1,50,150]].|>cpu

display_image(noi[1])


unbatch(res)
res=vcat(unbatch(res)...);
res=cat(unbatch(res)...;dims=2);
display_image(res)
init_dim=mod(28,3)*2
dims=[2,(1, 2, 4).*28...]
28*2