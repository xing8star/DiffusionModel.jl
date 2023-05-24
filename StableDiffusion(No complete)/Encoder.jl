using Zygote
include("layers.jl")

rng=Lux.Random.GLOBAL_RNG
@kwdef struct Embed  <: Lux.AbstractExplicitContainerLayer{(:embed,:pos_embed)}
    embed=Embedding(49408=>768)
    pos_embed=Embedding(77=>768)
    pos_ids=reshape(Int32.(range(1,77)),:,1)
end

function (self::Embed)(x,ps,st)
    embed,st_embed = self.embed(x,ps.embed,st.embed)
    pos_embed,st_pos_embed = self.pos_embed(self.pos_ids,ps.pos_embed,st.pos_embed)
    embed.+pos_embed,merge(st, (embed=st_embed,pos_embed=st_pos_embed))
end



# self=Attention(1)
# ps,st=Lux.setup(Lux.Random.GLOBAL_RNG,self)
# self(Lux.rand32(rng,768,77,1,2),ps,st)

squeeze(x)=reshape(x,[size(x,i) for i in [1,2,4]]...)

@kwdef struct ClipEncoder  <: Lux.AbstractExplicitContainerLayer{(:block1,:block2,:block3)}
    block1=Chain(LayerNorm((768,1)),unsqueeze(dims=3),Attention(1),squeeze)
    block2=Chain(LayerNorm((768,1)),Dense(768=>3072))
    block3=Dense(3072=>768)
end
function (self::ClipEncoder)(x,ps,st)
    x1,st_block1=self.block1(x,ps.block1,st.block1)
    x = x + x1
    res = x
    x,st_block2=self.block2(x,ps.block2,st.block2)

    x = x .* sigmoid(x * 1.702f0)
    x3,st_block3=self.block3(x,ps.block3,st.block3)
    res + x3,merge(st, (block1=st_block1,block2=st_block2,block3=st_block3))
end

# self=ClipEncoder()
# self(Lux.rand32(rng,768,77,2),ps,st)

encoder=Chain(Embed(),[ClipEncoder() for i = 1:12]...,LayerNorm((768,1)))

# ps,st=Lux.setup(Lux.Random.GLOBAL_RNG,encoder)

# encoder(reshape(Int32.(1:77),:,1),ps,st)
# size(ans[1])
