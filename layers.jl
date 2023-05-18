using Lux
using Lux.NNlib
upsample(dim)=ConvTranspose((4,4),dim=>dim;stride=2,pad=1)
downsample(dim)=Conv((4,4),dim=>dim;stride=2,pad=1)
function SinusoidalPositionEmbeddings(dim)
    dim=Float32(dim) 
    half_dim=fld(dim,2)
    embeddings=log(10000f0)/(half_dim-1)
    embeddings= exp.(collect(range(stop=half_dim))*-embeddings)
    embeddings=reshape(embeddings,1,:) |> device
    function (time)
        # embeddings=reshape(time,:,1)*reshape(embeddings,1,:)
        time=Float32.(time) 
        embedding=time*embeddings
        embedding=cat(sin.(embedding),cos.(embedding),dims=2)
        permutedims(embedding,(2,1))
        # Float32.(embeddings)
    end
end

silu(x)=x*sigmoid(x)
struct Block <: Lux.AbstractExplicitContainerLayer{(:proj,:norm)}
    proj::Lux.AbstractExplicitLayer
    norm::Lux.AbstractExplicitLayer
    activation
    Block(dim,dim_out,groups=8)=new(
        Conv((3,3),dim=>dim_out;pad=1),
        GroupNorm(dim_out,groups),silu
        )
end

function (b::Block)(x,ps,st,scale_shift=nothing)
    x,st_proj=b.proj(x,ps.proj,st.proj)
    x,st_norm=b.norm(x,ps.norm,st.norm)
    if !isnothing(scale_shift)
        scale,shift=scale_shift
        x=x*(scale+1)+shift
    end
    b.activation(x),merge(st, (proj=st_proj,norm=st_norm))
end


struct ResnetBlock <: Lux.AbstractExplicitContainerLayer{(:mlp,:block1,:block2,:resconv)}
    mlp
    block1
    block2
    resconv
    function ResnetBlock(dim,dim_out;time_emb_dim=nothing, groups=8)
        mlp=if !isnothing(time_emb_dim) Chain(silu,Dense(time_emb_dim=>dim_out)) else NoOpLayer() end
        block1=Block(dim,dim_out,groups)
        block2=Block(dim_out,dim_out,groups)
        resconv=if dim!=dim_out Conv((1,1),dim=>dim_out) else NoOpLayer() end
        new(mlp,block1,block2,resconv)
    end
end

addwh(x)=reshape(x,1,1,size(x)...)

function (b::ResnetBlock)(x,ps,st,time_emb=nothing)
    h,st_block1=b.block1(x,ps.block1,st.block1)
    if !(isnothing(b.mlp)||isnothing(time_emb))
        time_emb,_=b.mlp(time_emb,ps.mlp,st.mlp)
        h=addwh(time_emb)+h
    end
    h,st_block2=b.block2(h,ps.block2,st.block2)
    x,st_resconv=b.resconv(x,ps.resconv,st.resconv)
    h+x,merge(st, (block1=st_block1,block2=st_block2,resconv=st_resconv))
end

struct ConvNextBlock <: Lux.AbstractExplicitContainerLayer{(:mlp,:ds_conv,:net,:resconv)}
    mlp
    ds_conv
    net
    resconv
    function ConvNextBlock(dim,dim_out;time_emb_dim=nothing,mult=2,norm=true)
        mlp=if !isnothing(time_emb_dim) Chain(gelu,Dense(time_emb_dim=>dim)) else NoOpLayer() end
        ds_conv=Conv((7,7),dim=>dim;pad=3,groups=dim)
        net=Chain(if norm GroupNorm(dim,1) else NoOpLayer() end,
            Conv((3,3),dim=>dim_out*mult,gelu;pad=1),
            GroupNorm(dim_out*mult,1),
            Conv((3,3),dim_out*mult=>dim_out,gelu;pad=1)
        )
        resconv=if dim!=dim_out Conv((1,1),dim=>dim_out) else NoOpLayer() end
        new(mlp,ds_conv,net,resconv)
    end 
end
function (b::ConvNextBlock)(x,ps,st,time_emb=nothing)
    h,st_ds_conv=b.ds_conv(x,ps.ds_conv,st.ds_conv)
    if !(b.mlp isa NoOpLayer||isnothing(time_emb))
        condition,st_mlp=b.mlp(time_emb,ps.mlp,st.mlp)
        h=addwh(condition).+h
    end
    h,st_net=b.net(h,ps.net,st.net)
    x,st_resconv=b.resconv(x,ps.resconv,st.resconv)
    h+x,merge(st, (ds_conv=st_ds_conv,net=st_net,resconv=st_resconv))
end

struct Attention <: Lux.AbstractExplicitContainerLayer{(:to_qkv,:to_out)}
    scale
    heads
    hidden_dim
    to_qkv
    to_out
    function Attention(dim,heads=4,dim_head=32)
        hidden_dim=dim_head * heads
        new(Float32(dim_head^(-0.5)),
            heads,dim_head*heads,
            Conv((1,1),dim=>hidden_dim*3;bias=false),
            Conv((1,1),hidden_dim=>dim)
        )
    end
end

NNlib.batched_transpose(A::AbstractArray{T, 4}) where T=permutedims(A,(2,1,3,4))
using MLUtils
function (b::Attention)(x,ps,st)
    h,w,c,n=size(x)
    qkv,st_to_qkv=b.to_qkv(x,ps.to_qkv,st.to_qkv)
    qkv=chunk(qkv,3,dims=3)
    # let
    #     c=size(qkv,3)/3 |>Int
    #     [qkv[:,:,(i-1)*c+1:i*c,:] for i in 1:3]
    # end
    q,k,v=map(x->reshape(x,size(x,1)*size(x,2),size(x,3)÷b.heads,b.heads,size(x,4)),qkv)
    q=q*b.scale
    sim=batched_mul(batched_transpose(q),k)
    sim=sim .- maximum(sim,dims=1)
    attn=softmax(sim)
    out=batched_mul(attn,batched_transpose(v))
    out=reshape(out,(h,w,size(out,3)*size(out,1),size(out,4)))
    out,st_to_out=b.to_out(out,ps.to_out,st.to_out)
    out,merge(st, (to_qkv=st_to_qkv,to_out=st_to_out))
end

# function op(a,b)
#     a,b=map(x->reshape(x,size(x,3),size(x,4)),[a,b])
#     x=a'*b
#     reshape(x,1,1,size(x)...)
# end

struct LinearAttention <: Lux.AbstractExplicitContainerLayer{(:to_qkv,:to_out)}
    scale
    heads
    hidden_dim
    to_qkv
    to_out
    function LinearAttention(dim,heads=4,dim_head=32)
        hidden_dim=dim_head * heads
        new(Float32(dim_head^(-0.5)),
        heads,dim_head*heads,
        Conv((1,1),dim=>hidden_dim*3;bias=false),
        Chain(Conv((1,1),hidden_dim=>dim),GroupNorm(dim,1))
        )
    end
end

function (b::LinearAttention)(x,ps,st)
    h,w,c,n=size(x)
    qkv,st_to_qkv=b.to_qkv(x,ps.to_qkv,st.to_qkv)
    qkv=chunk(qkv,3,dims=3)
    q,k,v=map(x->reshape(x,size(x,1)*size(x,2),size(x,3)÷b.heads,b.heads,size(x,4)),qkv)
    q=softmax(q,dims=2)
    k=softmax(k,dims=1)
    q=q*b.scale
    context=batched_mul(k,batched_transpose(v))
    out=batched_mul(batched_transpose(context),q)
    out=reshape(out,(h,w,b.heads*size(out,2),size(out,4)))
    out,st_to_out=b.to_out(out,ps.to_out,st.to_out)
    out,merge(st, (to_qkv=st_to_qkv,to_out=st_to_out))
end

PreNorm(dim,fn)=Chain(GroupNorm(dim,1),fn)

struct Residual<: Lux.AbstractExplicitContainerLayer{(:layer,)}
    layer
end
function (b::Residual)(x,ps,st)
    out,st=b.layer(x,ps,st)
    out+x,st
end
struct Blocks4<: Lux.AbstractExplicitContainerLayer{(:block1,:block2,:attn,:up_down_sample)}
    block1
    block2
    attn
    up_down_sample
end

struct Unet <: Lux.AbstractExplicitContainerLayer{(:init_conv,:time_mlp,:downs,:ups,:mid_block1,
    :mid_attn,:mid_block2,:final_conv)}
    channels
    init_conv
    time_mlp
    downs
    ups
    mid_block1
    mid_attn
    mid_block2
    final_conv
    function Unet(;dim,init_dim=nothing,out_dim=nothing,dim_mults=(1,2,4,8),
        channels=3,with_time_emb=true,resnet_block_groups=8,use_convnext=true,convnext_mult=2)

        if isnothing(init_dim) init_dim=mod(dim,3)*2 end
        dims=[init_dim,dim_mults.*dim...]
        in_out=zip(dims[begin:end-1],dims[begin+1:end])
        block_klass(dim, dim_out; time_emb_dim = nothing) =if use_convnext
             ConvNextBlock(dim, dim_out; time_emb_dim, mult = convnext_mult,norm = true)
        else
            ResnetBlock(dim, dim_out; time_emb_dim,groups=resnet_block_groups)
        end
        if with_time_emb
            time_dim = dim * 4
            time_mlp = Chain(
                SinusoidalPositionEmbeddings(dim),
                Dense(dim=>time_dim,gelu),
                Dense(time_dim=>time_dim)
            )
        else
            time_dim = nothing
            time_mlp = NoOpLayer()
        end
        time_mlp
        blocks =[]

        num_resolutions = length(in_out)

        for (ind,(dim_in,dim_out)) in enumerate(in_out)
            is_last=ind>=num_resolutions
            push!(blocks,Blocks4(
                block_klass(dim_in,dim_out,time_emb_dim=time_dim),
                block_klass(dim_out,dim_out,time_emb_dim=time_dim),
                Residual(PreNorm(dim_out,LinearAttention(dim_out))),
                if !is_last downsample(dim_out) else NoOpLayer() end
                )
            )
        end
        downs=Chain(blocks...)
        mid_dim = dims[end]
        mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        empty!(blocks)
        for (ind, (dim_in, dim_out)) in enumerate(reverse(collect(in_out)[begin+1:end]))
            is_last = ind >= num_resolutions
            push!(blocks,
                Blocks4(
                    block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    if !(is_last) upsample(dim_in) else NoOpLayer() end
                )
            )
        end
        ups=Chain(blocks...)
        if isnothing(out_dim) out_dim = channels end
        final_conv = Chain(
            block_klass(dim, dim), Conv((1,1),dim=>out_dim)
        )
        new(channels,
        Conv((7,7),channels=>init_dim;pad=3),
        time_mlp,
        downs,
        ups,
        mid_block1,
        mid_attn,
        mid_block2,
        final_conv)
    end
end
using Accessors
function (self::Unet)(x,ps,st;time)
    x,st_init_conv= self.init_conv(x,ps.init_conv,st.init_conv)
    t,st_time_mlp = self.time_mlp(time,ps.time_mlp,st.time_mlp)
    if t==time
        t=nothing
    end
    h=Zygote.Buffer(Vector{Array{Float32,4}}())
    

    for (i,block) in enumerate(self.downs.layers)
        local dps =ps.downs
        local dst =st.downs
        local st_=Zygote.@ignore Dict()
        x,st_[1] = block.block1(x,dps[i].block1,dst[i].block1, t)
        x,st_[2]= block.block2(x,dps[i].block2,dst[i].block2, t)
        x,st_[3] = block.attn(x,dps[i].attn,dst[i].attn)
        push!(h,x)
        x,st_[4]= block.up_down_sample(x,dps[i].up_down_sample,dst[i].up_down_sample)
        st=Zygote.@ignore @set st.downs[i]=merge(dst[i],[k=>st_[ind] for (ind,(k,v)) in enumerate(collect(pairs(st.downs[i])))])
    end
    
    # bottleneck
    x,st_mid_block1= self.mid_block1(x,ps.mid_block1,st.mid_block1,t)
    x,st_mid_attn= self.mid_attn(x,ps.mid_attn,st.mid_attn)
    x,st_mid_block2= self.mid_block2(x,ps.mid_block2,st.mid_block2,t)
    id=Zygote.@ignore reverse(eachindex(h))
    
    # upsample

    for (i,block) in enumerate(self.ups.layers)
        x =cat(x,h[id[i]];dims=3)
        local dps =ps.ups
        local dst =st.ups
        local st_=Zygote.@ignore Dict()
        x,st_[1] = block.block1(x,dps[i].block1,dst[i].block1, t)
        x,st_[2]= block.block2(x,dps[i].block2,dst[i].block2, t)
        x,st_[3] = block.attn(x,dps[i].attn,dst[i].attn)
        x,st_[4]= block.up_down_sample(x,dps[i].up_down_sample,dst[i].up_down_sample)
        st=Zygote.@ignore @set st.ups[i]=merge(dst[i],[k=>st_[ind] for (ind,(k,v)) in enumerate(collect(pairs(st.ups[i])))])
    end
    
    x,st_final_conv= self.final_conv(x,ps.final_conv,st.final_conv)
    x,merge(st, (init_conv=st_init_conv,time_mlp=st_time_mlp,
                    mid_block1=st_mid_block1,mid_attn=st_mid_attn,
                    mid_block2=st_mid_block2,final_conv=st_final_conv))
end

function clip(x,min,max)
    if x>max
        max
    elseif x<min
        min
    else
     x
    end    
end

"""
cosine schedule as proposed in https://arxiv.org/abs/2102.09672
"""
function cosine_beta_schedule(timesteps, s=0.008)
    steps = timesteps + 1
    x = collect(LinRange(0, timesteps, steps))
    alphas_cumprod = cos.(((x / timesteps) .+ s) / (1 + s) * π * 0.5).^ 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[1]
    betas = 1 .- (alphas_cumprod[begin+1:end] / alphas_cumprod[begin:end-1])
    clip.(betas, 0.0001, 0.9999)
end
function linear_beta_schedule(timesteps)
    beta_start = 0.0001
    beta_end = 0.02
    return collect(LinRange(beta_start, beta_end, timesteps))
end
function quadratic_beta_schedule(timesteps)
    beta_start = 0.0001
    beta_end = 0.02
    return collect(LinRange(beta_start^0.5, beta_end^0.5, timesteps))^ 2

end
function sigmoid_beta_schedule(timesteps)
    beta_start = 0.0001
    beta_end = 0.02
    betas = collect(LinRange(-6, 6, timesteps))
    return sigmoid(betas) * (beta_end - beta_start) + beta_start
end