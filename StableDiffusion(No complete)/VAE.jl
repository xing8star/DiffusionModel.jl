using Zygote
include("layers.jl")


upsample(dim)=Chain(Upsample(2),Conv((3,3),dim=>dim,pad=1))

struct VAE <: Lux.AbstractExplicitContainerLayer{(:encoder,:decoder)}
    encoder
    decoder
    function VAE(;dim=128,init_dim=dim,dim_mults=(1,2,4,4),
        channels=3,out_dim=channels,resnet_block_groups=32,use_convnext=true,convnext_mult=2)
        dims=[init_dim,dim_mults.*dim...]
        in_out=zip(dims[begin:end-1],dims[begin+1:end])
        block_klass(dim, dim_out; time_emb_dim = nothing) =if use_convnext
            ConvNextBlock(dim, dim_out; time_emb_dim, mult = convnext_mult,norm = true)
        else
            ResnetBlock(dim, dim_out; time_emb_dim,groups=resnet_block_groups)
        end
        blocks =Any[
        Conv((3,3),channels=>init_dim,pad=1)
        ]

        num_resolutions = length(in_out)
        #downs
        for (ind,(dim_in,dim_out)) in enumerate(in_out)
            is_last=ind>=num_resolutions
            push!(blocks,Chain(
                block_klass(dim_in,dim_out),
                block_klass(dim_out,dim_out),
                Residual(PreNorm(dim_out,LinearAttention(dim_out))),
                if !is_last downsample(dim_out) else NoOpLayer() end
                )
            )
        end
        #mid
        mid_dim = dims[end]
        push!(blocks,Chain(
                block_klass(mid_dim,mid_dim),
                Attention(mid_dim),
                block_klass(mid_dim,mid_dim)
                )
            )
        #out
        push!(blocks,Chain(
                GroupNorm(mid_dim,resnet_block_groups,silu, epsilon=1f-6),
                Conv((3,3),mid_dim=>8,pad=1)
                ),
            #正态分布层
            Conv((3,3),8=>8,pad=1)
            )
        encoder=Chain(blocks...)
        # empty!(blocks)
        blocks =[
        Conv((1,1),4=>4),
        #in
        Conv((3,3),4=>mid_dim,pad=1),
        #mid
        Chain(
                block_klass(mid_dim,mid_dim),
                Attention(mid_dim),
                block_klass(mid_dim,mid_dim)
                )
        ]
        #ups
        block_klass(dim_in,dim_out)=(
            ResnetBlock(dim_in,dim_out),
            [ResnetBlock(dim_out,dim_out) for _ = 1:2]...
        )
        push!(blocks,Chain(
                block_klass(mid_dim,mid_dim),
                upsample(mid_dim)
                )
            )
        for (ind, (dim_in, dim_out)) in enumerate(reverse(collect(in_out)[begin+1:end]))
            is_last = ind >= num_resolutions-1
            push!(blocks,Chain(
                block_klass(dim_out,dim_in),
                if !is_last upsample(dim_in) else NoOpLayer() end
                )
            )
        end
        #out
        push!(blocks,Chain(
                GroupNorm(dim,resnet_block_groups,silu, epsilon=1f-6),
                Conv((3,3),dim=>out_dim,pad=1)
                )
            )
        decoder=Chain(blocks...)
        new(encoder,decoder)
    end
end
# 8/2+1
function diffsample(h)
    mean = h[:,:, begin:4,:]
    logvar = h[:,:, 5:end,:]
    std = sqrt.(exp.(logvar))
    h = Lux.randn32(rng,size(mean))
    h = mean + std .* h
end

function (self::VAE)(x,ps,st)
    x,st_encoder=self.encoder(x,ps.encoder,st.encoder)
    x=diffsample(x)
    x,st_decoder=self.decoder(x,ps.decoder,st.decoder)

    x,merge(st, (encoder=st_encoder,decoder=st_decoder))
end

self=VAE()
ps,st=Lux.setup(Lux.Random.GLOBAL_RNG,self);
self(Lux.rand32(rng,128,128,3,1),ps,st)

myPad(x)=pad_zeros(x, (0,1,0,1),dims=(1,2))

# size(ans[1])

# dims=[128,(1,2,4,4).*128...]
# in_out=zip(dims[begin:end-1],dims[begin+1:end])