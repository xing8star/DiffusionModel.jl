
timesteps = 200

# define beta schedule
betas = linear_beta_schedule(timesteps) .|>Float32

# define alphas 
alphas = 1f0 .- betas
alphas_cumprod = cumprod(alphas,dims=1)
alphas_cumprod_prev = pad_constant(alphas_cumprod[begin:end-1], (1,0), 1.)
sqrt_recip_alphas = @. sqrt(1f0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = sqrt.(alphas_cumprod) 
# |>device
sqrt_one_minus_alphas_cumprod =@. sqrt(1f0 - alphas_cumprod) 
# sqrt_one_minus_alphas_cumprod=device(sqrt_one_minus_alphas_cumprod)
# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance =@. betas * (1f0 - alphas_cumprod_prev) / (1f0 - alphas_cumprod)
# posterior_variance=device(posterior_variance)

function q_sample(x_start, t, noise=nothing)
    if isnothing(noise)
        noise = randn_like(x_start,Float32)
    end
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t]
    sqrt_one_minus_alphas_cumprod_t =sqrt_one_minus_alphas_cumprod[t]
    @. sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
end 

include("loss.jl")

function p_losses(denoise_model,ps,st, x_start, t, loss::Function, noise=nothing)
    if isnothing(noise)
        noise = randn_like(x_start,Float32)
    end
    x_noisy = q_sample(x_start, t, noise)
    predicted_noise,st = denoise_model(x_noisy,ps,st;time=t)
    loss(noise,predicted_noise),st
end 

function p_sample(model,ps,st, x, t, t_index)
    betas_t=betas[t_index]
    sqrt_one_minus_alphas_cumprod_t=sqrt_one_minus_alphas_cumprod[t_index]
    sqrt_recip_alphas_t=sqrt_recip_alphas[t_index]
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean,_=model(x,ps,st, time=t)
    model_mean =@. sqrt_recip_alphas_t * (
        x - betas_t * model_mean / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0
        model_mean
    else
        posterior_variance_t = posterior_variance[t_index]
        noise = randn_like(x,Float32)
        # Algorithm 2 line 4:
        @. model_mean + sqrt(posterior_variance_t) * noise 
    end
end


function p_sample_loop(model,ps,st, shape)
    b = shape[end]
    # start from pure noise (for each example in the batch)
    img = randn(Float32,shape) |>device
    imgs = []

    for i in timesteps:-1:1
        img = p_sample(model,ps,st, img, fill(i,(b,))|>device, i)
        append!(imgs,[cpu(img)])
    end
    imgs 
end
