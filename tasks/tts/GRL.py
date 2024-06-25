from torch import nn
from torch.autograd import Function
import torch

import torch.nn.functional as F
from scipy.cluster.vq import kmeans2

class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5,
                 print_vq_prob=False):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.n_embeddings = n_embeddings
        self.decay = decay
        self.epsilon = epsilon
        self.print_vq_prob = print_vq_prob
        self.register_buffer('data_initialized', torch.zeros(1))
        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        B, _ = x.shape
        M, D = self.embedding.size()
        x_flat = x.detach()

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)  # [B, N_vq]
        indices = torch.argmin(distances.float(), dim=-1)  # [B]
        quantized = F.embedding(indices, self.embedding)
        # quantized = quantized.view_as(x)
        return x_flat, quantized, indices

    def forward(self, x):
        """

        :param x: [B, D]
        :return: [B, T, D]
        """
        B, _ = x.shape
        M, D = self.embedding.size()
        if self.training and self.data_initialized.item() == 0:
            print('| running kmeans in VQVAE')  # data driven initialization for the embeddings
            x_flat = x.detach()
            rp = torch.randperm(x_flat.size(0))
            x_float32 = x_flat[rp].float().data.cpu().numpy()
            kd = kmeans2(x_float32, self.n_embeddings, minit='points')
            self.embedding.copy_(torch.from_numpy(kd[0]))
            x_flat, quantized, indices = self.encode(x)
            encodings = F.one_hot(indices, M).float()
            self.ema_weight.copy_(torch.matmul(encodings.t(), x_flat))
            self.ema_count.copy_(torch.sum(encodings, dim=0))

        x_flat, quantized, indices = self.encode(x)
        encodings = F.one_hot(indices, M).float()
        indices = indices.reshape(B)

        if self.training and self.data_initialized.item() != 0:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)
        self.data_initialized.fill_(1)

        e_latent_loss = F.mse_loss(x, quantized.detach(), reduction='none')
        nonpadding = (x.abs().sum(-1) > 0).float()
        e_latent_loss = (e_latent_loss.mean(-1) * nonpadding).sum() / nonpadding.sum()
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        if self.print_vq_prob:
            print("| VQ code avg_probs: ", avg_probs)
        return quantized, loss, indices, perplexity

class EmotionClassifier_3(nn.Module):
    def __init__(self, emo_num, embed_dim, lambda_reversal):
        super(EmotionClassifier_3, self).__init__()
        
        self.classifier = nn.Sequential(
            GradientReversal(lambda_reversal),
            LinearNorm(embed_dim, embed_dim, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(embed_dim, embed_dim, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(embed_dim, emo_num, w_init_gain='linear')
        )
    
    def forward(self, x):
        outputs = self.classifier(x)  # (B, nb_speakers)
        
        return outputs

class EmotionClassifier(nn.Module):
    def __init__(self, emo_num, embed_dim, lambda_reversal):
        super(EmotionClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            GradientReversal(lambda_reversal),
            # LinearNorm(embed_dim, embed_dim, w_init_gain='relu'),
            # nn.ReLU(),
            # LinearNorm(embed_dim, embed_dim, w_init_gain='relu'),
            # nn.ReLU(),
            LinearNorm(embed_dim, emo_num, w_init_gain='linear')
        )
    
    def forward(self, x):
        outputs = self.classifier(x)  # (B, nb_speakers)
        
        return outputs
    
    
class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        x = self.linear_layer(x)  # (*, out_dim)
        
        return x
    
    
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_reversal):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_reversal

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class Emo_mlp(nn.Module):
    def __init__(self, embed_dim1, embed_dim2, embed_dim3):
        super(Emo_mlp, self).__init__()
        
        self.classifier = nn.Sequential(
            LinearNorm(embed_dim1, embed_dim2, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(embed_dim2, embed_dim3, w_init_gain='relu'),
        )
    
    def forward(self, x):
        outputs = self.classifier(x)  # (B, nb_speakers)
        
        return outputs