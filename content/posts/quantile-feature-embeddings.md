+++
title = "Quantile Embeddings for Dense Features"
date = "2022-06-01"
type = "post"
description = "How to convert continuous features into learnable embeddings using quantile bucketization."
in_search_index = true
[taxonomies]
tags = ["machine-learning", "recommender-systems"]
[extra]
+++

---

Embeddings are commonly used to represent continuous features, and since these are trainable, often improve model performance at scale. 
Compared to raw feature values, which have limited expressivity and are sensitive to outliers, embeddings provide a rich and stable feature representation. 

Feature bucketization with linear boundaries is one technique to map a continuous value into an embedding space, however, can suffer from skewed representation. 
The fix is to use quantile bucketization where the quantile boundaries (10%, 20% ... 90%) guarantee roughly equal training examples per bucket.
 

![Quantile bucketization to embedding lookup](/images/quantile-buckets-diagram.png)

```python
class QuantileEmbedding(nn.Module):
    def __init__(self, q_boundaries: List[float], embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(len(q_boundaries) + 1, embedding_dim)
        self.register_buffer("q_boundaries", torch.Tensor(q_boundaries))

    def forward(self, input: torch.Tensor):
        indices = torch.bucketize(input, self.q_boundaries)
        return self.embeddings(indices)

    def ordinal_loss(self) -> torch.Tensor:
        w = self.embeddings.weight  # [num_buckets, embedding_dim]
        return torch.mean((w[1:] - w[:-1]) ** 2)
```

Bucketization discards ordinal structure as nearby embeddings are learned independently. We can use an `ordinal_loss` to penalize the squared L2 distance between adjacent embeddings.
It can be used as a weighted auxiliary term to the main loss:

`L = L_main + lambda * quantile_embedding.ordinal_loss`

Quantile embeddings are now a standard in most large-scale recommendation systems, as mentioned in [DCN V2 paper](https://arxiv.org/abs/2008.13535).

*Practical notes:* use 5/10 quantile boundaries, can be fixed or generated daily as training artifacts; use 2/4/8 embedding dimension per float feature
