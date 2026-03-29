+++
title = "Sequence Features for Recommenders"
date = "2025-11-08"
type = "post"
description = "How to represent variable-length interaction histories as sequence embeddings."
in_search_index = true
[taxonomies]
tags = ["machine-learning", "recommender-systems"]
[extra]
+++

---

Sequence features capture interaction history over time, such as recently viewed items, clicked categories, or watched creators. They add temporal context that single ID features miss, and are often among the strongest signals in ranking models.

The common pattern is: map each token in the sequence to an embedding, then apply learning on top to extract signals from sequence history.

```python
class WeightedPooling(nn.Module):
    def __init__(self, vocab_size: int, dim: int, seq_len: int):
        super().__init__()
        self.position_logits = nn.Parameter(torch.zeros(seq_len))
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        w = F.softmax(self.position_logits, dim=0).unsqueeze(0).unsqueeze(-1)
        return (w * x).sum(dim=1)
```


*Practical notes:* cap max sequence length (for example 50/100), keep padding index fixed, and store sequence preprocessing logic consistently between training and serving.
