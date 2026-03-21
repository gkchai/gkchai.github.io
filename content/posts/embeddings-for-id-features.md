+++
title = "Embeddings for ID Features"
date = "2025-11-01"
type = "post"
description = "How to represent sparse categorical IDs with embeddings."
in_search_index = true
[taxonomies]
tags = ["machine-learning", "recommender-systems"]
[extra]
+++

---

ID features (`user_id`, `item_id`, `author_id`) are high-cardinality categorical signals. Instead of one-hot vectors, which scale poorly and cannot capture similarity between IDs, embeddings are both efficient and expressive. The embedding mapping can be done with a fixed vocab with a out-of-vocab (OOV) embeddings, or through random hashing.


![VocabEmbed and HashEmbed lookup diagram](/images/vocabembed-vs-hashembed-diagram.svg)

![VocabEmbed and HashEmbed lookup diagram](/images/vocabembed-vs-hashembed-diagram.svg)

```python
class VocabEmbed(nn.Module):
    def __init__(self, vocab: List[int], dim: int):
        super().__init__()
        self.register_buffer("vocab", torch.tensor(sorted(set(vocab)), dtype=torch.long))
        self.emb = nn.Embedding(self.vocab.shape[0], dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.to(dtype=torch.long, device=self.vocab.device)
        pos = torch.searchsorted(self.vocab, ids)
        in_vocab = (pos < self.vocab.shape[0]) & (self.vocab[pos.clamp(max=self.vocab.shape[0] - 1)] == ids)
        if not torch.all(in_vocab):
            raise ValueError("ids contain values not in vocab")
        return self.emb(pos)


class HashEmbed(nn.Module):
    def __init__(self, num_buckets: int, dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        bucket = torch.remainder(ids.to(dtype=torch.long), self.num_buckets)
        return self.emb(bucket)

```

*Practical notes:* 32/48/64 are commonly used embedding dimensions. Vocab can be generated daily or weekly from top IDs in the training data, set to a max limit, e.g. 0.5M, depending on the corpus size.
