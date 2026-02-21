+++
title = "Loss Functions for Two-Tower Retrieval"
date = "2025-06-10"
type = "post"
description = "The objectives that make (or break) two-tower retrieval: in-batch softmax, sampled negatives, and pairwise losses."
in_search_index = true
[taxonomies]
tags = ["machine-learning", "recommender-systems"]
[extra]
+++

---

Two-Tower (TT) models are used in retrieval because they can efficiently score thousands of candidates with embedding search. Since it is impractical to compute softmax over all user and item pairs, a standard TT model uses a sampled softmax loss with in-batch negatives. Additionally, logQ correction is applied to prevent negative bias towards popular items that are more likely to be treated as negatives [1].

*Self-supervised Loss* 

As proposed in [2], learning better self-supervised item representations helps retrieval performance, especially on examples where labels are sparse. An auxiliary contrastive loss term is added, derived from multiple views of item embeddings generated with masking and augmentation of item features:

`Total loss = L_main + α * L_SSL`

Increasingly, TT models are being treated as early-stage rankers and loss functions that are applied to late-stage heavy rankers are being used to improve generalization of two-tower retrieval. 

*Pairwise Loss*

Often used to improve pairwise relevance of ranked results, this can be applied to TT loss by comparing scores of positive and negative examples as described in [3]:

`Total loss = L_main + β * max(0, m - (s(u, i+) - s(u, i-)))`

It is preferred to use hard negatives rather than in-batch negatives.


*Distillation Loss*

In addition to known binary labels from user feedback, prediction scores from late-stage ranker can serve as soft labels and the KL divergence between the TT predictions (q) and ranker predictions (p) can be treated as a loss function for training [4].  This can help the TT candidates align with the reranker, and more strongly drive topline metrics with TT modeling improvements.


`Total loss = L_main + β * DKL(p||q)`


*Practical notes:* Start with sampled softmax (baseline), add self-supervised loss, improve negatives (mixed negative sampling), add pairwise margin loss, add distillation loss last.


[1] [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://storage.googleapis.com/gweb-research2023-media/pubtools/5716.pdf), 2019

[2] [Self-supervised Learning for Large-scale Item Recommendations](https://arxiv.org/abs/2007.12865), Google 2020 

[3] [Semantic Search At LinkedIn](https://arxiv.org/abs/2602.07309), 2026

[4] [GPU-Serving Two-Tower Models for Lightweight Ads Engagement Prediction](https://medium.com/pinterest-engineering/gpu-serving-two-tower-models-for-lightweight-ads-engagement-prediction-5a0ffb442f3b), Pinterest 2026