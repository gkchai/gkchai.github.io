+++
title = "Recommender Systems: A Curated List of Papers"
date = "2025-10-01"
type = "post"
description = "A curated reading list of recommender systems papers, from ranking and retrieval to generative recommendations and RL-aligned ranking."
in_search_index = true
[taxonomies]
tags = ["machine-learning", "recommender-systems"]
[extra]
+++

---

My reading list for recommender systems, organized by the areas that matter most in production.

## Value Modeling & Calibration

The utility function that combines action probabilities into a single ranking score is the conduit between the ML model and the product. In most cases, the actions are determined by the product definition, e.g. like, watch duration, dwell time, share, purchase. There are also actions that are outside the immediate user session and significantly impact long-term behavior, e.g. number of play days over the next 7 days, repeat purchases. Calibration becomes important when we want to associate monetary value to predictions, for instance, in ads ranking, or when the ranking score is a weighted sum of probabilities across heterogeneous models or actions.

- [Learned Ranking Function: From Short-term Behavior Predictions to Long-term User Satisfaction](https://arxiv.org/abs/2408.06512) -- Google/YouTube, 2024
- [Ranking with Long-Term Constraints](https://arxiv.org/abs/2307.04923) -- Google, 2023
- [What We Know About Using Non-Engagement Signals in Content Ranking](https://integrityinstitute.org/research/what-we-know-about-using-non-engagement-signals-in-content-ranking) -- Integrity Institute, 2024
- [Multi-Objective Recommendation via Multivariate Policy Learning](https://arxiv.org/abs/2405.02141) -- Spotify, 2024
- [Feedback Shaping: A Modeling Approach to Nurture Content Creation](https://arxiv.org/abs/2106.11541) -- LinkedIn, 2021
- [Multi-task Learning and Calibration for Utility-based Home Feed Ranking](https://dl.acm.org/doi/10.1145/3394486.3403392) -- Pinterest, 2020
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) -- Cornell, 2017
- [Why Model Calibration Matters and How to Achieve It](https://research.google/blog/why-model-calibration-matters-and-how-to-achieve-it/) -- Google, 2021

## Multi-task Ranking

Shared-bottom networks can suffer from negative transfer across tasks. MMoE and PLE introduce gating to decouple conflicting gradients through soft parameter sharing. ESMM solves sample selection bias in conversion modeling by jointly predicting over the entire impression space. Uncertainty weighting and GradNorm are well-known methods to automatically balance task losses, while PCGrad and MultiBalance directly resolve conflicting gradients.

- [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/10.1145/3219819.3220007) -- Google, 2018
- [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236) -- Tencent, 2020
- [Recommending What Video to Watch Next: A Multitask Ranking System](https://dl.acm.org/doi/10.1145/3298689.3346997) -- Google, 2019
- [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931) -- Alibaba, 2018
- [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115) -- Cambridge, 2018
- [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://arxiv.org/abs/1711.02257) -- ICML, 2018
- [Gradient Surgery for Multi-Task Learning (PCGrad)](https://arxiv.org/abs/2001.06782) -- Stanford/Google, 2020
- [MultiBalance: Multi-Objective Gradient Balancing in Industrial-Scale Multi-Task Recommendation System](https://arxiv.org/abs/2411.11871) -- Meta, 2024
- [Improving Training Stability for Multitask Ranking Models in Recommender Systems](https://arxiv.org/abs/2302.09178) -- Google, 2023

## Feature Interactions

Good feature architectures can solve the information bottleneck and significantly improve model performance. Deep and wide feature crossing (DCN) is the current standard. DCN V2 replaces DCN v1's rank-1 cross layer with a full-rank (or mixture-of-low-rank) cross network. DHEN ensembles heterogeneous interaction modules in a deep hierarchy, capturing non-overlapping interaction patterns. Wukong stacks factorization machines to learn any-order interactions and demonstrates a scaling law across two orders of magnitude in model FLOPs. RankMixer replaces handcrafted crossing modules with hardware-aware token mixing, scaling ranking models by 100x parameters at similar latency.

- [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535) -- Google, 2020
- [DHEN: A Deep and Hierarchical Ensemble Network for Large-Scale Click-Through Rate Prediction](https://arxiv.org/abs/2203.11014) -- Meta, 2022
- [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/abs/2403.02545) -- Meta, 2024
- [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551) -- 2025

## Sequence Modeling & Attention

User action history is one of the strongest ranking signals. The key shift was moving from bag-of-interactions to ordered sequences with attention. DIN introduced target-aware attention (candidate as query, history as keys), and is effective in practice. Self-attention captures inter-action relationships. Graph attention handles relational structure like social networks.

- [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978) -- Alibaba, 2017
- [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1905.06874) -- Alibaba, 2019
- [Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction](https://arxiv.org/abs/2006.05639) -- Alibaba, 2020
- [TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest](https://arxiv.org/abs/2306.00087) -- Pinterest, 2023
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) -- Google, 2017
- [Exposing Attention Glitches with Flip-Flop Language Modeling](https://arxiv.org/abs/2305.16102) -- Stanford, 2023

## Embeddings at Scale

At hundreds of millions of IDs, naive embedding tables hit memory limits and hash collisions silently degrade quality. Monolith solves this with collisionless tables. Frozen pretrained embeddings (XLM, E5) inject semantic understanding without end-to-end training cost, and are especially useful for cold-start items with text metadata but no behavioral signal.

- [Monolith: Real Time Recommendation System With Collisionless Embedding Table](https://arxiv.org/abs/2209.07663) -- ByteDance, 2022
- [Efficient Data Representation Learning in Google-scale Systems](https://arxiv.org/abs/2309.07572) -- Google, 2023
- [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) -- Facebook AI, 2019
- [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533) -- Microsoft, 2022

## Two Tower Retrieval

One tower for the user, one for the item, dot product for scoring, ANN index for serving. In-batch negatives over-penalize popular items since they appear as negatives disproportionately often. LogQ correction (Yi et al., 2019) fixes this. 

- [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/10.1145/2959100.2959190) -- Google, 2016
- [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://dl.acm.org/doi/10.1145/3298689.3346996) -- Google, 2019
- [Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/) -- Google, 2020
- [Self-supervised Learning for Large-scale Item Recommendations](https://arxiv.org/abs/2007.12865) -- Google, 2020
- [Cross-Batch Negative Sampling for Training Two-Tower Recommenders](https://arxiv.org/abs/2110.15154) -- Huawei, 2021
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) -- Russian Academy of Sciences, 2016
- [Deep Retrieval: Learning A Retrievable Structure for Large-Scale Recommendations](https://arxiv.org/abs/2007.07203) -- ByteDance, 2020
- [Full Index Deep Retrieval: End-to-End User and Item Structures for Cold-start and Long-tail Item Recommendation](https://arxiv.org/abs/2309.07402) -- ByteDance/SJTU, 2023

## Exploration

Popurality bias in recommendations is a long-standing issue. Fresh items have no signal, so models trained on historical data are biased against them. Standard A/B tests compound the problem because they penalize exploration by comparing signal-rich control against signal-poor variants.

- [Values of User Exploration in Recommender Systems](https://dl.acm.org/doi/10.1145/3460231.3474236) -- Google, 2021
- [Long-Term Value of Exploration: Measurements, Findings and Algorithms](https://arxiv.org/abs/2305.09498) -- Google, 2023
- [Nonlinear Bandits Exploration for Recommendations](https://arxiv.org/abs/2311.14592) -- Google, 2023
- [Online Matching: A Real-time Bandit System for Large-scale Recommendations](https://dl.acm.org/doi/10.1145/3580305.3599882) -- Google, 2023
- [Fresh Content Needs More Attention: Multi-funnel Fresh Content Recommendation](https://dl.acm.org/doi/10.1145/3580305.3599881) -- Google, 2023

## Generative Recommendations

BERT4Rec and HSTU showed recommendation works as next-item prediction. TIGER proved you can tokenize items into semantic IDs via hierarchical clustering and generate recommendations autoregressively. OneRec replaced an entire multi-stage pipeline with one generative model, achieving better app stay time with dramatically lower system complexity. A recent Google paper argues the real win is the clustering, not the autoregressive generation. Full softmax over item clusters gets equivalent results 10x faster.

- [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690) -- Alibaba, 2019
- [Effective and Efficient Training for Sequential Recommendation using Recency Sampling](https://arxiv.org/abs/2207.02643) -- University of Glasgow, 2022
- [Learning from Negative User Feedback and Measuring Responsiveness for Sequential Recommenders](https://arxiv.org/abs/2308.12256) -- Google, 2023
- [Recommender Systems with Generative Retrieval (TIGER)](https://arxiv.org/abs/2305.05065) -- Google, 2023
- [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152) -- Meta, 2024
- [OneRec: Unifying Retrieve and Rank with Generative Recommendation](https://arxiv.org/abs/2506.13695) -- Kuaishou, 2025
- [Semantic ID Based Recommendation: An Order of Magnitude Faster with Comparable Quality](https://arxiv.org/abs/2509.03746) -- Google, 2025


## Others

- [Practical Lessons from Predicting Clicks on Ads at Facebook](https://dl.acm.org/doi/10.1145/2648584.2648589) -- Facebook, 2014
- [Trustworthy Online Marketplace Experimentation with Budget-split Design](https://arxiv.org/abs/2012.08724) -- LinkedIn, 2020
- [Towards Understanding the Overfitting Phenomenon of Deep Click-Through Rate Prediction Models](https://arxiv.org/abs/2209.06053) -- Alibaba, 2022
- [Fairness in Recommendation Ranking through Pairwise Comparisons](https://arxiv.org/abs/1903.00780) -- Google, 2019
