/* eslint-disable */

export const markdownData = `
# SAEBench: A Comprehensive Benchmark for Sparse Autoencoders

Adam Karvonen*, Can Rager*, Johnny Lin*, Curt Tigges*, Joseph Bloom*,
David Chanin, Yeu-Tong Lau, Eoin Farrell, Arthur Conmy, Callum McDougall, Kola Ayonrinde, Matthew Wearden,
Samuel Marks, Neel Nanda
\*\*equal contribution*

## TL;DR

- We are releasing SAE Bench, a suite of 8 diverse sparse autoencoder (SAE) evaluations including unsupervised metrics and downstream tasks. Use [our codebase](https://github.com/adamkarvonen/SAEBench) to evaluate your own SAEs!
- You can compare 200+ SAEs of varying sparsity, dictionary size, architecture, and training time on [Neuronpedia](https://neuronpedia.org/sae-bench).
- Think we're missing an eval? We'd love for you to contribute it to our codebase! [Email us](mailto:adam.karvonen@gmail.com,canrager@gmail.com).

<div className="flex flex-col gap-y-2 justify-start mt-2 min-w-[280px] max-w-[280px]">

<a href="https://www.neuronpedia.org/sae-bench" target="_blank" rel="noopener noreferrer" className="ignore-saebench gap-x-1.5 flex flex-row items-center rounded-full bg-sky-600 px-4 py-2.5 text-xs font-bold text-white hover:bg-sky-700 sm:ml-1 sm:flex">
    🔍 Explore the Benchmark & Rankings
</a>

<a href="https://github.com/adamkarvonen/SAEBench" target="_blank" rel="noopener noreferrer" className="ignore-saebench gap-x-1.5 flex flex-row items-center rounded-full bg-black px-4 py-2.5 text-xs font-bold text-white hover:bg-slate-800 sm:ml-1 sm:flex">
    <svg className="inline h-4 w-4" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
    Evaluate your SAEs with SAEBench
</a>

<a href="mailto:adam.karvonen@gmail.com,canrager@gmail.com" target="_blank" rel="noopener noreferrer" className="ignore-saebench gap-x-1.5 flex flex-row items-center rounded-full bg-slate-600 px-4 py-2.5 text-xs font-bold text-white hover:bg-slate-700 sm:ml-1 sm:flex">
    ✉️ Contact Us
</a>

</div>

# March 2025 Update
This blog post was originally published with the SAE Bench beta release on December 10, 2024. We are now releasing SAE Bench 1.0. We’ve made substantial improvements, added new SAE architectures, and conducted further evaluations that qualitatively alter some of our earlier findings—these key updates are summarized below. For up to date and comprehensive details, please refer to the SAE Bench paper [(https://arxiv.org/abs/2503.09532)](https://arxiv.org/abs/2503.09532). For comparability, we leave our initial blogpost from December 2024 below. Neuronpedia's [SAEBench Evaluation browser](/sae-bench) features both the new results (\`SAEBench – Jan 25\`) and previous results (\`SAEBench – Dec 24\`) as separate releases.

## New suite of open-source SAEs
We made minor changes to the training code for ReLU SAEs, added Matryoshka SAEs, and standardized the training hyperparameters across SAEs. The new suite of SAEBench SAEs includes 7 variants:

- ReLU (Anthropic April Update)
- TopK
- BatchTopK
- JumpReLU
- Gated
- P-anneal
- Matryoshka BatchTopK

Trained across 3 widths (4k, 16k, and 65k), 6 sparsities (~20 to ~640), on layer 8 of Pythia-160M and layer 12 of Gemma-2-2B. Additionally, we have checkpoints throughout training for TopK and ReLU variants for Gemma-2-2B 16k and 65k widths. The SAEs are located in the following HuggingFace repos:

- [Pythia-160M 4k width](https://huggingface.co/adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108)
- [Pythia-160M 16k width](https://huggingface.co/adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108)
- [Pythia-160M 65k width](https://huggingface.co/adamkarvonen/saebench_pythia-160m-deduped_width-2pow16_date-0108)
- [Gemma-2-2B 4k width](https://huggingface.co/adamkarvonen/saebench_gemma-2-2b_width-2pow12_date-0108)
- [Gemma-2-2B 16k width](https://huggingface.co/canrager/saebench_gemma-2-2b_width-2pow14_date-0107)
- [Gemma-2-2B 65k width](https://huggingface.co/canrager/saebench_gemma-2-2b_width-2pow16_date-0107)

Find the evaluation results in the SAEBench interface by selecting the \`SAEBench - Jan 25\` release. 

## New qualitative findings

### Matryoshka SAEs perform best on 5 out of 8 metrics
In the typically used L0 range of 20-200, Matryoshka SAEs perform best on the TPP, SCR, RAVEL, and Feature Absorption metrics, and near best on the Sparse Probing metric. However, Matryoshkas do slightly underperform on the traditional sparsity - fidelity tradeoff. These differences are most apparent at the largest scale of 65k width, and improvements may be diminished or non-existent at smaller scales.

![Scores for the Loss Recovered, Automated Interpretability, Absorption, SCR, and Sparse Probing metrics on the 65k width Gemma-2-2B suite of SAEs.](/saebench/plot_2x4_sae_bench_gemma-2-2b_65k_architecture_series_layer_12.png)
_Scores for the Loss Recovered, Automated Interpretability, Absorption, SCR, and Sparse Probing metrics on the 65k width Gemma-2-2B suite of SAEs._

We also evaluated several proposed SAE approaches (TopK, BatchTopK, P-Anneal, Gated, JumpReLU) that had primarily focused on improving reconstruction accuracy. We observed it is often difficult to differentiate these approaches on our SAE Bench metrics. These findings emphasize the need for diverse metrics beyond proxies such as reconstruction accuracy.


### Matryoshka Scaling Dynamics
Previously we had observed that there is no size / sparsity combination that appears to balance performance on all metrics. The SCR and Feature Absorption metrics had worse performance with scale, also called inverse scaling, for all previously tested SAE architectures. However, Matryoshka SAEs do not appear to have these trade-offs, and generally improve or maintain performance with increasing scale. A large Matryoshka SAE with an L0 of 50-150 may offer strong performance across a range of tasks, and we would be excited to see further investigation into the usage of Matryoshka SAEs.

![Scaling SAE width from 4k to 65k for across SAE architectures. For each architecture / width pair, we mean over all results in the L0 range between 40 and 200. Most notably the hierarchical Matryoshka SAE shows positive scaling behavior. Due to varying L0 distributions across architectures, this visualization is intended primarily for analyzing scaling
trends rather than architecture comparisons.](/saebench/plot_2x3_sae_bench_gemma-2-2b_scaling_width_series_layer_12.png)
_Scaling SAE width from 4k to 65k for across SAE architectures. For each architecture / width pair, we mean over all results in the L0 range between 40 and 200. Most notably the hierarchical Matryoshka SAE shows positive scaling behavior. Due to varying L0 distributions across architectures, this visualization is intended primarily for analyzing scaling
trends rather than architecture comparisons._

### Feature absorption in ReLU SAEs
In our original blog post, we had found that ReLU SAEs had decreased levels of feature absorption. When training our baseline suite of SAEs, we improved our ReLU training approach from the original Towards Monosemanticity approach to the [Anthropic April Update](https://transformer-circuits.pub/2024/april-update/index.html) approach. After this improvement, we found that ReLU SAEs actually had the highest levels of feature absorption. After some inspection, we believe that the decreased feature absorption in our original results was due to a high percentage of dead features.


### RAVEL: Evaluating causal effects on model generation.
If an SAE effectively captures independent concepts, each should be encoded by dedicated latents, achieving clear disentanglement. To measure this, we implement the RAVEL (Resolving Attribute–Value Entanglements in Language Models) evaluation from [Huang et al](https://arxiv.org/abs/2402.17700v2) which tests how cleanly interpretability methods separate related attributes within language models. We’re excited to add this metric as it directly evaluates causal effects on downstream behavior. We find that RAVEL scores generally increase with L0, and for width 65k sparsity values between 40-200, Matryoshka performs best.


RAVEL evaluates whether targeted interventions on SAE latents can selectively change a model’s predictions for specific attributes without unintended side effects—for instance, making the model believe Paris is in Japan while preserving the knowledge that the language spoken remains French. Concretely, RAVEL works as follows: given prompts like "Paris is in the country of France," "People in Paris speak the language French" and "Tokyo is a city” we encode the tokens Paris and Tokyo using the SAE. We train a binary mask to transfer latent values from Tokyo to Paris, decode the modified latents, and insert them back into the residual stream for the model to generate completions. The final disentanglement score averages two metrics: the Cause Metric, measuring successful attribute changes due to the intervention, and the Isolation Metric, verifying minimal interference with other attributes.

### Randomly initialized models
Inspired by [Heap et al](https://arxiv.org/abs/2501.17727v1), we evaluated SAEs trained on randomly initialized models as a baseline. We find that SAEs on trained models generally obtain significantly higher scores on our metrics, except for TPP. However, we caution that most of our metrics are meant for comparisons of different SAEs on the same model, and results should be interpreted with caution. For example, the TPP metric measures relative changes in probe accuracy within a given model. Given that the linear probes on the randomly initialized model start from a substantially lower baseline accuracy, direct comparisons of TPP across models may be misleading.

![Evaluation results for SAEs trained on the randomly initialized and final versions of Pythia-1B.](/saebench/pythia-1b-random.png)
_Evaluation results for SAEs trained on the randomly initialized and final versions of Pythia-1B._


# December 2024 Post
## Introduction

Sparse Autoencoders (SAEs) have become one of the most popular tools for AI interpretability. A lot of recent interpretability work has been focused on studying SAEs, in particular on improving SAEs, e.g. the [Gated SAE](https://arxiv.org/abs/2404.16014), [TopK SAE](https://cdn.openai.com/papers/sparse-autoencoders.pdf), [BatchTopK SAE](https://www.alignmentforum.org/posts/Nkx6yWZNbAsfvic98/batchtopk-a-simple-improvement-for-topk-saes), [ProLu SAE](https://www.alignmentforum.org/posts/HEpufTdakGTTKgoYF/prolu-a-nonlinearity-for-sparse-autoencoders), [Jump Relu SAE](https://arxiv.org/abs/2407.14435), [Layer Group SAE](https://arxiv.org/abs/2410.21508), [Feature Choice SAE](https://openreview.net/forum?id=TIjBKgLyPN), [Feature Aligned SAE](https://openreview.net/forum?id=NB8qn8iIW9), and [Switch SAE](https://arxiv.org/abs/2410.08201). But how well do any of these improvements actually work?

The core challenge is that we don't know how to measure how good an SAE is. The fundamental premise of SAEs is a useful interpretability tool that unpacks concepts from model activations. The lack of ground truth labels for model internal features led the field to measure and optimize the proxy of sparsity instead. This objective successfully provided interpretable SAE latents. But sparsity has known problems as a proxy, such as [feature absorption](https://arxiv.org/pdf/2409.14507v3) and [composition of independent features](https://www.alignmentforum.org/posts/TMAmHh4DdMr4nCSr5/showing-sae-latents-are-not-atomic-using-meta-saes). Yet, most SAE improvement work merely measures whether reconstruction is improved at a given sparsity, potentially missing problems like uninterpretable high frequency latents, or increased composition.

In the absence of a single, ideal metric, we argue that the best way to measure SAE quality is to give a more detailed picture with a range of diverse metrics. In particular, SAEs should be evaluated according to their performance on downstream tasks, a robust signal of usefulness.

Our comprehensive benchmark provides insight to fundamental questions about SAEs, like what the ideal sparsity,training time, and other hyperparameters. To showcase this, we've trained a custom suite of 200+ SAEs of varying dictionary size, sparsity, training time, and architecture (holding all else constant). Browse the evaluation results covering Pythia-70m and Gemma-2-2B on [Neuronpedia](https://neuronpedia.org/sae-bench).

SAEBench enables a range of use cases, such as measuring progress with new SAE architectures, revealing unintended SAE behavior, tuning training hyperparameters, and selecting the best SAE for a particular task. We find that these evaluation results are nuanced and there is no one ideal SAE configuration - instead, the best SAE varies depending on the specifics of the downstream task. Because of this, we cannot combine the results into a single number without obscuring tradeoffs. Instead, we provide a range of quantitative metrics so that researchers can measure the nuanced effects of experimental changes.

We are releasing a beta version of [SAEBench](https://github.com/adamkarvonen/SAEBench), including a [convenient demonstration notebook](https://github.com/adamkarvonen/SAEBench/blob/main/sae_bench_demo.ipynb) that evaluates custom SAEs on multiple benchmarks and plots the results. Our flexible codebase allows you to easily add your own evaluations.

## Metrics

What makes a good SAE? We evaluate several desirable properties with dedicated metrics.
![SAEBench Metrics](/saebench/whatmakesagoodsae.png)

Unfold the metrics below for detailed descriptions.

#### Unsupervised Metrics (Core)

<details id="details-unsupervised-metrics" class="anchor">
<summary>Show Details</summary>

We provide implementations to measure sparsity and reconstruction error, the most commonly used metrics for comparing SAEs. We measure both the reconstruction error for the model's activations (Mean Squared Error aka MSE, fraction of variance explained, cosine similarity) and for the model's output logits (KL divergence, and change in cross-entropy loss). We also provide a range of unsupervised metrics, including feature density statistics, the L1 sparsity, and measures of shrinkage such as the L2 ratio and relative reconstruction bias.

| Metric                     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| L0 Sparsity                | <ul><li>Measures the average number of non-zero feature activations. A lower number indicates higher sparsity.</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Cross Entropy Loss Score   | <ul><li>Measures the change in model performance when replacing activations with the corresponding SAE reconstruction during a forward pass.</li><li>This metric is quantified as \`(H∗ − H_0) / (H_orig − H_0)\`, where H_orig is the cross-entropy loss of the original model for next-token prediction, H∗ is the cross-entropy loss after substituting the model activation x with its SAE reconstruction during the forward pass, and H_0 is the cross-entropy loss when zero-ablating x.</li><li>Higher is better. The range of the metric is 0-1.</li></ul> |
| Feature Density Statistics | <ul><li>We track the activation frequency for every SAE latent.</li><li>This can be informative for both the proportion of dead features that never activate (indicating wasted capacity) and dense features that activate with a very high frequency (which tend to be less interpretable).</li><li>These can also be visualized as a log-scale histogram.</li></ul>                                                                                                                                                                                              |
| L2 Ratio                   | <ul><li>Compares the L2 norm (Euclidean distance) of different components, typically between the original and reconstructed representations.</li><li>Helps assess how well the model preserves the magnitude of the input signals.</li></ul>                                                                                                                                                                                                                                                                                                                       |
| Explained Variance         | <ul><li>Quantifies how much of the variability in the data is captured by the model.</li><li>Higher values (closer to 1) indicate the model better explains the patterns in the input data.</li></ul>                                                                                                                                                                                                                                                                                                                                                              |
| KL Divergence              | <ul><li>Kullback-Leibler divergence measures the difference between two probability distributions, often used to compare the model's output distribution to the target distribution.</li><li>Lower values indicate better alignment.</li></ul>                                                                                                                                                                                                                                                                                                                     |

<iframe
  src="/sae-bench?modelId=gemma-2-2b&dSae=65536&layer=12&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=core%7C%7Cmodel_performance_preservation%7C%7Cce_loss_score&groupBy=saeClass&embed=true"
  className="h-[400px] max-h-[400px] border-0 w-full mt-3 mb-5"
  loading="lazy"
></iframe>

</details>

#### Feature Absorption

<details id="details-feature-absorption" class="anchor">
<summary>Show Details</summary>

Sparsity incentivizes an undesirable phenomenon called [feature absorption](https://arxiv.org/pdf/2409.14507v3). Imagine an SAE learned two distinct latents tracking the features "starts with S" and "short". Since "short" always starts with S, the SAE can increase sparsity by absorbing the "starts with S" feature into the "short" latent and then no longer needs to fire the "starts with S" latent when the token "short" is present, as it already includes the "starts with S" feature direction.

<img src="/saebench/feature-absorption.png" alt="Feature Absorption" className="sm:w-[75%]">

_Credit: Callum McDougall, [ARENA Tutorials](https://colab.research.google.com/drive/1ePkM8oBHIEZ2kcqAiA3waeAmz8RSdHmq#scrollTo=6lix5OebZ38q)_

In general, feature absorption is incentivised any time there's a pair of concepts, A & B, where A implies B (i.e. if A activates then B will always also be active, but not necessarily the other way round). This will happen with categories/hierarchies, e.g. India => Asia, pig => mammal, red => color, etc. If the SAE learns a latent for A and a latent for B, then both will fire on inputs with A. But this is redundant–A implies B, so there's no need for the B latent to light up on A. And if the model learns a latent for A and a latent for "B except for A", then only one activates. This is sparser, but clearly less interpretable!

Feature absorption often happens in an unpredictable manner, resulting in unusual gerrymandered features. For example, the "starts with S" feature may fire on 95% of tokens beginning with S, yet fail to fire on an arbitrary 5% as the "starts with S" feature has been absorbed for this 5% of tokens. This is an undesirable property that we would like to minimize.

To quantify feature absorption, we follow the example in [Chanin et al.](https://arxiv.org/pdf/2409.14507v3) and use a first letter classification task. First, tokens consisting of only English letters and an optional leading space are split into a train and test set, and a supervised logistic regression probe is trained on the train set using residual stream activations from the model. This probe is used as ground truth for the feature direction in the model. Next, k-sparse probing is performed on SAE latents from the train set to find which latents are most relevant for the task. The k=1 sparse probing latent is considered as a main SAE latent for the first letter task. To account for feature splitting, as k is increased from k=n to k=n+1, if the F1 score for the k=n+1 sparse probe represents an increase of more than τ\_{fs} than the F1 of the k=n probe, the k=n+1 feature is considered a feature split and is added to the set of main SAE latents performing the first letter task. We use τ_fs=0.03 in line with [Chanin et al](https://arxiv.org/pdf/2409.14507v3).

After the main feature split latents for the first letter task are found, we look for test set examples where the main feature split latents fail to correctly classify the token, but the logistic regression probe is able to correctly classify the sample. We then look for a different SAE latent that fires on this sample that has a cosine similarity with the probe of at least τ*{ps}, and where the SAE latent accounts for at least τ*{pa} portion of the probe projection in the activation. We use τ*{ps}=0.025 and τ*{pa}=0.4 in line with [Chanin et al.](https://arxiv.org/pdf/2409.14507v3).

This probe projection criteria is different from the metric in [Chanin et al.](https://arxiv.org/pdf/2409.14507v3), where a minimum latent ablation effect is used instead. However, latent ablation cannot work at later layers of the model after the model has moved first-letter information from the origin token to the prediction point, and thus limits the applicability of the metric. Using a probe direction contribution criteria allows us to apply this metric at all layers of the model and is thus more suitable for a benchmark.

<iframe
  src="/sae-bench?modelId=gemma-2-2b&dSae=65536&layer=12&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=absorption_first_letter%7C%7Cmean%7C%7Cmean_full_absorption_score&groupBy=saeClass&embed=true"
  className="h-[400px] max-h-[400px] border-0 w-full mt-3 mb-5"
  loading="lazy"
></iframe>

</details>

#### Unlearning

<details id="details-unlearning" class="anchor">
<summary>Show Details</summary>

We evaluate SAEs on their ability to selectively remove knowledge while maintaining model performance on unrelated tasks, following the methodology in [Applying sparse autoencoders to unlearn knowledge in language models](https://arxiv.org/abs/2410.19278).

This SAE unlearning evaluation uses the [WMDP-bio](https://www.wmdp.ai/) dataset, which contains multiple-choice questions containing dangerous biology knowledge. The intervention methodology involves clamping selected SAE feature activations to negative values whenever the features activate during inference. Feature selection utilizes a dual-dataset approach: calculating feature sparsity across a "forget" dataset (WMDP-bio corpus) and a "retain" dataset (WikiText). The selection and intervention process involves three key hyperparameters:

1. \`retain_threshold\` - maximum allowable sparsity on the retain set
2. \`n_features\` - number of top features to select
3. \`multiplier\` - magnitude of negative clamping

The procedure first discards features with retain set sparsity above retain_threshold, then selects the top n_features by forget set sparsity, and finally clamps their activations to negative multiplier when activated.

We quantify unlearning effectiveness through two metrics:

1. Accuracy on WMDP-bio questions
2. Accuracy on biology-unrelated MMLU subsets including:

- High school US history
- Geography
- College computer science
- Human aging

Both metrics only evaluate on questions that the base model answers correctly across all option permutations, to reduce noise from uncertain model knowledge. Lower WMDP-bio accuracy indicates successful unlearning, while higher MMLU accuracy demonstrates preserved general capabilities.

We sweep the three hyperparameters to obtain multiple evaluation results per SAE. To derive a single evaluation metric, we filter for results maintaining MMLU accuracy above 0.99 and select the minimum achieved WMDP-bio accuracy, thereby measuring optimal unlearning performance within acceptable side effect constraints.

<iframe
  src="/sae-bench?modelId=gemma-2-2b&dSae=65536&layer=12&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=unlearning%7C%7Cunlearning%7C%7Cunlearning_score&groupBy=saeClass&embed=true"
  className="h-[400px] max-h-[400px] border-0 w-full mt-3 mb-5"
  loading="lazy"
></iframe>

</details>

#### Spurious Correlation Removal (SCR)

<details id="details-scr" class="anchor">
<summary>Show Details</summary>

In the [SHIFT method](https://arxiv.org/abs/2403.19647), a human evaluator debiases a classifier by ablating SAE latents. We [automate SHIFT](https://arxiv.org/abs/2411.18895) and use it to measure whether an SAE has found separate latents for distinct concepts – the concept of gender, and the concepts for someone's profession for example. Distinct features will enable a more precise removal of spurious correlations, thereby effectively debiasing the classifier.

First, we filter datasets ([Bias in Bios](https://arxiv.org/abs/1901.09451) and [Amazon Reviews](https://amazon-reviews-2023.github.io/)) for two binary labels. For example, we select text samples of two professions (professor, nurse) and the gender labels (male, female) from the Bias in Bios dataset. We partition this dataset into a balanced set—containing all combinations of professor/nurse and male/female—and a biased set that only includes male+professor and female+nurse combinations. We then train a linear classifier $C_b$ on the biased dataset. The linear classifier picks up on both signals, such as gender and profession.

During the evaluation, we attempt to debias the classifier $C_b$ by selecting SAE latents related to one class (eg. gender) to increase classification accuracy for the other class (eg. profession).

We select set $L$ containing the top $n$ SAE latents according to their absolute probe attribution score with a probe trained specifically to predict the spurious signal (eg. gender). [Karvonen et al.](https://arxiv.org/abs/2411.18895) found that the scores obtained with feature selection through probe attribution had a strong correlation with scores obtained with feature selection using an LLM judge. Thus, we select features using probe attribution to avoid the cost and potential biases associated with an LLM judge.

For each original and spurious-feature-informed set L of selected features, we remove the spurious signal by defining a modified classifier $C_m = C_b \ L$ where all selected unrelated yet with high attribution latents are zero-ablated. The accuracy with which the modified classifier $C_m$ predicts the desired class when evaluated on the balanced dataset indicates SAE quality. A higher accuracy suggests that the SAE was more effective in isolating and removing the spurious correlation of e.g. gender, allowing the classifier to focus on the intended task of e.g. profession classification.

We consider a normalized evaluation score:

$$
S_{SHIFT} = (A_{abl} - A_{base}) / (A_{oracle} - A_{base})
$$

where, $A_{abl}$ is the probe accuracy after ablation, $A_{base}$ is the baseline accuracy (spurious probe before ablation), and $A_{oracle}$ is the skyline accuracy (probe trained directly on the desired concept).

This score represents the proportion of improvement achieved through ablation relative to the maximum possible improvement, allowing fair comparison across different classes and models.

<iframe
  src="/sae-bench?modelId=gemma-2-2b&dSae=65536&layer=12&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=scr%7C%7Cscr_metrics%7C%7Cscr_metric_threshold_20&groupBy=saeClass&embed=true"
  className="h-[400px] max-h-[400px] border-0 w-full mt-3 mb-5"
  loading="lazy"
></iframe>

</details>

#### Targeted Probe Perturbation (TPP)

<details id="details-tpp" class="anchor">
<summary>Show Details</summary>

SHIFT requires datasets with correlated labels. We generalize SHIFT to all multiclass NLP datasets by introducing the targeted probe perturbation (TPP) metric. At a high level, we aim to find sets of SAE latents that disentangle the dataset classes. Inspired by SHIFT, we train probes on the model activations and measure the effect of ablating sets of latents on the probe accuracy. Ablating a disentangled set of latents should only have an isolated causal effect on one class probe, while leaving other class probes unaffected.

We consider a dataset mapping text to exactly one of $m$ concepts $c \in C$. For each class with index $i = 1, ..., m$ we select the set $L_i$ of the most relevant SAE latents as described in section [ref]. Note that we select the top signed importance scores, as we are only interested in latents that actively contribute to the targeted class.

For each concept $c_i$, we partition the dataset into samples of the targeted concept and a random mix of all other labels.

We define the model with probe corresponding to concept $c_j$ with $j = 1, ..., m$ as a linear classifier $C_j$ which is able to classify concept $c_j$ with accuracy $A_j$. Further, $C_{i,j}$ denotes a classifier for $c_j$ where latents $L_i$ are ablated. Then, we iteratively evaluate the accuracy $A_{i,j}$ of all linear classifiers $C_{i,j}$ on the dataset partitioned for the corresponding class $c_j$. The targeted probe perturbation score:

$$
S_{TPP} = mean_{(i=j)} (A_{i,j} - A_j) - mean_{(i≠j)} (A_{i,j} - A_j)
$$

represents the effectiveness of causally isolating a single probe. Ablating a disentangled set of latents should only show a significant accuracy decrease if $i = j$, namely if the latents selected for class $i$ are ablated in the classifier of the same class $i$, and remain constant if $i \neq j$.

<iframe
  src="/sae-bench?modelId=gemma-2-2b&dSae=65536&layer=12&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=tpp%7C%7Ctpp_metrics%7C%7Ctpp_threshold_20_total_metric&groupBy=saeClass&embed=true"
  className="h-[400px] max-h-[400px] border-0 w-full mt-3 mb-5"
  loading="lazy"
></iframe>

</details>

#### Automated Interpretability

<details id="details-autointerp" class="anchor">
<summary>Show Details</summary>

In automated interpretability evaluation, we use gpt4o-mini as an LLM judge to quantify the interpretability of SAE features at scale in line with [Bills et al.](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html). Our implementation is similar to the [detection score proposed by EleutherAI](https://arxiv.org/abs/2410.13928), the [observability tool by Transluce](https://transluce.org/observability-interface) is noteworthy as well. The evaluation consists of two phases: generation and scoring.

In the generation phase, we obtain SAE activation values on webtext sequences. We select sequences with the highest activation values (top-k) and sample additional sequences with probability proportional to their activation values. These sequences are formatted by highlighting activating tokens with \`<<token>>\` syntax, and prompt an LLM to generate explanations for each feature based on these formatted sequences.

The scoring phase begins by creating a test set containing top activation sequences, importance-weighted sequences, and random sequences from the remaining distribution. Given a feature explanation and the shuffled test set of unlabeled sequences, another LLM judge predicts which sequences would activate the feature. The automatic interpretability score reflects the accuracy of predicted activations.

<iframe
  src="/sae-bench?modelId=gemma-2-2b&dSae=65536&layer=12&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=autointerp%7C%7Cautointerp%7C%7Cautointerp_score&groupBy=saeClass&embed=true"
  className="h-[400px] max-h-[400px] border-0 w-full mt-3 mb-5"
  loading="lazy"
></iframe>

</details>

#### RAVEL

<details id="details-ravel" class="anchor">
<summary>Show Details</summary>

If an SAE is working as intended, different concepts should have dedicated latents, i.e. they should be disentangled. RAVEL evaluates how well interpretability methods identify and separate facts in language models. For example, can we use the SAE to have the model think that Paris is in Japan but that the language is still French?

The dataset contains 5 entity types (cities, Nobel laureates, verbs, physical objects, and occupations), each with 400-800 instances and 4-6 distinct attributes. For example, cities have attributes like country, continent, and language. Attributes are tested using 30-90 prompt templates in natural language and JSON format, with both zero-shot and few-shot variants.

RAVEL evaluates feature disentanglement through a three-stage process:

1. Filtering to create a dataset of Entity-Attribute pairs that the model predicts with high accuracy
2. Identifying attribute-specific features using a probe trained on the model's latent representations
3. Computing a disentanglement score that averages cause and isolation metrics

For an entity $E$ (e.g., "Paris"), the cause score measures if intervening on feature $F_A$ successfully changes the model's prediction of attribute $A$ (e.g., country) from its original value $A_E$ ("France") to a target value $A_E'$ ("Japan"). The isolation score verifies that this intervention preserves predictions for all other attributes $B$ (e.g., language) at their original values $B_E$ ("French"). The final disentanglement score averages cause and isolation metrics across, with higher scores indicating better attribute separation. For instance, when changing Paris's country from France to Japan, a well-disentangled model would make this change while keeping the language attribute as French rather than incorrectly changing it to Japanese.

Follow-up work by [Chaudhary & Geiger](https://github.com/MaheepChaudhary/SAE-Ravel) adapted the RAVEL benchmark to sparse autoencoders. While our implementation is closely related to the original RAVEL evaluation, a differential binary masking-based latent selection in line with [Chaudhary & Geiger](https://github.com/MaheepChaudhary/SAE-Ravel) is forthcoming.

<iframe
  src="/sae-bench?modelId=gemma-2-2b&release=sae_bench_0125&dSae=65536&trainingTokens=499998720&layer=12&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=ravel%7C%7Cravel%7C%7Cdisentanglement_score&logX=true&groupBy=saeClass&embed=true"
  className="h-[400px] max-h-[400px] border-0 w-full mt-3 mb-5"
  loading="lazy"
></iframe>

</details>

#### Sparse Probing

<details id="details-sparse-probing" >
<summary>Show Details</summary>

We evaluate our sparse autoencoders' ability to learn specified features through a series of targeted probing tasks across diverse domains, including language identification, profession classification, and sentiment analysis. For each task, we encode inputs through the SAE, apply mean pooling over non-padding tokens, and select the top-K latents using maximum mean difference. We train a logistic regression probe on the resulting representations and evaluate classification performance on held-out test data. Our evaluation spans 35 distinct binary classification tasks derived from five datasets.

Our probing evaluation encompasses five datasets spanning different domains and tasks:

| Dataset          | Task Type                           | Description                                            |
| ---------------- | ----------------------------------- | ------------------------------------------------------ |
| \`bias_in_bios\` | Profession Classification           | Predicting professional roles from biographical text   |
| Amazon Reviews   | Product Classification & Sentiment  | Dual tasks: category prediction and sentiment analysis |
| Europarl         | Language Identification             | Detecting document language                            |
| GitHub           | Programming Language Classification | Identifying coding language from source code           |
| AG News          | Topic Categorization                | Classifying news articles by subject                   |

To ensure consistent computational requirements across tasks, we sample 4,000 training and 1,000 test examples per binary classification task and truncate all inputs to 128 tokens. For GitHub data, we follow Gurnee et al. by excluding the first 150 characters (approximately 50 tokens) as a crude attempt to avoid license headers. We evaluated both mean pooling and max pooling across non-padding tokens, and used mean pooling as it obtained slightly higher accuracy. From each dataset, we select subsets containing up to five classes. Multiple subsets may be drawn from the same dataset to maintain positive ratios ≥0.2.

<iframe
  src="/sae-bench?modelId=gemma-2-2b&dSae=65536&layer=12&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=sparse_probing%7C%7Csae%7C%7Csae_top_1_test_accuracy&groupBy=saeClass&embed=true"
  className="h-[400px] max-h-[400px] border-0 w-full mt-3 mb-5"
  loading="lazy"
></iframe>

</details>

## SAE Bench Use Cases

The Neuronpedia interface contains benchmark results for Gemma-Scope SAEs (JumpReLU architecture, 16k - 1M width, on Gemma-2-2B and -9B models) and the open-source SAEBench suite of sparse autoencoders (Standard (ReLU) and TopK architecture, 4k, 16k, 65k width for Gemma-2-2B).

In addition to SAEs, we provide the PCA of the residual stream (fit over a dataset of 200M model activations from the SAE training dataset) as a baseline. Note that the number of principal components is fixed to the model's hidden dimension (ie. hidden_dim = 2304 for Gemma-2-2B) while the number of SAE latents is often significantly higher.

Note that while we do compare Gemma Scope JumpReLU SAEs with our SAEs, the comparison is not apples to apples due to differences in training datasets, number of training tokens, training dataset context length, and other variables. We plan on training JumpReLU and Gated SAEs for direct comparisons of SAE architectures. In addition, we plan on investigating [improvements](https://transformer-circuits.pub/2024/april-update/index.html#training-saes) to our Standard ReLU training approach, as our Standard SAEs currently have many dead latents.

### Revealing Unintended Behavior

Our feature absorption evaluation demonstrates the importance of having a suite of benchmarks to evaluate new methods. Optimizing the sparsity fidelity tradeoff, the field adopted JumpReLU and TopK sparse autoencoders which significantly worsened feature absorption. This is especially present at low L0 SAEs that are commonly used as they contain more human interpretable features. (Note, that feature absorption increases with training time. While the number of training tokens is comparable for Standard and TopK SAEs, the Gemma-Scope JumpReLU SAEs have been trained significantly longer.)

<iframe
  src="/sae-bench?modelId=gemma-2-2b&layer=12&dSae=65536&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=absorption_first_letter%7C%7Cmean%7C%7Cmean_full_absorption_score&groupBy=saeClass&embed=true"
  className="h-[400px] max-h-[400px] border-0"
  loading="lazy"
  title="Absorption score across architectures for fixed 65k width. Note that unlike our other metrics, a lower score or rate of feature absorption is better."
></iframe>

_Absorption score across architectures for fixed 65k width. Note that unlike our other metrics, a lower score or rate of feature absorption is better._

### Measuring Progress in New Architectures

Despite feature absorption, the new SAE architectures TopK and JumpReLU significantly outperform the standard ReLU architecture when evaluated on Layer 12 out of 26 in Gemma-2-2B. We therefore recommend the use of new architectures over Standard SAEs in line with [Lindsey et al.](https://transformer-circuits.pub/2024/august-update/index.html#interp-evals)

<div className="grid sm:grid-cols-2 grid-cols-1 gap-x-2 gap-y-2 w-full">

<iframe
  src="/sae-bench?modelId=gemma-2-2b&layer=12&dSae=65536&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=scr%7C%7Cscr_metrics%7C%7Cscr_metric_threshold_20&groupBy=saeClass&embed=true"
  className="w-full h-[400px] max-h-[400px] border-0"
  loading="lazy"
  title=""
></iframe>

<iframe
  src="/sae-bench?modelId=gemma-2-2b&layer=12&dSae=65536&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=core%7C%7Cmodel_performance_preservation%7C%7Cce_loss_score&groupBy=saeClass&embed=true"
  className="w-full h-[400px] max-h-[400px] border-0"
  loading="lazy"
  title=""
></iframe>

<iframe
  src="/sae-bench?modelId=gemma-2-2b&layer=12&dSae=65536&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=autointerp%7C%7Cautointerp%7C%7Cautointerp_score&groupBy=saeClass&embed=true"
  className="w-full h-[400px] max-h-[400px] border-0"
  loading="lazy"
  title=""
></iframe>

<iframe
  src="/sae-bench?modelId=gemma-2-2b&layer=12&dSae=65536&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=unlearning%7C%7Cunlearning%7C%7Cunlearning_score&groupBy=saeClass&embed=true"
  className="w-full h-[400px] max-h-[400px] border-0"
  loading="lazy"
  title=""
></iframe>

</div>

_Diverse metric scores require tradeoffs in sparsity._

### Tuning Training Parameters

We find that metrics often vary across widths, sparsities, architectures, number of training tokens and model layers. Thus, we recommend to perform ablation studies when proposing new SAE architectures or evaluation metrics to ensure that the results are general and hold in a range of cases, rather than being an artifact of underlying variables such as sparsity. It is especially important to examine performance across multiple sparsities and model layers.

As examples, the impact of dictionary width on sparse probing varies significantly from layer 5 to layer 19 in Gemma-2-2B. In early layers, wider dictionaries are strictly worse, yet the results become more mixed in later layers. This may reflect how the model processes information, such as findings that later layers typically have more [abstract features](https://transformer-circuits.pub/2024/jan-update/index.html#dict-learning).

<div className="grid sm:grid-cols-2 grid-cols-1 gap-x-2 gap-y-2 w-full">

<iframe
  src="/sae-bench?modelId=gemma-2-2b&layer=5&release=gemmascope&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=sparse_probing%7C%7Csae%7C%7Csae_top_1_test_accuracy&groupBy=dSae&embed=true"
  className="w-full h-[400px] max-h-[400px] border-0"
  loading="lazy"
  title=""
></iframe>

<iframe
  src="/sae-bench?modelId=gemma-2-2b&layer=19&release=gemmascope&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=sparse_probing%7C%7Csae%7C%7Csae_top_1_test_accuracy&groupBy=dSae&embed=true"
  className="w-full h-[400px] max-h-[400px] border-0"
  loading="lazy"
  title=""
></iframe>

</div>

_The optimum sparsity for sparse probing significantly varies between layers._

#### SAE evaluations can inform training time

We further monitor performance on SAE Bench metrics throughout training and verify that metrics consistently increase on average. One initial motivation for the project was informing the duration of SAE training runs by signaling diminishing returns for performance on downstream applications. We observe that the number of training tokens before encountering a performance plateau significantly varies across metrics.

<img src="/saebench/autointerp-gemma-layer-5.png" alt="Tokens vs Autointerp Score Gemma Layer 5" className="sm:w-[75%]">

_The automated interpretability score monotonically increases throughout training on average. Notably, TopK is more token efficient than the Standard SAE architecture._

<img src="/saebench/scr-top10.png" alt="Tokens vs SCR Top 10 Metric Gemma Layer 5" className="sm:w-[75%] mt-3">

<img src="/saebench/unlearning.png" alt="Tokens vs Unlearning Score Gemma Layer 5" className="sm:w-[75%] mt-3">

### Selecting the Best Width and Sparsity for Specific Tasks

#### The Best SAE width varies across tasks

One general application of SAEs is the controlled modification of model behavior[^1]. For example, the debiasing technique SHIFT by Marks et al (CITE) modifies classification behavior through human selection of interpretable SAE latents. The number of features a human can manually select is limited, so we would prefer to make our modification using a smaller number of latents. In this case, smaller SAEs are clearly better for a fixed intervention budget in the lower L0 regime (left subplot). This is perhaps unsurprising, as with a fixed number of latents we are modifying a larger percentage of the smaller SAE.

The [Spurious Correlation Removal (SCR)](https://arxiv.org/abs/2411.18895) metric formulates SHIFT as an SAE evaluation. In the SCR setting, we automatically select SAE latents instead of relying on human judgement, effectively removing the intervention budget. Increasing the intervention budget from 10 up to 500 ablated latents, we do not see a clear advantage to using wider SAEs (right subplot).

<div className="grid sm:grid-cols-2 grid-cols-1 gap-x-2 gap-y-2 w-full">

<iframe
  src="/sae-bench?modelId=gemma-2-2b&layer=12&saeClass=topk&trainingTokens=299999232%2C199999488&release=sae_bench&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=scr%7C%7Cscr_metrics%7C%7Cscr_metric_threshold_10&groupBy=dSae&embed=true"
  className="w-full h-[400px] max-h-[400px] border-0"
  loading="lazy"
  title="SAE Bench Gemma-2-2B TopK Width Series Layer 12 L0 vs SCR Top 10 Metric"
></iframe>

<img src="/saebench/best-of.png" alt="SAE Bench Gemma-2-2b TopK Width Series Layer 12 L0 vs SCR Best of [10,50,100,500]" className="">

</div>

_The number of features per concept varies with (known as [feature splitting](https://transformer-circuits.pub/2023/monosemantic-features)). Given a concept removal task, the intervention budget therefore determines the optimal sparsity. In line with [Sharkey](https://www.alignmentforum.org/posts/64MizJXzyvrYpeKqm/sparsify-a-mechanistic-interpretability-research-agenda), we observe that the choice of SAE width is a tradeoff between short description length and reconstruction error._

We also find that feature absorption gets worse with increased width.

<iframe
  src="/sae-bench?modelId=gemma-2-2b&layer=12&release=gemmascope&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=absorption_first_letter%7C%7Cmean%7C%7Cmean_full_absorption_score&groupBy=dSae&embed=true"
  className="h-[400px] max-h-[400px] border-0"
  loading="lazy"
  title="Absorption score across architectures for fixed 65k width. Note that unlike our other metrics, a lower score or rate of feature absorption is better."
></iframe>

For automated interpretability and reconstruction error however, we observe clear returns to increasing width.

<div className="grid sm:grid-cols-2 grid-cols-1 gap-x-2 gap-y-2 w-full">

<iframe
  src="/sae-bench?modelId=gemma-2-2b&layer=12&release=gemmascope&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=autointerp%7C%7Cautointerp%7C%7Cautointerp_score&groupBy=dSae&embed=true"
  className="w-full h-[400px] max-h-[400px] border-0"
  loading="lazy"
  title="Gemma-Scope Gemma-2-2B Width Series Layer 12 L0 vs Autointerp Score"
></iframe>

<iframe
  src="/sae-bench?modelId=gemma-2-2b&layer=12&release=gemmascope&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=core%7C%7Cmodel_performance_preservation%7C%7Cce_loss_score&groupBy=dSae&embed=true"
  className="w-full h-[400px] max-h-[400px] border-0"
  loading="lazy"
  title="Gemma-Scope Gemma-2-2B Width Series Layer 12 L0 vs Loss Recovered"
></iframe>

</div>

#### The Best SAE sparsity varies across tasks

The choice of optimal sparsity is highly dependent on the task as well. We clearly see the tradeoff between human interpretability and reconstruction loss.

<div className="grid sm:grid-cols-2 grid-cols-1 gap-x-2 gap-y-2 w-full">

<iframe
  src="/sae-bench?modelId=gemma-2-2b&layer=12&dSae=65536&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=autointerp%7C%7Cautointerp%7C%7Cautointerp_score&groupBy=saeClass&embed=true"
  className="w-full h-[400px] max-h-[400px] border-0"
  loading="lazy"
  title="Gemma-Scope Gemma-2-2B Width Series Layer 12 L0 vs Autointerp Score"
></iframe>

<iframe
  src="/sae-bench?modelId=gemma-2-2b&layer=12&dSae=65536&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=core%7C%7Cmodel_performance_preservation%7C%7Cce_loss_score&groupBy=saeClass&embed=true"
  className="w-full h-[400px] max-h-[400px] border-0"
  loading="lazy"
  title="Gemma-Scope Gemma-2-2B Width Series Layer 12 L0 vs Loss Recovered"
></iframe>

</div>

_SAEs in the high sparsity regime (low L0) perform relatively better on automated interpretability but worse on reconstruction error (Loss Recovered)._

Even for similar downstream tasks we find different sparsities: Confusingly, unlike the closely related SCR evaluation, TPP has a strong correlation with sparsity, and higher L0 usually performs significantly better.

<div className="grid sm:grid-cols-2 grid-cols-1 gap-x-2 gap-y-2 w-full">

<iframe
  src="/sae-bench?modelId=gemma-2-2b&layer=12&dSae=16384&trainingTokens=299999232%2C-1&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=tpp%7C%7Ctpp_metrics%7C%7Ctpp_threshold_20_total_metric&groupBy=saeClass&embed=true"
  className="w-full h-[400px] max-h-[400px] border-0"
  loading="lazy"
  title="Gemma-Scope Gemma-2-2B Width Series Layer 12 L0 vs TPP Top 20 Metric"
></iframe>

<iframe
  src="/sae-bench?modelId=gemma-2-2b&layer=12&dSae=16384&trainingTokens=299999232%2C-1&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=scr%7C%7Cscr_metrics%7C%7Cscr_metric_threshold_20&groupBy=saeClass&embed=true"
  className="w-full h-[400px] max-h-[400px] border-0"
  loading="lazy"
  title="Gemma-Scope Gemma-2-2B Width Series Layer 12 L0 vs SCR Top 20 Metric"
></iframe>

</div>

_The optimal sparsity differs for SCR and TPP – two metrics that are conceptually closely related._

For some tasks lower L0 is typically clearly better, For other tasks, higher L0 is typically better. For other tasks such as SCR, there isn't a clear relationship with L0, but the best performance is generally obtained within a window of 20-200.

Better for Lower L0:

- [Interpretability: SAE Bench Gemma-2-2B 65K Width Series Layer 12 - L0 vs Auto-Interp](/sae-bench?modelId=gemma-2-2b&layer=12&dSae=65536&trainingTokens=-1%2C299999232%2C199999488&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=autointerp%7C%7Cautointerp%7C%7Cautointerp_score&groupBy=saeClass)

Non-Linear Relationship with L0:

- [SCR: SAE Bench Gemma-2-2B 65K Width Series Layer 12 - L0 vs SCR Top 20 Metric](/sae-bench?modelId=gemma-2-2b&layer=12&dSae=65536&release=gemmascope%2Csae_bench&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=scr%7C%7Cscr_metrics%7C%7Cscr_metric_threshold_20&groupBy=saeClass)

Better for Higher L0:

- [Feature Absorption: SAE Bench Gemma-2-2B 65K Width Series Layer 12 - L0 vs Mean Absorption Score](/sae-bench?modelId=gemma-2-2b&layer=12&dSae=65536&release=gemmascope%2Csae_bench&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=absorption_first_letter%7C%7Cmean%7C%7Cmean_full_absorption_score&groupBy=saeClass)
- [TPP and Recon Accuracy: SAE Bench Gemma-2-2B 65K Width Series Layer 12 - L0 vs TPP Top 20 Metric](/sae-bench?modelId=gemma-2-2b&layer=12&dSae=65536&release=gemmascope%2Csae_bench&metricX=core%7C%7Csparsity%7C%7Cl0&metricY=tpp%7C%7Ctpp_metrics%7C%7Ctpp_threshold_20_total_metric&groupBy=saeClass)

**We would love to hear your feedback. Get in touch with us via [email](mailto:adam.karvonen@gmail.com,canrager@gmail.com).**

## Acknowledgements & Thanks

This work was conducted as part of the ML Alignment \& Theory Scholars (MATS) Program and supported by a grant from OpenPhilanthropy. The program's collaborative environment, bringing multiple researchers together in one location, made this diverse collaboration possible. We are grateful to Alex Makelov for discussions and implementations of sparse control, to McKenna Fitzgerald for her guidance and support throughout the program, as well as to Bart Bussmann, Patrick Leask, Javier Ferrando, Oscar Obeso, Stepan Shabalin, Arnab Sen Sharma and David Bau for their valuable input. Our thanks go to the entire MATS and Lighthaven staff.

[^1]: Though SAEs surprisingly often work well, the choice of the most performant interpretability technique depends on the use case.

# Citation

<pre style="white-space: pre-wrap; font-size: 11px; color: #0f172a; background-color: #f1f5f9; padding: 10px; border-radius: 5px;">
@misc{karvonen2025saebench,
      title={SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability}, 
      author={Adam Karvonen and Can Rager and Johnny Lin and Curt Tigges and Joseph Bloom and David Chanin and Yeu-Tong Lau and Eoin Farrell and Callum McDougall and Kola Ayonrinde and Matthew Wearden and Arthur Conmy and Samuel Marks and Neel Nanda},
      year={2025},
      eprint={2503.09532},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.09532}, 
}
</pre>
`;
