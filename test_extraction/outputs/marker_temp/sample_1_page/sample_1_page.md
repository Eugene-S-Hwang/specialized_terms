## **UEval: A Benchmark for Unified Multimodal Generation**

Bo Li Yida Yin Wenhao Chai Xingyu Fu\* Zhuang Liu\*

Princeton University

Website: [https://zlab-princeton](https://zlab-princeton.github.io/UEval)*.*github*.*io/UEval

[Code](https://github.com/zlab-princeton/UEval_benchmark) [Dataset](https://huggingface.co/datasets/zlab-princeton/UEval) [Leaderboard](https://zlab-princeton.github.io/UEval/leaderboard/Full-Leaderboard/1.0)

## **Abstract**

We introduce UEval, a benchmark to evaluate *unified models*, *i.e*., models capable of generating both images and text. UEval comprises 1,000 expert-curated questions that require both images and text in the model output, sourced from 8 real-world tasks. Our curated questions cover a wide range of reasoning types, from step-by-step guides to textbook explanations. Evaluating open-ended multimodal generation is non-trivial, as simple LLM-as-a-judge methods can miss the subtleties. Different from previous works that rely on multimodal Large Language Models (MLLMs) to rate image quality or text accuracy, we design a rubric-based scoring system in UEval. For each question, reference images and text answers are provided to a MLLM to generate an initial rubric, consisting of multiple evaluation criteria, and human experts then refine and validate these rubrics. In total, UEval contains 10,417 validated rubric criteria, enabling scalable and fine-grained automatic scoring. UEval is challenging for current unified models: GPT-5-Thinking scores only 66.4 out of 100, while the best open-source model reaches merely 49.1. We observe that reasoning models often outperform non-reasoning ones, and transferring reasoning traces from a reasoning model to a non-reasoning model significantly narrows the gap. This suggests that reasoning may be important for tasks requiring complex multimodal understanding and generation.

## **1 Introduction**

Unified multimodal models (Tong et al., 2024b; Zhou et al., 2025a; Deng et al., 2025) aim to integrate multimodal understanding and generation capabilities within a single system. Current evaluations of these models are largely confined to two paradigms: visual question answering (Marino et al., 2019; Liu et al., 2024b; Yue et al., 2024; Fu et al., 2025), which requires generating a textual answer from one or more input images, and text-to-image generation (Huang et al., 2023; Ghosh et al., 2023; Lin et al., 2024), which takes a textual description as input and asks the model to produce a corresponding image.

These paradigms overlook a central component of multimodal reasoning scenarios: unified multimodal generation that *produces both text and images* in response to a single query (Figure 1). In many real-world tasks, effective responses require images to illustrate specific concepts while simultaneously producing text to explain those visual elements. Without such evaluation, existing benchmarks fail to capture the rich interplay between language and vision that characterizes real-world multimodal reasoning.

While recent efforts (An et al., 2024; Liu et al., 2024a; Xia et al., 2025; Niu et al., 2025; Zhao et al., 2025) have proposed new benchmarks to evaluate unified models, there remains a lack of standardized approaches for evaluating unified multimodal generation. To address this gap, we introduce UEval, a challenging benchmark to assess unified models (Wang et al., 2024; Chen et al., 2025c; Yang et al., 2025; Google, 2025b; Xie et al., 2025) at scale. Unlike prior benchmarks, UEval requires models to reason and respond to complex user queries jointly in images and natural language, providing a rigorous testbed across diverse real-world scenarios.

\* Co-advising 1