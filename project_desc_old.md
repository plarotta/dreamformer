# A CLS-inspired dual-memory transformer with dream consolidation

**The most promising path toward sequence models with human-like memory is a transformer architecture that pairs a high-fidelity episodic buffer with a compressed semantic store, bridges them through a prioritized replay buffer, and runs periodic "dream" consolidation phases to transfer knowledge between stores.** This design draws on five converging research threads—differentiable neural computers, prioritized experience replay, memory-augmented transformers, complementary learning systems theory, and sleep-inspired consolidation—that have matured independently but never been fully integrated. The architecture described here, which we call **DreamFormer**, provides a concrete blueprint for that integration, complete with mathematical formulations for every component.

---

## The neuroscience blueprint: why two memories beat one

Complementary Learning Systems theory (McClelland, McNaughton & O'Reilly, 1995) established that a single learning system cannot simultaneously support rapid encoding of specific experiences and gradual extraction of statistical structure. The hippocampus uses **sparse, pattern-separated representations** (~1% neuronal activation) to store individual episodes in one shot without overwriting prior memories. The neocortex uses **dense, distributed representations** (~10–20% activation) and learns slowly across many exposures to discover latent statistical regularities. Attempting fast learning in a distributed system causes catastrophic interference—new knowledge overwrites old.

The updated CLS theory (Kumaran, Hassabis & McClelland, 2016) added three crucial refinements. First, the neocortex's learning rate is **schema-dependent**: when new information is consistent with existing knowledge structures, consolidation can be rapid. Second, the hippocampus can support limited generalization through recurrent similarity computation (the REMERGE model). Third, replay during sleep is **selective**, not uniform—experiences are prioritized by reward magnitude, prediction error, novelty, and emotional salience. Sharp-wave ripples in hippocampal CA1 replay experienced sequences at **~20× compression** during NREM sleep, with dopaminergic and noradrenergic modulation gating which memories persist.

These biological principles translate directly into engineering requirements: a fast-write episodic store, a slow-write semantic store, selective replay weighted by informativeness, and offline consolidation phases that alternate between focused replay (NREM analog) and generative exploration (REM analog).

---

## DreamFormer architecture overview

The architecture comprises four interacting subsystems layered on top of a standard transformer backbone. During normal "wake" processing, the transformer processes input sequences while reading from and writing to two external memory stores. A replay buffer captures key–value snapshots of processing. Periodically, a "dream" phase samples from the replay buffer and consolidates information between the two stores. All operations are differentiable.

### The four subsystems at a glance

The **short-term memory (STM)** is a small, high-resolution matrix **M_stm ∈ ℝ^{N_s × d}** with full DNC-style read/write heads supporting content-based, forward-temporal, and backward-temporal addressing. It stores recent episodic representations with high fidelity and fast turnover. Capacity is deliberately limited (e.g., N_s = 256 slots) to force consolidation.

The **long-term memory (LTM)** is an associative compressive matrix **M_ltm ∈ ℝ^{d_k × d_v}** per attention head, following Infini-attention's linear attention formulation. It provides unbounded, constant-size storage at the cost of lossy compression. Information enters LTM only through the consolidation mechanism, never through direct writing during wake processing.

The **replay buffer** is a fixed-capacity priority queue **B = {(k_i, v_i, p_i)}** of key–value pairs drawn from the STM, stored in a sum-tree for O(log N) prioritized sampling. Priorities are assigned by prediction-error magnitude, following Schaul et al.'s proportional PER formulation.

The **dream consolidation module** runs between processing episodes, sampling from the replay buffer and deciding what to compress into LTM, what to keep in STM, and what to discard—implementing a full NREM-then-REM cycle.

---

## Differentiable read/write mechanisms for both stores

### STM: full DNC-style addressing

The STM inherits the Differentiable Neural Computer's complete addressing machinery. A controller (the transformer's output at a designated layer) emits an interface vector ξ_t that parameterizes all memory operations.

**Content-based addressing** produces a weighting over STM slots by computing scaled cosine similarity between a query key k_t and each memory row:

$$w^c_t(i) = \text{softmax}\Big(\beta_t \cdot \frac{k_t \cdot M_{stm}(i)}{\|k_t\| \cdot \|M_{stm}(i)\|}\Big)$$

where β_t > 0 is a learned sharpness parameter. This allows associative lookup—finding the slot whose content best matches a query.

**Write allocation** uses the DNC's usage-tracking mechanism. A usage vector u_t ∈ [0,1]^{N_s} tracks how much each slot is in use:

$$u_t = (u_{t-1} + w^w_{t-1} - u_{t-1} \odot w^w_{t-1}) \odot \psi_t$$

where ψ_t = ∏_i (1 − f^i_t · w^{r,i}_{t-1}) is a retention vector controlled by free gates f^i_t. Slots are sorted by ascending usage to produce an allocation weighting a_t that assigns highest weight to the least-used slot. The write weighting interpolates between content-based and allocation-based writing:

$$w^w_t = g^w_t \big[g^a_t \cdot a_t + (1 - g^a_t) \cdot c^w_t\big]$$

**Memory update** follows the erase-then-add protocol:

$$M_{stm,t} = M_{stm,t-1} \odot (1 - w^w_t \cdot e_t^\top) + w^w_t \cdot v_t^\top$$

**Temporal linking** tracks write order through a link matrix L_t, enabling forward and backward traversal of the write sequence—critical for maintaining sequential structure in episodic memories. Each read head blends three modes (backward, content, forward) through a learned softmax over mode weights π_t.

The STM also incorporates the improvements from Csordás & Schmidhuber (2019): **memory masking** for key–value separation, **improved deallocation** that erases slot contents when usage drops, and **link distribution sharpening** to prevent blurring over long sequences.

### LTM: compressive associative memory via linear attention

The LTM follows Infini-attention's formulation, representing long-term knowledge as a fixed-size associative matrix. Unlike the STM, the LTM has no discrete slots—information is compressed into a continuous associative binding.

**Memory retrieval** uses the query projections Q from the transformer:

$$A_{ltm} = \frac{\sigma(Q) \cdot M_{ltm}}{\sigma(Q) \cdot z_{ltm}}$$

where σ = ELU + 1 is a non-negative activation, and z_ltm ∈ ℝ^{d_k} is a normalization vector tracking the sum of all stored keys.

**Memory update** uses the delta rule variant to avoid redundant storage:

$$M_{ltm} \leftarrow M_{ltm} + \sigma(K)^\top \Big(V - \frac{\sigma(K) \cdot M_{ltm}}{\sigma(K) \cdot z_{ltm}}\Big)$$

This subtracts existing content before adding, so re-encoding an already-stored pattern leaves the matrix unchanged. The normalization vector updates as z_ltm ← z_ltm + Σ_t σ(K_t). Critically, **LTM updates only occur during dream consolidation**, not during wake processing. This enforces the CLS principle that the slow system should not be modified by individual experiences.

### Cross-memory attention gating

During inference, the transformer blends STM and LTM retrievals through a learned gate:

$$A_{out} = \sigma(\beta_{gate}) \odot A_{ltm} + (1 - \sigma(\beta_{gate})) \odot A_{stm}$$

where β_gate is a per-head scalar. After training, heads specialize: some become "episodic" (gate ≈ 0, favoring STM), some "semantic" (gate ≈ 1, favoring LTM), and some "mixer" heads that blend both sources. This mirrors biological findings that hippocampal and neocortical systems jointly contribute to memory retrieval with task-dependent weighting.

---

## The replay buffer: prioritized staging area

The replay buffer B stores **(key, value, metadata)** tuples captured from STM during wake processing. It operates as a staging area between short-term processing and long-term consolidation—the computational analog of hippocampal traces awaiting sleep replay.

### Priority assignment by prediction error

Each entry receives a priority based on the model's prediction error when processing the corresponding sequence segment:

$$p_i = |\mathcal{L}(x_i; \theta)| + \varepsilon$$

where L is the cross-entropy loss on the segment and ε is a small constant (e.g., 0.01) preventing zero priorities. This generalizes Schaul et al.'s TD-error formulation: **prediction error is the domain-general version of TD-error**, measuring how much a given experience deviates from the model's expectations.

### Sampling probability and bias correction

Sampling probability follows the PER formulation:

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

with α controlling prioritization strength (α = 0.6 is a good default). Importance-sampling weights correct the resulting distributional shift:

$$w_i = \Big(\frac{1}{|B|} \cdot \frac{1}{P(i)}\Big)^\beta \bigg/ \max_j w_j$$

where β is annealed linearly from 0.4 to 1.0 over training. Early in training, prioritization is aggressive (β low, accepting bias for faster learning); at convergence, full correction ensures unbiased gradients.

### Sum-tree implementation

Priorities are stored in a **binary sum-tree** with N leaf nodes and N−1 internal nodes. Each parent stores the sum of its children. Sampling is O(log N): divide [0, p_total] into k stratified segments, sample uniformly within each, then traverse top-down. Priority updates are also O(log N) by propagating differences to the root. This makes the buffer practical at scales of **10^6+ entries**.

### Beyond prediction error: multi-criterion priorities

Following neuroscience evidence that biological replay weights multiple factors, priorities can incorporate additional signals:

- **Novelty bonus**: Information-theoretic surprise of the input given the model's prior (KL divergence between predicted and observed token distributions)
- **Recency weighting**: Exponential decay favoring recent entries, counteracting staleness
- **Reducible loss** (following the ReLo framework): Distinguishing learnable prediction errors from irreducible noise, preventing the buffer from oversampling fundamentally unpredictable sequences

The composite priority becomes p_i = (|L_i| + λ_nov · nov_i + λ_rec · rec_i) · I(reducible_i), where the indicator function filters out irreducible noise.

---

## Dream consolidation: the NREM-REM cycle

The dream module is the architectural centerpiece, implementing offline consolidation between wake episodes. It draws on the Wake-Sleep Consolidated Learning (WSCL, 2024) framework, the Perturbed and Adversarial Dreaming (PAD) model, and phasic consolidation findings showing that **dedicated consolidation phases reduce total replay needed by ~55%** compared to continuous replay.

### Phase 1: NREM-like focused replay

The NREM phase replays stored experiences from the buffer and consolidates important information into LTM. It runs for T_nrem steps, each consisting of:

1. **Sample a minibatch** of B entries from the replay buffer using prioritized sampling
2. **Evaluate consolidation scores** for each entry using a learned gating network:

$$g^{cons}_i = \sigma\Big(W_{cons} \cdot [k_i \| v_i \| \text{access\_count}_i \| \text{age}_i]\Big)$$

Entries with g^{cons} above a threshold τ_cons are selected for LTM consolidation.

3. **Update LTM** with selected entries using the delta-rule update:

$$M_{ltm} \leftarrow M_{ltm} + \sigma(K_{sel})^\top \Big(V_{sel} - \frac{\sigma(K_{sel}) \cdot M_{ltm}}{\sigma(K_{sel}) \cdot z_{ltm}}\Big)$$

4. **Apply EWC-style regularization** to prevent LTM updates from destroying previously consolidated knowledge:

$$\mathcal{L}_{NREM} = \mathcal{L}_{replay}(\theta) + \frac{\lambda_{ewc}}{2} \sum_i F_i (\theta_i - \theta^*_i)^2$$

where F_i is the diagonal Fisher Information for parameter i, computed over recent consolidated entries.

5. **Update STM** by freeing slots whose content has been consolidated (setting free gates to 1 for consolidated entries), making room for new episodic memories.

6. **Update replay priorities**: After processing each sampled entry, recalculate its priority with the updated model. Entries whose priority drops below a threshold for multiple consecutive evaluations are marked as "consolidated" and eligible for eviction.

### Phase 2: REM-like generative exploration

The REM phase generates novel synthetic sequences to improve generalization and extract abstract patterns. It runs for T_rem steps:

1. **Generate synthetic key–value pairs** by sampling from the model's own latent space. Following the PAD framework, this involves random linear combinations of entries from the replay buffer:

$$k_{dream} = \sum_j \alpha_j \cdot k_j, \quad v_{dream} = \sum_j \alpha_j \cdot v_j$$

where α is sampled from a Dirichlet distribution, creating novel recombinations of stored experiences.

2. **Process dream sequences** through the transformer, computing attention over both STM and LTM, and backpropagate gradients through the transformer backbone (but **not** through LTM—only the transformer parameters are updated during REM).

3. **Apply synaptic homeostasis**: Scale down all transformer weights by a small factor (e.g., 0.999), following the synaptic homeostasis hypothesis (Tononi & Cirelli, 2014). This prunes weak connections while preserving strong ones, improving the signal-to-noise ratio of learned representations.

4. **Update cross-memory gate parameters** to rebalance STM vs. LTM reliance based on what was consolidated in the NREM phase.

### Alternating NREM-REM cycles

Following Schapiro et al.'s (2022) finding that alternating NREM and REM stages facilitates graceful continual learning, the dream phase runs **multiple NREM-REM cycles** (typically 3–5) per consolidation event. NREM focuses hippocampal-to-neocortical transfer (STM→LTM consolidation); REM allows the transformer backbone to freely explore and integrate new with existing representations. The ratio is roughly **3:1 NREM-to-REM**, mirroring biological sleep architecture.

---

## How information flows through the system

The complete data lifecycle has five stages, creating a multi-timescale memory hierarchy:

**Stage 1 — Perception (transformer forward pass):** Input tokens are processed through the transformer. At a designated layer (typically in the upper third of the network), the model reads from both STM and LTM via the cross-memory attention gate.

**Stage 2 — Episodic encoding (STM write):** The transformer's interface vector parameterizes DNC-style writes to STM. New experiences are written to the least-used slots. Content-based addressing allows the model to update existing slots if the new information is similar enough (avoiding redundant storage).

**Stage 3 — Buffer capture:** At the end of each processed segment, the model's key–value activations at the memory-augmented layer are stored in the replay buffer with priority equal to the segment's prediction error. If the buffer is full, the lowest-priority entry is evicted (or entries are managed via reservoir sampling for distributional balance).

**Stage 4 — NREM consolidation:** Prioritized samples from the replay buffer are evaluated by the consolidation gate. High-scoring entries are compressed into LTM via linear-attention updates. Corresponding STM slots are freed. The transformer backbone receives EWC-regularized gradient updates from replayed sequences.

**Stage 5 — REM exploration:** The model generates dream sequences by recombining buffer entries, processes them through the full architecture, and updates transformer parameters (but not LTM). Synaptic downscaling prunes weak connections. Gate parameters adjust.

This mirrors the biological flow: sensory input → hippocampal encoding → hippocampal replay during NREM → neocortical consolidation → REM integration → updated cortical representations influence future encoding.

---

## Connections to existing architectures and key design decisions

Several existing systems validate individual components of this design. The **Compressive Transformer** (Rae et al., 2020) demonstrated that a three-level memory hierarchy (sequence → cached activations → compressed memories) outperforms flat architectures, with **1D convolution** as the best compression function and **attention-reconstruction loss** (preserving attention behavior rather than raw content) as the training signal. DreamFormer's LTM compression follows the same principle—Infini-attention's delta-rule update preserves the information needed for attention computation, not the raw representations.

The **MT-DNC** (Liang et al., 2025) showed that brain-inspired memory transformation between working memory and long-term memory improves reasoning on question-answering tasks, validating the dual-store approach. The **CLS-ER** system (Arani et al., 2022) demonstrated that maintaining short-term and long-term semantic memories with different update rates achieves state-of-the-art continual learning by converging to flatter loss-landscape minima.

Key design decisions in DreamFormer are guided by these precedents:

- **STM uses discrete slots with DNC addressing** rather than Infini-attention's continuous compression, because episodic memory requires high-fidelity, individually retrievable entries. The DNC's temporal link matrix preserves sequential structure that would be lost in compression.
- **LTM uses linear attention** rather than a kNN store (Memorizing Transformers), because semantic memory should be compressed and abstractive. The **114× compression ratio** of Infini-attention over kNN storage makes this practical at scale.
- **Consolidation is phasic, not continuous**, following the finding that dedicated consolidation phases reduce replay samples needed by ~55%. This also avoids the instability of simultaneously updating all memory systems.
- **Replay priorities use prediction error** as the primary signal, following both PER's empirical success and neuroscience evidence that hippocampal replay preferentially reactivates high-prediction-error experiences.

---

## Mathematical summary of the complete system

The full DreamFormer training objective combines wake and dream losses:

$$\mathcal{L}_{total} = \underbrace{\mathcal{L}_{wake}(\theta)}_{\text{standard LM loss}} + \lambda_1 \underbrace{\sum_i w_i \cdot \mathcal{L}_{replay}(x_i; \theta)}_{\text{IS-corrected NREM replay}} + \lambda_2 \underbrace{\mathcal{L}_{dream}(\theta)}_{\text{REM generative loss}} + \lambda_3 \underbrace{\sum_i F_i(\theta_i - \theta^*_i)^2}_{\text{EWC regularization}}$$

where w_i are importance-sampling weights from PER, x_i are prioritized replay samples, and F_i is the Fisher Information diagonal. The LTM update is separate from the gradient-based loss—it operates through the delta-rule mechanism on selected key–value pairs during NREM.

The **consolidation gate** is trained end-to-end by including a reconstruction reward: entries that, when removed from STM and read back from LTM, produce similar attention outputs receive positive learning signal:

$$\mathcal{L}_{gate} = \| \text{attn}(Q, K_{stm}, V_{stm}) - \text{attn}(Q, K_{ltm}, V_{ltm}) \|^2$$

This mirrors the Compressive Transformer's attention-reconstruction loss, ensuring that consolidation preserves *functionally relevant* information rather than raw representations.

---

## Conclusion: toward biologically grounded sequence memory

DreamFormer unifies five research threads into a single coherent architecture. The **DNC's differentiable read/write mechanisms** provide the STM with precise, controllable memory access including temporal traversal. **Infini-attention's compressive associative memory** provides an LTM with constant-size, unbounded-history storage. **Prioritized experience replay** converts the staging buffer from a passive FIFO into an active curation system that surfaces the most informative experiences. **CLS theory** provides the foundational principle that fast and slow systems must coexist and interact through selective replay. And **sleep consolidation research** provides the blueprint for the dream phase, where NREM-like focused replay transfers episodic knowledge to semantic storage while REM-like generative exploration strengthens abstract representations.

Three insights from this synthesis stand out. First, the **precision-capacity tradeoff** between kNN memory and linear-attention compression maps directly onto the hippocampal-neocortical distinction—this is not merely an analogy but a convergent design principle. Second, **prediction error is the universal priority signal**: it is TD-error in RL, cross-entropy loss in language modeling, and surprise-modulated dopamine in biology. Third, **phasic consolidation is strictly more efficient than continuous replay**, a finding that emerged independently in both the neuroscience literature (sleep is a dedicated brain state optimized for consolidation) and recent ML work (55% sample reduction with dedicated consolidation phases).

The most important open challenge is **scaling the dream phase**. Generative replay has been validated primarily on image classification; extending it to high-dimensional autoregressive sequence modeling (language, code, time-series) requires generators capable of producing realistic, diverse synthetic sequences. Diffusion-based approaches (DDGR, t-DGR) show promise but remain computationally expensive. The latent-space replay approach—generating dream sequences in the model's internal representation space rather than input space—offers the most practical path forward, following brain-inspired replay (van de Ven et al., 2020) which showed that replaying internal representations is both more efficient and more effective than replaying raw inputs.