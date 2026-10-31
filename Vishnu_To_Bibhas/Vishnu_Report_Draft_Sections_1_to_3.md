# Vishnu's Phase II Report Draft (Sections 1, 2, and 3)
**Author:** Dhanishetti Vishnu (Roll No. 23118032)  
**Collaborator:** Bibhas Nath (Roll No. 23118019)  
**Project:** IoT Botnet Detection via Feature Selection and Hyperparameter-Optimized XGB (Phase II Research Extension)

---

## Section 1: Introduction and Research Motivation for Methodology Extension

The rapid adoption of Internet of Things (IoT) technologies in critical infrastructure, healthcare, industrial automation, and smart cities has drastically widened the attack surface. Resource-constrained devices often lack robust hardware-based security controls, leaving them vulnerable to coordinated botnet malware such as Mirai and BASHLITE.

In the baseline paper, Antony & Roy (2026) demonstrated that flow-based classification paired with Stepwise Recursive Feature Elimination (RFE) and eXtreme Gradient Boosting (XGBoost) could achieve high detection accuracy on the IoT Network Intrusion Detection Dataset (Mirai subset). However, a critical investigation of the baseline reveals prominent algorithmic limitations:

1. **Greedy Trapping in Stepwise RFE:** Standard RFE operates as a greedy backward-elimination filter, removing features one by one based solely on linear or localized tree-weight coefficients. Consequently, RFE cannot explore multi-feature combinatorial spaces and readily gets trapped in sub-optimal local minima.
2. **Computational Inefficiency of Wrapper RFE:** Applying iterative model retraining across all 23 features incurs heavy computational penalties, which scales poorly when handling high-velocity IoT traffic.
3. **Absence of Bio-Inspired Global Search:** Modern IoT security threats exhibit non-linear feature co-dependencies. Metaheuristic swarm optimizers (such as Particle Swarm Optimization) possess adaptive stochastic exploration capabilities that navigate non-convex landscapes far more effectively than deterministic elimination.

**Research Objective:** To develop a **Two-Stage Hybrid Feature Selection Framework** that couples fast ensemble filter pre-selection with Binary Particle Swarm Optimization (B-PSO) to identify a more compact, robust, and computationally lightweight feature subset.

---

## Section 2: Detailed Architectural Differences Between Baseline & Proposed Extensions

| Architectural Dimension | Baseline Paper (Antony & Roy, 2026) | Proposed Phase II Extension (Group 6) | Technical Justification |
| :--- | :--- | :--- | :--- |
| **Stage 1: Pre-Filtering** | None (All 23 features fed directly into RFE) | **Extra-Trees & Mutual Information Pre-Filter** | Quickly prunes bottom 50% noisy/static features in linear time ($O(N \log N)$), dramatically shrinking the search space. |
| **Stage 2: Wrapper Search** | Deterministic Stepwise RFE | **Binary Particle Swarm Optimization (B-PSO)** | Provides stochastic global exploration with personal-best ($P_{\text{best}}$) and global-best ($G_{\text{best}}$) convergence. |
| **Objective Fitness** | Single-metric F1-score maximization | **Multi-Objective Cost Formulation** | Jointly minimizes classification error while penalizing excessive feature counts ($0.99 \cdot \text{Error} + 0.01 \cdot \frac{F_s}{F_a}$). |
| **Selected Feature Count** | 8 Features | **6 Features (73.9% Dimensionality Reduction)** | Delivers equivalent or superior discriminative power with 2 fewer features, slashing memory and inspection latency. |
| **Handover Interface** | Manual CSV slicing | **Standardized JSON Schema & Automated CSV Exports** | Guarantees zero data leakage and seamless downstream model hyperparameter optimization. |

---

## Section 3: Mathematical Formulation of Proposed Feature Selection

### 3.1 Stage 1: Filter-Based Feature Pre-selection
To eliminate uninformative features before swarm optimization, an Extra-Trees ensemble estimator computes Gini-based impurity reduction:
$$I(f_j) = \frac{1}{M} \sum_{m=1}^{M} \Delta i(s_{m,j})$$
Features are ranked in descending order, and the top $k = \lceil 0.50 \cdot D \rceil = 12$ features are retained.

### 3.2 Stage 2: Binary Particle Swarm Optimization (B-PSO) Dynamics
In B-PSO, a swarm of $N = 20$ particles navigates the 12-dimensional binary space over $T = 25$ iterations. Each particle $i$ possesses a continuous velocity vector $V_i \in \mathbb{R}^D$ and a binary position vector $X_i \in \{0, 1\}^D$:

1. **Velocity Update:**
   $$V_i^{t+1} = w \cdot V_i^t + c_1 r_1 \left(P_{\text{best}, i}^t - X_i^t\right) + c_2 r_2 \left(G_{\text{best}}^t - X_i^t\right)$$
   where $w$ is the time-varying inertia weight linearly decreasing from $w_{\text{max}} = 0.9$ to $w_{\text{min}} = 0.4$:
   $$w = w_{\text{max}} - t \cdot \left(\frac{w_{\text{max}} - w_{\text{min}}}{T_{\text{max}}}\right)$$
   and $c_1 = c_2 = 1.5$ are cognitive and social acceleration coefficients.

2. **Sigmoid Transfer Function & Binary Position Mapping:**
   $$S\left(V_{i,j}^{t+1}\right) = \frac{1}{1 + e^{-V_{i,j}^{t+1}}}$$
   $$X_{i,j}^{t+1} = \begin{cases} 1, & \text{if } \text{rand}() \le S\left(V_{i,j}^{t+1}\right) \\ 0, & \text{otherwise} \end{cases}$$

3. **Multi-Objective Fitness Function:**
   $$f(X_i) = \alpha \cdot \left(1 - \text{F1}_{\text{CV}}(X_i)\right) + \beta \cdot \left(\frac{F_{\text{selected}}}{F_{\text{total}}}\right) + \text{Penalty}$$
   with $\alpha = 0.99, \beta = 0.01$.

### 3.3 Domain Security Semantics of Selected Features
The resulting 6 optimal features selected by B-PSO provide comprehensive protocol defense coverage:
- **`udp.srcport`**: Pinpoints ephemeral port distributions in UDP reflection attacks.
- **`ip.flags`**: Catches abnormal fragmentation bitmasks in malicious packet crafting.
- **`tcp.dstport`**: Identifies scans against ports 23, 2323 (Telnet), and 80 (HTTP).
- **`ip.ttl`**: Detects anomalous static hop counts produced by Mirai attack engines.
- **`ip.proto`**: Distinguishes protocol switching across TCP, UDP, and ICMP floods.
- **`tcp`**: Flags stateful TCP stream transitions.
