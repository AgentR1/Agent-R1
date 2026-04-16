# Step-level MDP

This page focuses on the modeling layer of the problem. The central claim is that long-horizon agent optimization should not be formulated only as a token-level decision process. Instead, the natural transition unit for agent training is the interaction step.

## Token-Level Formulation

Token-level Markov decision process formulation is a natural extension of autoregressive language modeling. Given prompt \(x\) and response \(y = (y_1, \dots, y_L)\), the policy factorizes as

\[
\pi_\theta(y \mid x) = \prod_{i=1}^{L}\pi_\theta(y_i \mid x, y_{<i}).
\]

This factorization induces a token-level decision process almost for free. At token position \(i\), the state and action can be written as

\[
s_i^{\mathrm{tok}} = (x, y_{<i}), \qquad a_i^{\mathrm{tok}} = y_i,
\]

with transition

\[
s_{i+1}^{\mathrm{tok}} = (x, y_{\le i}).
\]

Such a formulation aligns cleanly with policy-gradient training and remains highly effective for single-turn alignment or reasoning tasks in which the environment is static or only weakly coupled to intermediate generations.

## Step-Level Reformulation

The difficulty is that multi-turn agents do not interact with the world only through token append operations. They call tools, receive observations, update working memory, revise context, branch on execution outcomes, and sometimes perform explicit context management between rounds. In such settings, the semantically meaningful transition is no longer "emit one more token," but rather "complete one interaction step and receive new environment feedback." When optimization remains purely token-centric, high-level decisions are fragmented across many low-level actions, and environment-mediated transitions are obscured by a long flat token trace.

This motivates step-level MDP formulation. Let \(s_t\) denote the observation available at interaction step \(t\), let \(a_t\) denote the complete interaction action chosen at that step, and let the environment return reward \(r_t\) together with the next observation \(s_{t+1}\). A trajectory is then written as

\[
\tau = \{(s_t, a_t, r_t, s_{t+1})\}_{t=0}^{T-1},
\]

with objective

\[
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t r_t\right].
\]

Here \(a_t\) is not a single token. It may internally contain a token sequence, a structured tool call, or a mixed response that combines reasoning and external actions. What changes is the unit at which the MDP is defined: the RL transition is now one interaction step rather than one appended token.

![Comparison between token-level and step-level MDP formulations](../assets/step-level-mdp.png)

<div style="text-align: center; color: #666;" markdown>
Comparison between token-level MDP formulation and step-level MDP formulation. The key shift is that the atomic action changes from a single token to a complete agent-environment interaction step.
</div>

The move to step-level MDP does not imply that token information is discarded. On the contrary, token-space consistency remains important for stable optimization inside each action. What changes is the unit at which the decision process is defined. Step-level formulation better reflects the causal structure of agent behavior: a step begins with an observation, produces an action, triggers an external transition, and only then exposes the next observation.

Step-level MDP also clarifies which parts of the loop belong to the policy and which belong to the environment. The policy is responsible for choosing \(a_t\) conditioned on \(s_t\). The environment is responsible for turning that interaction action into \(s_{t+1}\), possibly through tool execution, response parsing, external feedback, or context rewriting. This separation is difficult to maintain in a pure token-append abstraction, but it becomes explicit once the interaction round is treated as the transition unit.

## What This Leads To

However, establishing step-level MDP formulation mathematically is only half the battle. To actually optimize a policy over these \((s_t, a_t, r_t, s_{t+1})\) transitions, the training pipeline must also record and replay the interaction history in a way that honors the same step boundaries. If the empirical trajectory representation misaligns with this theoretical MDP, optimization remains fragile. That broader logic is summarized in [`Step-Level Training Logic`](../background/step-level-training-logic.md) and developed in the next two concept pages on trajectory representation and credit assignment.
