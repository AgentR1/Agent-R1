# Step-Level Credit Assignment

Once the decision process is formulated at the step level and the trajectory is represented in a step-native form, reward propagation should also move to the same granularity. Otherwise, a mismatch remains between the unit at which decisions are modeled and the unit at which responsibility is assigned.

![Comparison of token-level, trajectory-level, and step-level credit assignment](../assets/step-level-credit-assignment.png)

<div style="text-align: center; color: #666;" markdown>
Comparison of token-level, trajectory-level, and step-level credit assignment. The main change is not how actions are tokenized, but where delayed rewards are attributed and propagated.
</div>

## Granularity Mismatch

Trajectory-level credit assignment is too coarse for this purpose. Assigning one scalar signal to the whole rollout may be simple and stable, but it cannot distinguish productive intermediate actions from harmful ones when an episode contains many interaction rounds.

Token-level credit assignment lies at the opposite extreme. It reuses the standard machinery of language-model RL, yet in agent settings it is often too fine. The strategically decisive event may be a retrieval call, a decomposition step, a context-management choice, or a tool invocation, while the reward arrives only later. If delayed return is attributed directly through surface tokens, the learning signal becomes diluted relative to the actual interaction choice.

## Step-Level Objective

The natural counterpart of a step-level MDP is therefore step-level credit assignment. In this view, value estimation, temporal-difference residuals, generalized advantage estimation, and PPO-style optimization are all organized around the interaction step. The policy may still factor internally over tokens, but the unit that receives advantage and responsibility is the complete interaction action rather than an isolated token.

This distinction matters especially under delayed reward. In many agent tasks, the final outcome depends on an earlier decision that changes the later trajectory: choosing the right tool, retrieving the right evidence, or preserving the right context for subsequent turns. Step-level credit assignment makes it possible to attribute success or failure to that earlier interaction decision without collapsing the signal into one trajectory-level scalar or dispersing it across many locally meaningless token choices.

## How This Appears in Code

Agent-R1's GAE implementation first aggregates token rewards into a step reward, then computes advantages over step indices:

```python
def compute_gae_advantage_return(
    token_level_rewards,
    values,
    response_mask,
    trajectory_uids,
    step_indices,
    gamma,
    lam,
):
    # Step-level reward: sum of token rewards inside the step.
    rewards = (token_level_rewards * response_mask).sum(dim=1)

    rewards_map[traj_inv, step_ids] = rewards
    values_map[traj_inv, step_ids] = values

    for t in reversed(range(max_step)):
        nextvalues = values_map[:, t + 1] if t < max_step - 1 else 0.0
        delta = rewards_map[:, t] + gamma * nextvalues - values_map[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
```

The important point is that the code does not propagate advantage over one flat token stream. It first builds step-level rewards and values, computes GAE over the step timeline, and only then broadcasts the result back to token positions when needed by PPO training.

The relevant implementation lives in:

- `agent_r1/core_algos.py`
- `agent_r1/ray_agent_trainer.py`
