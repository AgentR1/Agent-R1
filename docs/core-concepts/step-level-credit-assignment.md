# Step-Level Credit Assignment

## Credit Should Follow the Same Unit as the MDP

Once Agent-R1 adopts a `step-level MDP`, reward propagation should also happen at the step level.

Otherwise, the framework still has a granularity mismatch: the decision process is modeled as step-based interaction, but the optimizer assigns responsibility either too coarsely or too finely.

![Comparison of token-level, trajectory-level, and step-level credit assignment](../assets/step-level-credit-assignment.png)

<div style="text-align: center; color: #666;" markdown>
Comparison of token-level, trajectory-level, and step-level credit assignment. The main change is not how actions are tokenized, but where delayed rewards are attributed and propagated.
</div>

## The Mismatch in Other Granularities

### Trajectory-Level Credit

Trajectory-level credit assignment gives one scalar signal to the whole rollout. This is simple and sometimes stable, but it is too coarse for long agent trajectories where some intermediate decisions are clearly helpful and others are harmful.

### Token-Level Credit

Token-level credit assignment reuses the standard machinery of language-model RL, but it is often too fine for agent tasks. In many settings, the important decision is not one token by itself, but a whole interaction move:

- choosing to search
- calling a tool
- decomposing a problem
- restructuring context before the next step

If delayed reward is pushed directly through surface tokens, the learning signal becomes diluted relative to the actual decision unit.

## Why Step-Level Credit Fits Better

Step-level credit assignment is the middle ground that matches the interaction loop.

In a step-level PPO view:

- value is defined over the step state
- temporal-difference residuals are computed across environment-mediated steps
- generalized advantage estimation propagates reward across steps
- the PPO objective assigns credit to complete interaction steps, even if the policy internally factors over tokens

The important idea is that tokenization inside the model does not force token-level credit. The policy may still generate one token at a time, while optimization attributes responsibility to the full interaction step.

## Why This Matters for Agent Tasks

Agent rewards are often delayed. A successful final outcome may depend on an earlier decision such as retrieving the right evidence, making the right tool call, or preserving the right context for later turns.

Step-level credit assignment makes it easier to attribute that outcome to the earlier interaction choice that actually changed the trajectory.

This is why credit assignment in Agent-R1 should be understood as part of the same step-level logic as the MDP itself.
