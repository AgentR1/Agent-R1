# Step-Level Trajectory Representation

## Representation Is Not the Same as MDP

`Step-level MDP` defines what the RL transition is. Trajectory representation answers a different question: how the interaction history is stored and replayed for optimization.

Those two layers are related, but they should not be conflated. A framework can talk about step-level decision making while still storing data in a way that weakens replay fidelity or hides step boundaries.

## Why Representation Matters

For long-horizon agent training, optimization depends on more than the abstract transition definition. The training stack also needs a faithful replay format:

- rollout-time behavior should match replay-time behavior
- token masks and log-probabilities should stay aligned
- step boundaries should remain explicit enough for value estimation and reward propagation

If representation stays too close to chat logs or a monolithic token stream, the framework loses part of the benefit of a step-level view.

![Evolution of trajectory representation toward step-level structure](../assets/step-level-trajectory-representation.png)

<div style="text-align: center; color: #666;" markdown>
The evolution of trajectory representation from message-based traces to token-space-consistent records and finally to step-based sequences.
</div>

## Three Representation Styles

### Text-Space Messages

The simplest format is a sequence of chat messages. This is readable and easy to interoperate with, but it can introduce retokenization drift. If rollout tokens are decoded into text and tokenized again during training, the replayed token sequence may differ from the original rollout.

That mismatch is especially harmful when masks, log-probabilities, or reward annotations depend on exact token boundaries.

### Flat Token-Space Storage

Another option is to store prompts and responses directly as token IDs. This avoids retokenization drift and preserves rollout-time tokenization during replay.

The tradeoff is that the whole multi-turn trajectory becomes one flat append-only stream. That makes context reconstruction, truncation, and step-aware manipulation much more awkward.

### Structured Step-Level Representation

Agent-R1 is best understood through a structured step-level view of trajectories:

- each step stores the observation shown to the policy
- each step stores the full generated action
- each step stores reward and related step metadata

This preserves token-level information inside the action while keeping the interaction boundary explicit. In practice, that is the representation that best matches step-level training.

## Why This Fits Agent-R1

The key intuition is simple: if the environment interacts with the policy step by step, the stored trajectory should make those steps visible too.

That is why trajectory representation in Agent-R1 is not just a serialization detail. It is part of keeping the framework aligned around the interaction step as the basic unit of agent training.
