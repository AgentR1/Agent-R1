# Step-Level Trajectory Representation

Trajectory representation answers a different question from MDP formulation. It does not define the RL transition directly. Instead, it defines how an interaction history is stored and replayed for optimization. The two layers are related, but they should not be conflated. A framework may adopt step-level decision making while still storing trajectories in a way that weakens replay fidelity or obscures the boundaries required for optimization.

![Evolution of trajectory representation toward step-level structure](../assets/step-level-trajectory-representation.png)

<div style="text-align: center; color: #666;" markdown>
The evolution of trajectory representation from message-based traces to token-space-consistent records and finally to step-based sequences. This figure is intended as background and concept setup rather than the main technical claim.
</div>

## Text-Space Representation

One common representation for multi-turn agents is a sequence of chat-style messages. This format is simple and interoperable with standard chat interfaces, but it hides a serious inconsistency. Rollout takes place in token space, whereas replay may reconstruct text and tokenize it again during optimization. Because the mapping from token sequence to text and back is not reversible in general, the replayed sequence may differ from the one that originally produced the trajectory.

Once this retokenization drift occurs, masks, log-probabilities, and reward annotations can no longer be aligned reliably with the original rollout. This is why message-space convenience is not enough for stable step-level optimization.

## Flat Token-Space Representation

A stronger alternative is flat token-space storage, where prompts and responses are preserved directly as token IDs. This restores rollout-training consistency, but it still treats the whole interaction as one monolithic append-only sequence. That structure is workable for some training pipelines, yet it remains too rigid for long-horizon agents whose interaction history may need to be reconstructed, truncated, or reorganized at step boundaries.

## Structured Step-Level Representation

The representation that best matches Agent-R1's perspective is a structured step-level trajectory. Each interaction round is stored as a distinct unit containing the observation shown to the policy, the action produced at that step, and the reward or metadata attached to that interaction. This preserves token-level information inside each action while keeping the step itself explicit as the unit of replay and analysis.

The distinction matters because MDP formulation defines what the RL transition is, while representation defines how the interaction history is stored and replayed for optimization. If the MDP is step-level while the replay format obscures or corrupts step boundaries, optimization remains misaligned with the underlying decision process.

## How This Appears in Code

In Agent-R1, the trajectory is explicitly represented as a list of steps rather than a single monolithic sample:

```python
class AgentFlowStep(BaseModel):
    prompt_ids: list[int]
    response_ids: list[int]
    reward_score: Optional[float] = None
    extra_fields: dict[str, Any] = {}


class AgentFlowOutput(BaseModel):
    steps: list[_InternalAgentFlowStep]
    metrics: AgentFlowMetrics
```

This is the core implementation idea behind step-level trajectory representation: each rollout is organized as a sequence of step records, and each step keeps its own prompt ids, response ids, and reward signal.

The relevant implementation lives in:

- `agent_r1/agent_flow/agent_flow.py`
- `agent_r1/agent_flow/agent_env_loop.py`
