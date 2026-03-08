from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Observation:
    """Environment observation representing the full next-round LLM input.

    Exactly one of the three fields should be set, each corresponding to a
    different abstraction level:

    - ``text``: the complete prompt as a raw string; the loop tokenises it.
    - ``messages``: a full chat-messages list; the loop applies a chat template.
    - ``token_ids``: an already-tokenised prompt; the loop uses it directly.
    """

    text: str | None = None
    """Full prompt as a raw text string."""

    messages: list[dict] | None = None
    """Full chat messages list (OpenAI format)."""

    token_ids: list[int] | None = None
    """Fully tokenised prompt ids."""


@dataclass
class Action:
    """Action taken by the LLM."""

    text: str | None = None
    """Decoded LLM response text."""

    token_ids: list[int] | None = None
    """Raw LLM response token ids."""


class AgentEnv(ABC):
    """Abstract base class for agent environments.

    An environment follows the standard RL interface (reset / step) and
    manages its own internal state across steps. It exposes:

    - ``reset``: reset the environment and return the initial observation.
    - ``step``: receive the LLM response, execute actions, update internal
      state, and return the next observation.

    Subclasses should register via ``@AgentEnv.register(name)`` so that
    ``AgentEnv.from_config(env_type=name, ...)`` can instantiate them.
    """

    _registry: dict[str, type["AgentEnv"]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an AgentEnv subclass under *name*."""

        def decorator(subclass: type["AgentEnv"]) -> type["AgentEnv"]:
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def from_config(cls, env_type: str, **kwargs) -> "AgentEnv":
        """Instantiate a registered env by *env_type* with remaining kwargs.

        Raises:
            ValueError: If *env_type* is not registered.
        """
        if env_type not in cls._registry:
            raise ValueError(f"Unknown env type: {env_type!r}. Available: {list(cls._registry.keys())}")
        return cls._registry[env_type](**kwargs)

    @abstractmethod
    def reset(self, **kwargs) -> Observation:
        """Reset the environment to its initial state.

        Args:
            **kwargs: Additional keyword arguments for environment initialisation.

        Returns:
            Observation: The initial observation after reset.
        """
        raise NotImplementedError

    @abstractmethod
    async def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Process the LLM response, execute actions, and return the next observation.

        Args:
            action (Action): The LLM response.

        Returns: observation, reward, done, info
            observation (Observation): The full next-round LLM input.
            reward (float): The immediate reward from the environment.
            done (bool): Whether the task has finished.
            info (dict[str, Any]): Additional information for debugging or
                metrics logging.
        """
        raise NotImplementedError
