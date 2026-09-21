from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _default_data_root() -> str:
    return os.getenv("ALFWORLD_DATA_ROOT", "data/alfworld")


@dataclass
class AlfworldTextworldEnv:
    data_root: str = field(default_factory=_default_data_root)
    max_episode_steps: int = 50
    _env: Any = field(init=False, default=None)
    _current_game_path: Path | None = field(init=False, default=None)

    def _games_root(self) -> Path:
        return Path(self.data_root).expanduser().resolve() / "games"

    def _resolve_game_path(self, game_relative_path: str) -> Path:
        game_path = self._games_root() / game_relative_path
        if not game_path.exists():
            raise FileNotFoundError(
                f"ALFWorld game file not found: {game_path}. Expected runtime assets under {self._games_root()}."
            )
        return game_path

    def _close_env(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()
        self._env = None
        self._current_game_path = None

    def _build_env(self, game_path: Path):
        try:
            import textworld
            import textworld.gym
        except ImportError as e:
            raise ImportError(
                "TextWorld runtime is not installed. Please install ALFWorld/TextWorld dependencies first."
            ) from e

        wrappers: list[Any] = []
        try:
            from alfworld.agents.utils.misc import Demangler

            class AlfredDemangler(textworld.core.Wrapper):
                def __init__(self, *args, shuffle: bool = False, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.shuffle = shuffle

                def load(self, *args, **kwargs):
                    super().load(*args, **kwargs)
                    demangler = Demangler(game_infos=self._entity_infos, shuffle=self.shuffle)
                    for info in self._entity_infos.values():
                        info.name = demangler.demangle_alfred_name(info.id)

            wrappers.append(AlfredDemangler(shuffle=False))
        except Exception:
            # Friendly feedback still exists in game.tw-pddl, so demangling is optional.
            pass

        class AlfredInfos(textworld.core.Wrapper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._gamefile = None

            def load(self, *args, **kwargs):
                super().load(*args, **kwargs)
                self._gamefile = args[0]

            def reset(self, *args, **kwargs):
                state = super().reset(*args, **kwargs)
                try:
                    state["extra.gamefile"] = self._gamefile
                except Exception:
                    pass
                return state

        wrappers.append(AlfredInfos)

        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            extras=["gamefile"],
        )
        env_id = textworld.gym.register_games(
            [str(game_path)],
            request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=self.max_episode_steps,
            wrappers=wrappers,
        )
        return textworld.gym.make(env_id)

    def _ensure_env(self, game_path: Path) -> None:
        if self._env is not None and self._current_game_path == game_path:
            return
        self._close_env()
        self._env = self._build_env(game_path)
        self._current_game_path = game_path

    @staticmethod
    def _unwrap_batch_item(value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        return value

    @staticmethod
    def _batch_info_to_single(info: dict[str, Any] | None) -> dict[str, Any]:
        return {key: AlfworldTextworldEnv._unwrap_batch_item(value) for key, value in dict(info or {}).items()}

    @staticmethod
    def _won_to_success(value: Any) -> bool:
        value = AlfworldTextworldEnv._unwrap_batch_item(value)
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        return bool(float(value))

    def _normalize_reset_output(self, reset_output: Any) -> tuple[Any, dict[str, Any]]:
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            obs, info = reset_output
            return obs, dict(info or {})
        return reset_output, {}

    def _normalize_step_output(self, step_output: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        if isinstance(step_output, tuple) and len(step_output) == 5:
            obs, reward, terminated, truncated, info = step_output
            reward = self._unwrap_batch_item(reward)
            terminated = self._unwrap_batch_item(terminated)
            truncated = self._unwrap_batch_item(truncated)
            info = self._unwrap_batch_item(info)
            info = dict(info or {})
            # Preserve Gymnasium's distinction through the legacy done interface.
            info.update(terminated=bool(terminated), truncated=bool(truncated))
            return obs, float(reward), bool(terminated or truncated), info
        if isinstance(step_output, tuple) and len(step_output) == 4:
            obs, reward, done, info = step_output
            reward = self._unwrap_batch_item(reward)
            done = self._unwrap_batch_item(done)
            info = self._unwrap_batch_item(info)
            return obs, float(reward), bool(done), dict(info or {})
        if isinstance(step_output, tuple) and len(step_output) == 3:
            obs, reward, done = step_output
            reward = self._unwrap_batch_item(reward)
            done = self._unwrap_batch_item(done)
            return obs, float(reward), bool(done), {}
        raise RuntimeError(f"Unsupported step() output: {type(step_output)} / {step_output!r}")

    @staticmethod
    def _state_to_observation(state: Any) -> str:
        state = AlfworldTextworldEnv._unwrap_batch_item(state)
        if isinstance(state, dict):
            if "feedback" in state:
                return str(state["feedback"])
            if "observation" in state:
                return str(state["observation"])
        return str(state)

    @staticmethod
    def _state_to_info(state: Any, base_info: dict[str, Any]) -> dict[str, Any]:
        state = AlfworldTextworldEnv._unwrap_batch_item(state)
        info = AlfworldTextworldEnv._batch_info_to_single(base_info)
        if isinstance(state, dict):
            for key in ("won", "admissible_commands", "extra.gamefile"):
                if key in state:
                    info[key] = AlfworldTextworldEnv._unwrap_batch_item(state[key])
        info["success"] = AlfworldTextworldEnv._won_to_success(info.get("won", 0.0))
        return info

    def reset_with_info(self, game_relative_path: str, task_id: str | None = None) -> tuple[str, dict[str, Any]]:
        del task_id  # task_id is carried for logging / debugging; game_relative_path is the runtime selector.
        game_path = self._resolve_game_path(game_relative_path)
        self._ensure_env(game_path)
        raw_state, base_info = self._normalize_reset_output(self._env.reset())
        raw_state = self._unwrap_batch_item(raw_state)
        return self._state_to_observation(raw_state), self._state_to_info(raw_state, base_info)

    def reset(self, game_relative_path: str, task_id: str | None = None) -> str:
        observation, _ = self.reset_with_info(game_relative_path=game_relative_path, task_id=task_id)
        return observation

    def step(self, action_str: str) -> tuple[str, float, bool, dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() before step().")

        raw_state, reward, done, base_info = self._normalize_step_output(self._env.step([action_str]))
        raw_state = self._unwrap_batch_item(raw_state)
        info = self._state_to_info(raw_state, base_info)
        observation = self._state_to_observation(raw_state)
        return observation, reward, done, info
