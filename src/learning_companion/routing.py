"""Cost-safe model routing for Learning Companion."""

from __future__ import annotations

from dataclasses import dataclass

from learning_companion.settings import Settings

_SAFE_MODEL = "deepseek-v4-flash"
_PRO_MODEL_MARKERS = ("deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner", "/deepseek-chat", "/deepseek-v4-pro")


@dataclass(frozen=True)
class ModelRoute:
    """Effective model/call parameters for one LLM stage."""

    model: str
    complexity: str
    max_tokens: int
    temperature: float
    cache: bool


class ModelRouter:
    """Route LLM calls while enforcing Flash-only cost safety."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def route(
        self,
        *,
        stage: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> ModelRoute:
        text = "\n".join(str(message.get("content", "")) for message in messages)
        complexity = self._classify(stage=stage, prompt_chars=len(text))

        if complexity == "simple":
            configured_model = self.settings.router_simple_model
            token_limit = self.settings.router_simple_max_tokens
            effective_temperature = min(temperature, 0.2)
            cache = True
        elif complexity == "complex":
            configured_model = self.settings.router_complex_model
            token_limit = self.settings.router_complex_max_tokens
            effective_temperature = min(temperature, 0.3)
            cache = temperature == 0
        else:
            # standard — pass temperature through as-is
            configured_model = self.settings.router_standard_model
            token_limit = self.settings.router_standard_max_tokens
            effective_temperature = temperature
            cache = temperature == 0

        return ModelRoute(
            model=self._safe_model(configured_model),
            complexity=complexity,
            max_tokens=min(max_tokens, token_limit),
            temperature=effective_temperature,
            cache=cache,
        )

    def _classify(self, *, stage: str, prompt_chars: int) -> str:
        simple_stages = {"planner", "check.questions", "questions.fallback"}
        complex_stages = {"analyst.merge", "writer"}
        if stage in simple_stages:
            return "simple"
        if stage in complex_stages or prompt_chars >= self.settings.router_complex_min_chars:
            return "complex"
        if stage != "unknown" and prompt_chars <= self.settings.router_simple_max_chars:
            return "simple"
        return "standard"

    @staticmethod
    def _safe_model(model: str) -> str:
        normalized = model.lower()
        if any(marker in normalized for marker in _PRO_MODEL_MARKERS):
            return _SAFE_MODEL
        if "deepseek" in normalized and "v4-flash" not in normalized:
            return _SAFE_MODEL
        return model or _SAFE_MODEL
