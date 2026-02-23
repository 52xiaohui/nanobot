"""Spawn tool for creating background subagents."""

from typing import Any, TYPE_CHECKING

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.providers.base import LLMProvider


class SpawnTool(Tool):
    """
    Tool to spawn a subagent for background task execution.
    
    The subagent runs asynchronously and announces its result back
    to the main agent when complete.
    """
    
    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._provider: "LLMProvider | None" = None
        self._model: str | None = None
        self._request_options: dict[str, Any] | None = None
    
    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id

    def set_runtime(
        self,
        provider: "LLMProvider",
        model: str,
        request_options: dict[str, Any] | None,
    ) -> None:
        """Set runtime snapshot used by spawned subagents."""
        self._provider = provider
        self._model = model
        self._request_options = dict(request_options) if request_options else None
    
    @property
    def name(self) -> str:
        return "spawn"
    
    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently. "
            "The subagent will complete the task and report back when done."
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task (for display)",
                },
            },
            "required": ["task"],
        }
    
    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        """Spawn a subagent to execute the given task."""
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            provider=self._provider,
            model=self._model,
            request_options=self._request_options,
        )
