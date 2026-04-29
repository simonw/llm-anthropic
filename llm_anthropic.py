from anthropic import Anthropic, AsyncAnthropic, transform_schema
import enum
import llm
from llm.parts import (
    AttachmentPart,
    Message,
    ReasoningPart,
    StreamEvent,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
import json
from typing import Any, Dict, Optional, List
from pydantic import Field, field_validator, model_validator

DEFAULT_THINKING_TOKENS = 1024
DEFAULT_TEMPERATURE = 1.0


class ThinkingEffort(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


@llm.hookimpl
def register_models(register):
    # https://docs.anthropic.com/claude/docs/models-overview
    register(
        ClaudeMessages("claude-3-opus-20240229"),
        AsyncClaudeMessages("claude-3-opus-20240229"),
    )
    register(
        ClaudeMessages("claude-3-opus-latest"),
        AsyncClaudeMessages("claude-3-opus-latest"),
        aliases=("claude-3-opus",),
    )
    register(
        ClaudeMessages("claude-3-sonnet-20240229"),
        AsyncClaudeMessages("claude-3-sonnet-20240229"),
        aliases=("claude-3-sonnet",),
    )
    register(
        ClaudeMessages("claude-3-haiku-20240307"),
        AsyncClaudeMessages("claude-3-haiku-20240307"),
        aliases=("claude-3-haiku",),
    )
    # 3.5 models
    register(
        ClaudeMessages(
            "claude-3-5-sonnet-20240620", supports_pdf=True, default_max_tokens=8192
        ),
        AsyncClaudeMessages(
            "claude-3-5-sonnet-20240620", supports_pdf=True, default_max_tokens=8192
        ),
    )
    register(
        ClaudeMessages(
            "claude-3-5-sonnet-20241022",
            supports_pdf=True,
            supports_web_search=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-3-5-sonnet-20241022",
            supports_pdf=True,
            supports_web_search=True,
            default_max_tokens=8192,
        ),
    )
    register(
        ClaudeMessages(
            "claude-3-5-sonnet-latest",
            supports_pdf=True,
            supports_web_search=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-3-5-sonnet-latest",
            supports_pdf=True,
            supports_web_search=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-3.5-sonnet", "claude-3.5-sonnet-latest"),
    )
    register(
        ClaudeMessages(
            "claude-3-5-haiku-latest", supports_web_search=True, default_max_tokens=8192
        ),
        AsyncClaudeMessages(
            "claude-3-5-haiku-latest", supports_web_search=True, default_max_tokens=8192
        ),
        aliases=("claude-3.5-haiku",),
    )
    # 3.7
    register(
        ClaudeMessages(
            "claude-3-7-sonnet-20250219",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-3-7-sonnet-20250219",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            default_max_tokens=8192,
        ),
    )
    register(
        ClaudeMessages(
            "claude-3-7-sonnet-latest",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-3-7-sonnet-latest",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-3.7-sonnet", "claude-3.7-sonnet-latest"),
    )
    register(
        ClaudeMessages(
            "claude-opus-4-0",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-opus-4-0",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-4-opus",),
    )
    register(
        ClaudeMessages(
            "claude-sonnet-4-0",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-sonnet-4-0",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-4-sonnet",),
    )
    register(
        ClaudeMessages(
            "claude-opus-4-1-20250805",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            use_structured_outputs=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-opus-4-1-20250805",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            use_structured_outputs=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-opus-4.1",),
    )
    # claude-sonnet-4-5
    register(
        ClaudeMessages(
            "claude-sonnet-4-5",
            supports_pdf=True,
            supports_thinking=True,
            use_structured_outputs=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-sonnet-4-5",
            supports_pdf=True,
            supports_thinking=True,
            use_structured_outputs=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-sonnet-4.5",),
    )
    # claude-haiku-4-5
    register(
        ClaudeMessages(
            "claude-haiku-4-5-20251001",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=64000,
        ),
        AsyncClaudeMessages(
            "claude-haiku-4-5-20251001",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=64000,
        ),
        aliases=("claude-haiku-4.5",),
    )
    # claude-opus-4-5
    register(
        ClaudeMessages(
            "claude-opus-4-5-20251101",
            supports_pdf=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_web_search=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-opus-4-5-20251101",
            supports_pdf=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_web_search=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-opus-4.5",),
    )
    # claude-opus-4-6
    register(
        ClaudeMessages(
            "claude-opus-4-6",
            supports_pdf=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_adaptive_thinking=True,
            supports_web_search=True,
            use_structured_outputs=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-opus-4-6",
            supports_pdf=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_adaptive_thinking=True,
            supports_web_search=True,
            use_structured_outputs=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-opus-4.6",),
    )
    # claude-sonnet-4-6
    register(
        ClaudeMessages(
            "claude-sonnet-4-6",
            supports_pdf=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_adaptive_thinking=True,
            supports_web_search=True,
            use_structured_outputs=True,
            default_max_tokens=64000,
        ),
        AsyncClaudeMessages(
            "claude-sonnet-4-6",
            supports_pdf=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_adaptive_thinking=True,
            supports_web_search=True,
            use_structured_outputs=True,
            default_max_tokens=64000,
        ),
        aliases=("claude-sonnet-4.6",),
    )
    # claude-opus-4-7
    register(
        ClaudeMessages(
            "claude-opus-4-7",
            supports_pdf=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_adaptive_thinking=True,
            supports_web_search=True,
            use_structured_outputs=True,
            default_max_tokens=128000,
        ),
        AsyncClaudeMessages(
            "claude-opus-4-7",
            supports_pdf=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_adaptive_thinking=True,
            supports_web_search=True,
            use_structured_outputs=True,
            default_max_tokens=128000,
        ),
        aliases=("claude-opus-4.7",),
    )


class ClaudeOptions(llm.Options):
    max_tokens: int | None = Field(
        description="The maximum number of tokens to generate before stopping",
        default=None,
    )

    temperature: float | None = Field(
        description="Amount of randomness injected into the response. Defaults to 1.0. Ranges from 0.0 to 1.0. Use temperature closer to 0.0 for analytical / multiple choice, and closer to 1.0 for creative and generative tasks. Note that even with temperature of 0.0, the results will not be fully deterministic.",
        default=None,
    )

    top_p: float | None = Field(
        description="Use nucleus sampling. In nucleus sampling, we compute the cumulative distribution over all the options for each subsequent token in decreasing probability order and cut it off once it reaches a particular probability specified by top_p. You should either alter temperature or top_p, but not both. Recommended for advanced use cases only. You usually only need to use temperature.",
        default=None,
    )

    top_k: int | None = Field(
        description="Only sample from the top K options for each subsequent token. Used to remove 'long tail' low probability responses. Recommended for advanced use cases only. You usually only need to use temperature.",
        default=None,
    )

    user_id: str | None = Field(
        description="An external identifier for the user who is associated with the request",
        default=None,
    )

    prefill: str | None = Field(
        description="A prefill to use for the response",
        default=None,
    )

    hide_prefill: bool | None = Field(
        description="Do not repeat the prefill value at the start of the response",
        default=None,
    )

    stop_sequences: list[str] | str | None = Field(
        description=(
            "Custom text sequences that will cause the model to stop generating - "
            "pass either a list of strings or a single string"
        ),
        default=None,
    )
    cache: bool | None = Field(
        description="Use Anthropic prompt cache for any attachments or fragments",
        default=None,
    )

    web_search: Optional[bool] = Field(
        description="Enable web search capabilities",
        default=None,
    )

    web_search_max_uses: Optional[int] = Field(
        description="Maximum number of web searches to perform per request",
        default=None,
    )

    web_search_allowed_domains: Optional[List[str]] = Field(
        description="List of domains to restrict web searches to",
        default=None,
    )

    web_search_blocked_domains: Optional[List[str]] = Field(
        description="List of domains to exclude from web searches",
        default=None,
    )

    web_search_location: Optional[dict] = Field(
        description="User location for localizing search results (dict with city, region, country, timezone)",
        default=None,
    )

    @field_validator("stop_sequences")
    def validate_stop_sequences(cls, stop_sequences):
        error_msg = "stop_sequences must be a list of strings or a single string"
        if isinstance(stop_sequences, str):
            try:
                stop_sequences = json.loads(stop_sequences)
                if not isinstance(stop_sequences, list) or not all(
                    isinstance(seq, str) for seq in stop_sequences
                ):
                    raise ValueError(error_msg)
                return stop_sequences
            except json.JSONDecodeError:
                return [stop_sequences]
        elif isinstance(stop_sequences, list):
            if not all(isinstance(seq, str) for seq in stop_sequences):
                raise ValueError(error_msg)
            return stop_sequences
        else:
            raise ValueError(error_msg)

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, temperature):
        if not (0.0 <= temperature <= 1.0):
            raise ValueError("temperature must be in range 0.0-1.0")
        return temperature

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, top_p):
        if top_p is not None and not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be in range 0.0-1.0")
        return top_p

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, top_k):
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        return top_k

    @field_validator("web_search_max_uses")
    @classmethod
    def validate_web_search_max_uses(cls, max_uses):
        if max_uses is not None and max_uses <= 0:
            raise ValueError("web_search_max_uses must be a positive integer")
        return max_uses

    @field_validator("web_search_allowed_domains", "web_search_blocked_domains")
    @classmethod
    def validate_web_search_domains(cls, domains):
        if domains is not None:
            if not isinstance(domains, list):
                raise ValueError("web_search domains must be a list of strings")
            if not all(isinstance(domain, str) for domain in domains):
                raise ValueError("web_search domains must be a list of strings")
        return domains

    @field_validator("web_search_location")
    @classmethod
    def validate_web_search_location(cls, location):
        if location is not None:
            if not isinstance(location, dict):
                raise ValueError("web_search_location must be a dictionary")
            required_fields = {"city", "region", "country", "timezone"}
            if not all(field in location for field in required_fields):
                raise ValueError(f"web_search_location must contain: {required_fields}")
        return location

    @model_validator(mode="after")
    def validate_temperature_top_p(self):
        if self.temperature != 1.0 and self.top_p is not None:
            raise ValueError("Only one of temperature and top_p can be set")
        return self

    @model_validator(mode="after")
    def validate_web_search_domains_conflict(self):
        if (
            self.web_search_allowed_domains is not None
            and self.web_search_blocked_domains is not None
        ):
            raise ValueError(
                "Cannot use both web_search_allowed_domains and web_search_blocked_domains"
            )
        return self


class ClaudeOptionsWithThinking(ClaudeOptions):
    thinking: bool | None = Field(
        description="Enable thinking mode",
        default=None,
    )
    thinking_budget: int | None = Field(
        description="Number of tokens to budget for thinking", default=None
    )
    thinking_display: bool | None = Field(
        description="Request summarized thinking output (available in --json logs)",
        default=None,
    )
    thinking_adaptive: bool | None = Field(
        description='Force adaptive thinking mode (sends thinking={"type": "adaptive"})',
        default=None,
    )


class ClaudeOptionsWithThinkingEffort(ClaudeOptionsWithThinking):
    thinking_effort: ThinkingEffort | None = Field(
        description="Level of thinking effort to apply: low, medium, or high",
        default=None,
    )


def source_for_attachment(attachment):
    if attachment.url:
        return {
            "type": "url",
            "url": attachment.url,
        }
    else:
        return {
            "data": attachment.base64_content(),
            "media_type": attachment.resolve_type(),
            "type": "base64",
        }


class _Shared:
    needs_key = "anthropic"
    key_env_var = "ANTHROPIC_API_KEY"
    can_stream = True
    base_url = None

    supports_thinking = False
    supports_thinking_effort = False
    supports_adaptive_thinking = False
    supports_schema = True
    supports_tools = True
    supports_web_search = False
    default_max_tokens = 4096

    class Options(ClaudeOptions): ...

    def __init__(
        self,
        model_id,
        claude_model_id=None,
        supports_images=True,
        supports_pdf=False,
        supports_thinking=False,
        supports_thinking_effort=False,
        supports_adaptive_thinking=False,
        supports_web_search=False,
        use_structured_outputs=False,
        default_max_tokens=None,
        base_url=None,
    ):
        self.model_id = "anthropic/" + model_id
        self.claude_model_id = claude_model_id or model_id
        self.base_url = base_url
        self.use_structured_outputs = use_structured_outputs
        self.attachment_types = set()
        if supports_images:
            self.attachment_types.update(
                {
                    "image/png",
                    "image/jpeg",
                    "image/webp",
                    "image/gif",
                }
            )
        if supports_pdf:
            self.attachment_types.add("application/pdf")
        if supports_thinking:
            self.supports_thinking = True
            self.Options = ClaudeOptionsWithThinking
        if supports_thinking_effort:
            self.supports_thinking_effort = True
            self.Options = ClaudeOptionsWithThinkingEffort
        if supports_adaptive_thinking:
            self.supports_adaptive_thinking = True
        if default_max_tokens is not None:
            self.default_max_tokens = default_max_tokens
        self.supports_web_search = supports_web_search

    def prefill_text(self, prompt):
        if prompt.options.prefill and not prompt.options.hide_prefill:
            return prompt.options.prefill
        return ""

    def _model_dump_suppress_warnings(self, message):
        """
        Call model_dump() on a message while suppressing Pydantic serialization warnings.

        When using dynamically created Pydantic models with the SDK's stream() helper,
        the returned ParsedBetaMessage has strict type annotations that don't match
        our dynamic models, causing harmless serialization warnings. This suppresses
        those warnings while still producing correct output.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            return message.model_dump()

    # --- messages= support -------------------------------------------------
    #
    # This plugin consumes prompt.messages (the canonical list[Message]
    # that llm synthesizes from legacy inputs when messages= wasn't
    # explicitly passed). Each Message + its Parts is translated into
    # Anthropic content blocks; adjacent user-side messages (role="user"
    # or role="tool") are merged because Anthropic requires alternating
    # user/assistant turns.

    def _part_to_block(self, part) -> Optional[Dict[str, Any]]:
        """Translate one llm Part into an Anthropic content block."""
        pm = getattr(part, "provider_metadata", None) or {}
        anthropic_pm = pm.get("anthropic", {}) if isinstance(pm, dict) else {}
        if isinstance(part, TextPart):
            block: Dict[str, Any] = {"type": "text", "text": part.text}
            return block
        if isinstance(part, ReasoningPart):
            block = {"type": "thinking", "thinking": part.text}
            # Anthropic signed-thinking requires the signature echoed back.
            sig = (
                anthropic_pm.get("signature")
                if isinstance(anthropic_pm, dict)
                else None
            )
            if sig:
                block["signature"] = sig
            return block
        if isinstance(part, ToolCallPart):
            return {
                "type": "tool_use",
                "id": part.tool_call_id,
                "name": part.name,
                "input": part.arguments,
            }
        if isinstance(part, ToolResultPart):
            return {
                "type": "tool_result",
                "tool_use_id": part.tool_call_id,
                "content": part.output,
            }
        if isinstance(part, AttachmentPart) and part.attachment is not None:
            attachment = part.attachment
            attachment_type = (
                "document"
                if attachment.resolve_type() == "application/pdf"
                else "image"
            )
            return {
                "type": attachment_type,
                "source": source_for_attachment(attachment),
            }
        return None

    def _message_to_blocks(self, message: Message) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        for part in message.parts:
            block = self._part_to_block(part)
            if block is not None:
                blocks.append(block)
        if message.role == "assistant":
            filtered_blocks: List[Dict[str, Any]] = []
            seen_tool_use = False
            for block in blocks:
                block_type = block.get("type")
                if (
                    seen_tool_use
                    and block_type == "text"
                    and block.get("text") == " "
                ):
                    # The sync streaming path yields a display-only space
                    # after tool calls so chained text does not run together.
                    # Anthropic rejects assistant history that places text
                    # after tool_use instead of immediately before tool_result.
                    continue
                filtered_blocks.append(block)
                if block_type == "tool_use":
                    seen_tool_use = True
            blocks = filtered_blocks
        return blocks

    def _append_message(self, out: List[Dict[str, Any]], message: Message) -> None:
        """Append an Anthropic-shaped message, merging with the previous one
        if both would be user-side turns (tool_result + text in the same
        user message is the required shape for tool follow-ups)."""
        if message.role == "system":
            return  # system lives on the top-level kwargs["system"] field
        blocks = self._message_to_blocks(message)
        if not blocks:
            return
        # Anthropic: tool messages from llm become user messages with
        # tool_result blocks; assistant stays assistant.
        anthropic_role = "assistant" if message.role == "assistant" else "user"
        if out and out[-1]["role"] == anthropic_role and anthropic_role == "user":
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": anthropic_role, "content": blocks})

    def _append_prev_response_output(
        self, out: List[Dict[str, Any]], prev_response
    ) -> None:
        """Add the assistant turn from a previous Response. Mirrors the
        flat text+tool_calls pattern used by the OpenAI plugin."""
        assistant_content: List[Dict[str, Any]] = []
        text_content = prev_response.text_or_raise()
        if text_content:
            assistant_content.append({"type": "text", "text": text_content})
        for tool_call in prev_response.tool_calls_or_raise():
            assistant_content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.tool_call_id,
                    "name": tool_call.name,
                    "input": tool_call.arguments,
                }
            )
        if assistant_content:
            out.append({"role": "assistant", "content": assistant_content})

    def build_messages(self, prompt, conversation) -> list[dict]:
        messages: List[Dict[str, Any]] = []

        # Current turn — iterate prompt.messages (auto-synthesized from
        # legacy inputs if messages= was not explicitly passed). In llm
        # 0.32+ conversation and chain paths pre-bake the full input chain
        # here, so also walking conversation.responses would duplicate
        # prior turns and break tool-result ordering.
        for message in prompt.messages:
            self._append_message(messages, message)

        # Cache control: apply to the last content block of the final
        # user-side turn, matching the pre-upgrade behavior.
        if prompt.options.cache and messages:
            last_message = messages[-1]
            if (
                isinstance(last_message.get("content"), list)
                and last_message["content"]
            ):
                last_message["content"][-1]["cache_control"] = {"type": "ephemeral"}

        # Prefill — append an assistant turn the model will continue from.
        if prompt.options.prefill:
            if self.supports_adaptive_thinking:
                raise ValueError(
                    f"Prefilling assistant messages is not supported by {self.claude_model_id}. "
                    f"Use structured outputs or system prompt instructions instead."
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": prompt.options.prefill}],
                }
            )

        return messages

    def _extract_system(self, prompt) -> Optional[str]:
        """Pull the system prompt from prompt.messages or prompt.system.

        ``prompt.system`` already composes ``_system`` + ``system_fragments``;
        if messages= was passed explicitly and it contains a system-role
        message, fall back to reading that.
        """
        if prompt.system:
            return prompt.system
        for message in prompt.messages:
            if message.role == "system":
                texts = [p.text for p in message.parts if isinstance(p, TextPart)]
                if texts:
                    return "\n\n".join(texts)
        return None

    def build_kwargs(self, prompt, conversation):
        if prompt.schema and prompt.tools:
            raise ValueError(
                "llm-anthropic does not yet support using both schema and tools in the same prompt"
            )

        # Validate web search support
        if prompt.options.web_search and not self.supports_web_search:
            raise ValueError(
                f"Web search is not supported by model {self.model_id}. "
                f"Supported models include: claude-3.5-sonnet-latest, claude-3.5-haiku-latest, "
                f"claude-3.7-sonnet-latest, claude-4-opus, claude-4-sonnet, claude-opus-4.1, "
                f"claude-opus-4.6, claude-sonnet-4.6"
            )

        kwargs = {
            "model": self.claude_model_id,
            "messages": self.build_messages(prompt, conversation),
        }
        if prompt.options.user_id:
            kwargs["metadata"] = {"user_id": prompt.options.user_id}

        if prompt.options.top_p:
            kwargs["top_p"] = prompt.options.top_p
        else:
            kwargs["temperature"] = (
                prompt.options.temperature
                if prompt.options.temperature is not None
                else DEFAULT_TEMPERATURE
            )

        if prompt.options.top_k:
            kwargs["top_k"] = prompt.options.top_k

        system = self._extract_system(prompt)
        if system:
            kwargs["system"] = system

        if prompt.options.stop_sequences:
            kwargs["stop_sequences"] = prompt.options.stop_sequences

        thinking_effort_enabled = (
            self.supports_thinking_effort and prompt.options.thinking_effort
        )

        # Determine if thinking should be activated
        thinking_requested = False
        if self.supports_thinking:
            thinking_requested = (
                prompt.options.thinking
                or prompt.options.thinking_budget
                or prompt.options.thinking_display
                or prompt.options.thinking_adaptive
                or thinking_effort_enabled
            )

        if self.supports_thinking and thinking_requested:
            prompt.options.thinking = True
            if prompt.options.thinking_adaptive or thinking_effort_enabled:
                kwargs["thinking"] = {"type": "adaptive"}
            elif prompt.options.thinking_budget:
                # Explicit budget = manual mode (deprecated on 4.6 but still works)
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": prompt.options.thinking_budget,
                }
            elif self.supports_adaptive_thinking:
                # 4.6 models default to adaptive thinking
                kwargs["thinking"] = {"type": "adaptive"}
            else:
                # Pre-4.6 models: enabled with default budget
                budget_tokens = DEFAULT_THINKING_TOKENS
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }

            if prompt.options.thinking_display:
                kwargs["thinking"]["display"] = "summarized"

        # Handle effort in output_config
        if thinking_effort_enabled:
            if prompt.options.thinking_effort == ThinkingEffort.MAX:
                if not (
                    self.supports_adaptive_thinking and "opus" in self.claude_model_id
                ):
                    raise ValueError(
                        "thinking_effort='max' is only supported by claude-opus-4-6"
                    )
            kwargs.setdefault("output_config", {})[
                "effort"
            ] = prompt.options.thinking_effort.value

        max_tokens = self.default_max_tokens
        if prompt.options.max_tokens is not None:
            max_tokens = prompt.options.max_tokens
        if (
            self.supports_thinking
            and prompt.options.thinking_budget is not None
            and prompt.options.thinking_budget > max_tokens
        ):
            max_tokens = prompt.options.thinking_budget + 1
        kwargs["max_tokens"] = max_tokens

        # Determine which beta headers to use
        betas = []

        # Effort beta: only for pre-GA models (e.g., Opus 4.5)
        if (
            "output_config" in kwargs
            and "effort" in kwargs.get("output_config", {})
            and not self.supports_adaptive_thinking
        ):
            betas.append("effort-2025-11-24")

        # 128K output beta: not needed for 4.6 models
        if max_tokens > 64000 and not self.supports_adaptive_thinking:
            betas.append("output-128k-2025-02-19")
            if "thinking" in kwargs:
                kwargs["extra_body"] = {"thinking": kwargs.pop("thinking")}

        # Check if we should use new structured outputs
        use_structured_outputs = prompt.schema and self.use_structured_outputs

        if use_structured_outputs:
            kwargs.setdefault("output_config", {})["format"] = {
                "type": "json_schema",
                "schema": transform_schema(prompt.schema),
            }

        if betas:
            kwargs["betas"] = betas

        tools = []

        # Add web search tool if enabled
        if prompt.options.web_search:
            web_search_tool = {
                "type": "web_search_20250305",
                "name": "web_search",
            }

            # Add optional web search parameters
            if prompt.options.web_search_max_uses:
                web_search_tool["max_uses"] = prompt.options.web_search_max_uses

            if prompt.options.web_search_allowed_domains:
                web_search_tool["allowed_domains"] = (
                    prompt.options.web_search_allowed_domains
                )

            if prompt.options.web_search_blocked_domains:
                web_search_tool["blocked_domains"] = (
                    prompt.options.web_search_blocked_domains
                )

            if prompt.options.web_search_location:
                location = prompt.options.web_search_location.copy()
                location["type"] = "approximate"  # Required by API
                web_search_tool["user_location"] = location

            tools.append(web_search_tool)

        if prompt.schema and not use_structured_outputs:
            # Fall back to tools workaround for models that don't support structured outputs
            tools.append(
                {
                    "name": "output_structured_data",
                    "input_schema": prompt.schema,
                }
            )
            kwargs["tool_choice"] = {"type": "tool", "name": "output_structured_data"}

        if prompt.tools:
            tools.extend(
                [
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.input_schema,
                    }
                    for tool in prompt.tools
                ]
            )

        if tools:
            kwargs["tools"] = tools

        return kwargs

    def set_usage(self, response):
        usage = response.response_json.pop("usage")
        input_tokens = usage.pop("input_tokens")
        output_tokens = usage.pop("output_tokens")
        # Only include usage details if prompt caching was on or web search was used
        details = None
        if response.prompt.options.cache or usage.get("server_tool_use"):
            details = usage
        response.set_usage(input=input_tokens, output=output_tokens, details=details)

    def add_tool_usage(self, response, last_message) -> bool:
        tool_uses = [
            item for item in last_message["content"] if item["type"] == "tool_use"
        ]
        for tool_use in tool_uses:
            response.add_tool_call(
                llm.ToolCall(
                    tool_call_id=tool_use["id"],
                    name=tool_use["name"],
                    arguments=tool_use["input"],
                )
            )
        return bool(tool_uses)

    def __str__(self):
        return "Anthropic Messages: {}".format(self.model_id)


class ClaudeMessages(_Shared, llm.KeyModel):
    def execute(self, prompt, stream, response, conversation, key):
        client = Anthropic(api_key=self.get_key(key), base_url=self.base_url)
        kwargs = self.build_kwargs(prompt, conversation)
        prefill_text = self.prefill_text(prompt)
        if "betas" in kwargs:
            messages_client = client.beta.messages
        else:
            messages_client = client.messages

        if stream:
            with messages_client.stream(**kwargs) as stream_obj:
                current_block_id = None
                current_block_name = None
                is_server_tool = False

                if prefill_text:
                    yield StreamEvent(type="text", chunk=prefill_text)

                for chunk in stream_obj:
                    if chunk.type == "content_block_start":
                        block = chunk.content_block
                        block_type = getattr(block, "type", None)
                        current_block_id = getattr(block, "id", None)
                        current_block_name = getattr(block, "name", None)
                        is_server_tool = block_type in (
                            "server_tool_use",
                            "web_search_tool_result",
                        )

                        if block_type in ("tool_use", "server_tool_use"):
                            yield StreamEvent(
                                type="tool_call_name",
                                chunk=current_block_name or "",
                                tool_call_id=current_block_id,
                                server_executed=(block_type == "server_tool_use"),
                            )
                        elif block_type == "web_search_tool_result":
                            # Content is available inline on content_block_start
                            tool_use_id = getattr(block, "tool_use_id", None)
                            result_content = getattr(block, "content", [])
                            if result_content:
                                result_text = json.dumps(
                                    [
                                        b if isinstance(b, dict) else b.model_dump()
                                        for b in result_content
                                    ]
                                )
                            else:
                                result_text = ""
                            yield StreamEvent(
                                type="tool_result",
                                chunk=result_text,
                                tool_call_id=tool_use_id,
                                server_executed=True,
                                tool_name="web_search",
                            )

                    elif chunk.type == "content_block_delta":
                        delta = chunk.delta
                        delta_type = getattr(delta, "type", None)

                        if delta_type == "thinking_delta":
                            yield StreamEvent(type="reasoning", chunk=delta.thinking)
                        elif delta_type == "text_delta":
                            yield StreamEvent(type="text", chunk=delta.text)
                        elif delta_type == "input_json_delta":
                            yield StreamEvent(
                                type="tool_call_args",
                                chunk=delta.partial_json,
                                tool_call_id=current_block_id,
                                server_executed=is_server_tool,
                            )

                # This records usage and other data:
                last_message = self._model_dump_suppress_warnings(
                    stream_obj.get_final_message()
                )
                response.response_json = last_message

                if self.add_tool_usage(response, last_message):
                    # Avoid "can have dragons.Now that I " bug
                    yield StreamEvent(type="text", chunk=" ")
        else:
            completion = messages_client.create(**kwargs)
            for item in completion.content:
                item_type = getattr(item, "type", None)
                if item_type == "thinking":
                    yield StreamEvent(type="reasoning", chunk=item.thinking)
                elif item_type == "text":
                    text = (prefill_text + item.text) if prefill_text else item.text
                    prefill_text = ""  # Only prepend once
                    yield StreamEvent(type="text", chunk=text)
                elif item_type in ("tool_use", "server_tool_use"):
                    yield StreamEvent(
                        type="tool_call_name",
                        chunk=item.name,
                        tool_call_id=item.id,
                        server_executed=(item_type == "server_tool_use"),
                    )
                    yield StreamEvent(
                        type="tool_call_args",
                        chunk=json.dumps(item.input),
                        tool_call_id=item.id,
                        server_executed=(item_type == "server_tool_use"),
                    )
                elif item_type == "web_search_tool_result":
                    result_content = getattr(item, "content", [])
                    result_text = (
                        json.dumps(
                            [
                                block if isinstance(block, dict) else block.model_dump()
                                for block in result_content
                            ]
                        )
                        if result_content
                        else ""
                    )
                    yield StreamEvent(
                        type="tool_result",
                        chunk=result_text,
                        tool_call_id=getattr(item, "tool_use_id", None),
                        server_executed=True,
                        tool_name="web_search",
                    )
            response.response_json = completion.model_dump()
            self.add_tool_usage(response, response.response_json)
        self.set_usage(response)


class AsyncClaudeMessages(_Shared, llm.AsyncKeyModel):
    async def execute(self, prompt, stream, response, conversation, key):
        client = AsyncAnthropic(api_key=self.get_key(key), base_url=self.base_url)
        kwargs = self.build_kwargs(prompt, conversation)
        if "betas" in kwargs:
            messages_client = client.beta.messages
        else:
            messages_client = client.messages
        prefill_text = self.prefill_text(prompt)

        if stream:
            async with messages_client.stream(**kwargs) as stream_obj:
                current_block_id = None
                current_block_name = None
                is_server_tool = False

                if prefill_text:
                    yield StreamEvent(type="text", chunk=prefill_text)

                async for chunk in stream_obj:
                    if chunk.type == "content_block_start":
                        block = chunk.content_block
                        block_type = getattr(block, "type", None)
                        current_block_id = getattr(block, "id", None)
                        current_block_name = getattr(block, "name", None)
                        is_server_tool = block_type in (
                            "server_tool_use",
                            "web_search_tool_result",
                        )

                        if block_type in ("tool_use", "server_tool_use"):
                            yield StreamEvent(
                                type="tool_call_name",
                                chunk=current_block_name or "",
                                tool_call_id=current_block_id,
                                server_executed=(block_type == "server_tool_use"),
                            )
                        elif block_type == "web_search_tool_result":
                            tool_use_id = getattr(block, "tool_use_id", None)
                            result_content = getattr(block, "content", [])
                            if result_content:
                                result_text = json.dumps(
                                    [
                                        b if isinstance(b, dict) else b.model_dump()
                                        for b in result_content
                                    ]
                                )
                            else:
                                result_text = ""
                            yield StreamEvent(
                                type="tool_result",
                                chunk=result_text,
                                tool_call_id=tool_use_id,
                                server_executed=True,
                                tool_name="web_search",
                            )

                    elif chunk.type == "content_block_delta":
                        delta = chunk.delta
                        delta_type = getattr(delta, "type", None)

                        if delta_type == "thinking_delta":
                            yield StreamEvent(type="reasoning", chunk=delta.thinking)
                        elif delta_type == "text_delta":
                            yield StreamEvent(type="text", chunk=delta.text)
                        elif delta_type == "input_json_delta":
                            yield StreamEvent(
                                type="tool_call_args",
                                chunk=delta.partial_json,
                                tool_call_id=current_block_id,
                                server_executed=is_server_tool,
                            )

            response.response_json = self._model_dump_suppress_warnings(
                await stream_obj.get_final_message()
            )

            self.add_tool_usage(response, response.response_json)
        else:
            completion = await messages_client.create(**kwargs)
            for item in completion.content:
                item_type = getattr(item, "type", None)
                if item_type == "thinking":
                    yield StreamEvent(type="reasoning", chunk=item.thinking)
                elif item_type == "text":
                    text = (prefill_text + item.text) if prefill_text else item.text
                    prefill_text = ""
                    yield StreamEvent(type="text", chunk=text)
                elif item_type in ("tool_use", "server_tool_use"):
                    yield StreamEvent(
                        type="tool_call_name",
                        chunk=item.name,
                        tool_call_id=item.id,
                        server_executed=(item_type == "server_tool_use"),
                    )
                    yield StreamEvent(
                        type="tool_call_args",
                        chunk=json.dumps(item.input),
                        tool_call_id=item.id,
                        server_executed=(item_type == "server_tool_use"),
                    )
                elif item_type == "web_search_tool_result":
                    result_content = getattr(item, "content", [])
                    result_text = (
                        json.dumps(
                            [
                                block if isinstance(block, dict) else block.model_dump()
                                for block in result_content
                            ]
                        )
                        if result_content
                        else ""
                    )
                    yield StreamEvent(
                        type="tool_result",
                        chunk=result_text,
                        tool_call_id=getattr(item, "tool_use_id", None),
                        server_executed=True,
                        tool_name="web_search",
                    )
            response.response_json = completion.model_dump()
            self.add_tool_usage(response, response.response_json)
        self.set_usage(response)
