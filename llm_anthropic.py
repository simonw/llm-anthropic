from anthropic import Anthropic, AsyncAnthropic, transform_schema
import enum
import llm
from llm.models import _partition_tools
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
from urllib.parse import urlsplit
from pydantic import Field, field_validator, model_validator

DEFAULT_THINKING_TOKENS = 1024
DEFAULT_TEMPERATURE = 1.0
MCP_BETA = "mcp-client-2025-11-20"


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
            default_max_tokens=32000,
        ),
        AsyncClaudeMessages(
            "claude-opus-4-0",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            default_max_tokens=32000,
        ),
        aliases=("claude-4-opus",),
    )
    register(
        ClaudeMessages(
            "claude-sonnet-4-0",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            default_max_tokens=64000,
        ),
        AsyncClaudeMessages(
            "claude-sonnet-4-0",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            default_max_tokens=64000,
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
            default_max_tokens=32000,
        ),
        AsyncClaudeMessages(
            "claude-opus-4-1-20250805",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            use_structured_outputs=True,
            default_max_tokens=32000,
        ),
        aliases=("claude-opus-4.1",),
    )
    # claude-sonnet-4-5
    register(
        ClaudeMessages(
            "claude-sonnet-4-5",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            supports_code_execution=True,
            use_structured_outputs=True,
            default_max_tokens=64000,
        ),
        AsyncClaudeMessages(
            "claude-sonnet-4-5",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            supports_code_execution=True,
            use_structured_outputs=True,
            default_max_tokens=64000,
        ),
        aliases=("claude-sonnet-4.5",),
    )
    # claude-haiku-4-5
    register(
        ClaudeMessages(
            "claude-haiku-4-5-20251001",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            supports_code_execution=True,
            use_structured_outputs=True,
            default_max_tokens=64000,
        ),
        AsyncClaudeMessages(
            "claude-haiku-4-5-20251001",
            supports_pdf=True,
            supports_thinking=True,
            supports_web_search=True,
            supports_code_execution=True,
            use_structured_outputs=True,
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
            supports_code_execution=True,
            default_max_tokens=64000,
        ),
        AsyncClaudeMessages(
            "claude-opus-4-5-20251101",
            supports_pdf=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_web_search=True,
            supports_code_execution=True,
            default_max_tokens=64000,
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
            supports_code_execution=True,
            use_structured_outputs=True,
            default_max_tokens=128000,
        ),
        AsyncClaudeMessages(
            "claude-opus-4-6",
            supports_pdf=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_adaptive_thinking=True,
            supports_web_search=True,
            supports_code_execution=True,
            use_structured_outputs=True,
            default_max_tokens=128000,
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
            supports_code_execution=True,
            use_structured_outputs=True,
            default_max_tokens=128000,
        ),
        AsyncClaudeMessages(
            "claude-sonnet-4-6",
            supports_pdf=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_adaptive_thinking=True,
            supports_web_search=True,
            supports_code_execution=True,
            use_structured_outputs=True,
            default_max_tokens=128000,
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
            supports_code_execution=True,
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
            supports_code_execution=True,
            use_structured_outputs=True,
            default_max_tokens=128000,
        ),
        aliases=("claude-opus-4.7",),
    )
    # claude-opus-4-8
    register(
        ClaudeMessages(
            "claude-opus-4-8",
            supports_pdf=True,
            supports_system_messages=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_adaptive_thinking=True,
            supports_web_search=True,
            supports_code_execution=True,
            use_structured_outputs=True,
            default_max_tokens=128000,
        ),
        AsyncClaudeMessages(
            "claude-opus-4-8",
            supports_pdf=True,
            supports_system_messages=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_adaptive_thinking=True,
            supports_web_search=True,
            supports_code_execution=True,
            use_structured_outputs=True,
            default_max_tokens=128000,
        ),
        aliases=("claude-opus-4.8",),
    )
    # claude-fable-5
    register(
        ClaudeMessages(
            "claude-fable-5",
            supports_pdf=True,
            supports_system_messages=True,
            thinks_by_default=True,
            always_thinks=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_adaptive_thinking=True,
            supports_web_search=True,
            supports_code_execution=True,
            use_structured_outputs=True,
            default_max_tokens=128000,
        ),
        AsyncClaudeMessages(
            "claude-fable-5",
            supports_pdf=True,
            supports_system_messages=True,
            thinks_by_default=True,
            always_thinks=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_adaptive_thinking=True,
            supports_web_search=True,
            supports_code_execution=True,
            use_structured_outputs=True,
            default_max_tokens=128000,
        ),
        aliases=("claude-fable-5",),
    )
    # claude-sonnet-5
    register(
        ClaudeMessages(
            "claude-sonnet-5",
            supports_pdf=True,
            supports_system_messages=True,
            thinks_by_default=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_adaptive_thinking=True,
            supports_web_search=True,
            supports_code_execution=True,
            use_structured_outputs=True,
            default_max_tokens=128000,
        ),
        AsyncClaudeMessages(
            "claude-sonnet-5",
            supports_pdf=True,
            supports_system_messages=True,
            thinks_by_default=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_adaptive_thinking=True,
            supports_web_search=True,
            supports_code_execution=True,
            use_structured_outputs=True,
            default_max_tokens=128000,
        ),
        aliases=("claude-sonnet-5",),
    )
    # claude-opus-5
    register(
        ClaudeMessages(
            "claude-opus-5",
            supports_pdf=True,
            supports_system_messages=True,
            thinks_by_default=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_adaptive_thinking=True,
            supports_web_search=True,
            supports_code_execution=True,
            use_structured_outputs=True,
            default_max_tokens=128000,
        ),
        AsyncClaudeMessages(
            "claude-opus-5",
            supports_pdf=True,
            supports_system_messages=True,
            thinks_by_default=True,
            supports_thinking=True,
            supports_thinking_effort=True,
            supports_adaptive_thinking=True,
            supports_web_search=True,
            supports_code_execution=True,
            use_structured_outputs=True,
            default_max_tokens=128000,
        ),
        aliases=("claude-opus-5",),
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

    fast: bool | None = Field(
        description="Use fast mode for lower latency responses: https://platform.claude.com/docs/en/build-with-claude/fast-mode",
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
        if temperature is not None and not (0.0 <= temperature <= 1.0):
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

    @model_validator(mode="after")
    def validate_temperature_top_p(self):
        if self.temperature is not None and self.top_p is not None:
            raise ValueError("Only one of temperature and top_p can be set")
        return self


class ClaudeOptionsWithThinking(ClaudeOptions):
    thinking: bool | None = Field(
        description=(
            "Enable thinking mode. Claude 5 models think by default - "
            "set to false to disable thinking on models that allow it"
        ),
        default=None,
    )


class ClaudeOptionsWithThinkingEffort(ClaudeOptionsWithThinking):
    thinking_effort: ThinkingEffort | None = Field(
        description="Level of thinking effort to apply: low, medium, high, xhigh or max",
        default=None,
    )


def _validate_max_uses(max_uses):
    if max_uses is not None and (
        isinstance(max_uses, bool) or not isinstance(max_uses, int) or max_uses < 1
    ):
        raise ValueError("max_uses must be a positive integer")


def _validate_domain_filters(allowed_domains, blocked_domains):
    if allowed_domains is not None and blocked_domains is not None:
        raise ValueError("Cannot specify both allowed_domains and blocked_domains")
    for name, domains in (
        ("allowed_domains", allowed_domains),
        ("blocked_domains", blocked_domains),
    ):
        if domains is None:
            continue
        if not isinstance(domains, list) or not all(
            isinstance(domain, str) and domain for domain in domains
        ):
            raise ValueError(f"{name} must be a list of non-empty strings")


class WebSearch(llm.ServerSideTool):
    """Search the web using Anthropic's server-side web search tool.

    On Claude 4.6 and later models this uses ``web_search_20260318`` with
    dynamic content filtering; older models use ``web_search_20250305``.
    """

    name = "web_search"

    def __init__(
        self,
        max_uses: Optional[int] = None,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
        user_location: Optional[dict] = None,
    ):
        super().__init__()
        _validate_max_uses(max_uses)
        _validate_domain_filters(allowed_domains, blocked_domains)
        if user_location is not None:
            if not isinstance(user_location, dict):
                raise ValueError("user_location must be a dictionary")
            allowed_keys = {"type", "city", "region", "country", "timezone"}
            invalid_keys = set(user_location.keys()) - allowed_keys
            if invalid_keys:
                raise ValueError(
                    f"user_location contains invalid keys: {invalid_keys}. "
                    f"Allowed keys: {allowed_keys}"
                )
            user_location = dict(user_location)
            user_location.setdefault("type", "approximate")
            if user_location["type"] != "approximate":
                raise ValueError("user_location type must be approximate")
        self.max_uses = max_uses
        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains
        self.user_location = user_location

    def tool_spec(self, model):
        modern = getattr(model, "supports_adaptive_thinking", False)
        spec = {
            "type": "web_search_20260318" if modern else "web_search_20250305",
            "name": "web_search",
        }
        if self.max_uses is not None:
            spec["max_uses"] = self.max_uses
        if self.allowed_domains is not None:
            spec["allowed_domains"] = list(self.allowed_domains)
        if self.blocked_domains is not None:
            spec["blocked_domains"] = list(self.blocked_domains)
        if self.user_location is not None:
            spec["user_location"] = dict(self.user_location)
        return spec


class WebFetch(llm.ServerSideTool):
    """Fetch the full contents of a URL using Anthropic's server-side web
    fetch tool.

    Claude can only fetch URLs that already appear in the conversation -
    provided by the user or returned by a previous web search or fetch.
    On Claude 4.6 and later models this uses ``web_fetch_20260318`` with
    dynamic content filtering; older models use ``web_fetch_20250910``.
    """

    name = "web_fetch"

    def __init__(
        self,
        max_uses: Optional[int] = None,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
        citations: bool = False,
        max_content_tokens: Optional[int] = None,
        use_cache: Optional[bool] = None,
    ):
        super().__init__()
        _validate_max_uses(max_uses)
        _validate_domain_filters(allowed_domains, blocked_domains)
        if not isinstance(citations, bool):
            raise ValueError("citations must be a boolean")
        if max_content_tokens is not None and (
            isinstance(max_content_tokens, bool)
            or not isinstance(max_content_tokens, int)
            or max_content_tokens < 1
        ):
            raise ValueError("max_content_tokens must be a positive integer")
        if use_cache is not None and not isinstance(use_cache, bool):
            raise ValueError("use_cache must be a boolean")
        self.max_uses = max_uses
        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains
        self.citations = citations
        self.max_content_tokens = max_content_tokens
        self.use_cache = use_cache

    def tool_spec(self, model):
        modern = getattr(model, "supports_adaptive_thinking", False)
        if self.use_cache is not None and not modern:
            raise ValueError(
                f"use_cache is not supported by model {model.model_id} - "
                "it requires a Claude 4.6 or later model"
            )
        spec = {
            "type": "web_fetch_20260318" if modern else "web_fetch_20250910",
            "name": "web_fetch",
        }
        if self.max_uses is not None:
            spec["max_uses"] = self.max_uses
        if self.allowed_domains is not None:
            spec["allowed_domains"] = list(self.allowed_domains)
        if self.blocked_domains is not None:
            spec["blocked_domains"] = list(self.blocked_domains)
        if self.citations:
            spec["citations"] = {"enabled": True}
        if self.max_content_tokens is not None:
            spec["max_content_tokens"] = self.max_content_tokens
        if self.use_cache is not None:
            spec["use_cache"] = self.use_cache
        return spec


class AnthropicMCP(llm.ServerSideTool):
    """Call tools on a remote MCP server using Anthropic's MCP connector.

    Anthropic connects to the MCP server from their own infrastructure -
    the server must be reachable over HTTPS. Uses the ``mcp-client-2025-11-20``
    beta. Only MCP tool calls are supported (not resources or prompts).
    """

    name = "mcp"

    def __init__(
        self,
        url: str,
        name: Optional[str] = None,
        authorization_token: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
    ):
        super().__init__()
        if not isinstance(url, str) or not url:
            raise ValueError("url must be a non-empty string")
        if not url.startswith("https://"):
            raise ValueError("url must start with https://")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("name must be a non-empty string")
        if authorization_token is not None and not isinstance(authorization_token, str):
            raise ValueError("authorization_token must be a string")
        if allowed_tools is not None:
            if not isinstance(allowed_tools, list) or not all(
                isinstance(tool_name, str) and tool_name for tool_name in allowed_tools
            ):
                raise ValueError("allowed_tools must be a list of non-empty strings")
        self.url = url
        self.server_name = name or urlsplit(url).hostname
        self.authorization_token = authorization_token
        self.allowed_tools = allowed_tools

    def tool_spec(self, model):
        spec = {"type": "mcp_toolset", "mcp_server_name": self.server_name}
        if self.allowed_tools is not None:
            spec["default_config"] = {"enabled": False}
            spec["configs"] = {
                tool_name: {"enabled": True} for tool_name in self.allowed_tools
            }
        return spec

    def prepare_request(self, model, kwargs):
        server = {"type": "url", "url": self.url, "name": self.server_name}
        if self.authorization_token is not None:
            server["authorization_token"] = self.authorization_token
        servers = kwargs.setdefault("mcp_servers", [])
        if not any(existing.get("name") == self.server_name for existing in servers):
            servers.append(server)
        betas = kwargs.setdefault("betas", [])
        if MCP_BETA not in betas:
            betas.append(MCP_BETA)


class CodeExecution(llm.ServerSideTool):
    """Run Python and bash code in Anthropic's sandboxed server-side
    execution container.

    Pass an existing container ID as ``container`` to reuse the files and
    state from a previous response - the container ID is available in the
    logged response JSON.
    """

    name = "code_execution"

    def __init__(self, container: Optional[str] = None):
        super().__init__()
        if container is not None and not isinstance(container, str):
            raise ValueError("container must be a string container ID")
        self.container = container

    def tool_spec(self, model):
        return {"type": "code_execution_20260521", "name": "code_execution"}

    def prepare_request(self, model, kwargs):
        if self.container is not None:
            kwargs["container"] = self.container


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
    supports_code_execution = False
    thinks_by_default = False
    always_thinks = False
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
        supports_code_execution=False,
        thinks_by_default=False,
        always_thinks=False,
        use_structured_outputs=False,
        supports_system_messages=False,
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
        self.supports_code_execution = supports_code_execution
        self.thinks_by_default = thinks_by_default
        self.always_thinks = always_thinks
        self.supports_system_messages = supports_system_messages

    @property
    def supported_server_side_tools(self):
        tools = []
        if self.supports_web_search:
            tools += [WebSearch, WebFetch, AnthropicMCP]
        if self.supports_code_execution:
            tools.append(CodeExecution)
        return tuple(tools)

    def prefill_text(self, prompt):
        if prompt.options.prefill and not prompt.options.hide_prefill:
            return prompt.options.prefill
        return ""

    def _server_tool_result_event(self, block_type, block) -> StreamEvent:
        """Build a tool_result StreamEvent from a server tool result block.

        web_search_tool_result content is a list of result blocks;
        web_fetch_tool_result content is a single result object.
        """
        content = getattr(block, "content", None)
        if isinstance(content, list):
            result_text = (
                json.dumps(
                    [b if isinstance(b, dict) else b.model_dump() for b in content]
                )
                if content
                else ""
            )
        elif content is None:
            result_text = ""
        else:
            result_text = json.dumps(
                content if isinstance(content, dict) else content.model_dump()
            )
        return StreamEvent(
            type="tool_result",
            chunk=result_text,
            tool_call_id=getattr(block, "tool_use_id", None),
            server_executed=True,
            tool_name=block_type.removesuffix("_tool_result"),
        )

    def _block_part_index(self, chunk) -> Optional[int]:
        """Explicit StreamEvent part_index for an Anthropic content block.

        Each Anthropic block index gets its own part index so that
        consecutive thinking / redacted_thinking blocks assemble into
        distinct ReasoningParts instead of merging - merging would
        concatenate their text and keep only the last signature, and
        Anthropic rejects continuations whose thinking blocks don't
        match what the model generated. Offset by 1 so the prefill
        text event can use part_index 0 without colliding.
        """
        index = getattr(chunk, "index", None)
        return None if index is None else index + 1

    def _apply_container(self, message_dict, container):
        """Store the code execution container on the response JSON.

        Streaming needs this patched in from the message_delta event -
        the SDK's get_final_message() accumulator drops it - and in both
        modes the datetime expires_at must become a JSON-safe string.
        """
        if container is None:
            container = message_dict.get("container")
        if container is None:
            return
        if hasattr(container, "model_dump"):
            container = container.model_dump(mode="json")
        else:
            container = {
                key: value.isoformat() if hasattr(value, "isoformat") else value
                for key, value in container.items()
            }
        message_dict["container"] = container

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
            if (
                isinstance(anthropic_pm, dict)
                and anthropic_pm.get("type") == "redacted_thinking"
                and anthropic_pm.get("data")
            ):
                # Safety-redacted reasoning: the opaque data must be
                # replayed byte-for-byte as a redacted_thinking block.
                return {"type": "redacted_thinking", "data": anthropic_pm["data"]}
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
            mcp_server_name = (
                anthropic_pm.get("mcp_server_name")
                if isinstance(anthropic_pm, dict)
                else None
            )
            if part.server_executed and (
                mcp_server_name or (part.tool_call_id or "").startswith("mcptoolu")
            ):
                # MCP connector calls replay as mcp_tool_use blocks; the
                # API requires the server_name field to be echoed back.
                block = {
                    "type": "mcp_tool_use",
                    "id": part.tool_call_id,
                    "name": part.name,
                    "input": part.arguments,
                }
                if mcp_server_name:
                    block["server_name"] = mcp_server_name
                return block
            return {
                "type": "server_tool_use" if part.server_executed else "tool_use",
                "id": part.tool_call_id,
                "name": part.name,
                "input": part.arguments,
            }
        if isinstance(part, ToolResultPart):
            if part.server_executed:
                # Reconstruct the provider result block that arrived in the
                # assistant turn (e.g. web_fetch_tool_result) - the API
                # rejects plain tool_result blocks in assistant messages.
                try:
                    content = json.loads(part.output) if part.output else None
                except ValueError:
                    content = part.output
                block = {
                    "type": part.name + "_tool_result",
                    "tool_use_id": part.tool_call_id,
                }
                if content is not None:
                    block["content"] = content
                return block
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
                if seen_tool_use and block_type == "text" and block.get("text") == " ":
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

    def _append_system_message(
        self, out: List[Dict[str, Any]], message: Message, is_first: bool
    ) -> None:
        """Handle a system-role message from an explicit messages= chain.

        The first message in the chain is hoisted to the top-level
        kwargs["system"] field by _extract_system. Later ones become inline
        role="system" entries on models that support mid-conversation
        system messages (Opus 4.8 and the Claude 5 family)."""
        if is_first:
            return
        if not self.supports_system_messages:
            raise ValueError(
                f"{self.claude_model_id} does not support mid-conversation "
                "system messages - only the first message in messages= can "
                "use role='system' with this model"
            )
        blocks = [
            {"type": "text", "text": part.text}
            for part in message.parts
            if isinstance(part, TextPart)
        ]
        if not blocks:
            return
        if out and out[-1]["role"] == "system":
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": "system", "content": blocks})

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
        for index, message in enumerate(prompt.messages):
            if message.role == "system":
                self._append_system_message(messages, message, index == 0)
            else:
                self._append_message(messages, message)

        # The API requires an inline system entry to immediately follow a
        # user turn (and precede an assistant turn or end the array), but
        # authors naturally write new instructions before their next user
        # message - so bubble each system entry past a directly following
        # user turn.
        for i in range(len(messages) - 1):
            if messages[i]["role"] == "system" and messages[i + 1]["role"] == "user":
                messages[i], messages[i + 1] = messages[i + 1], messages[i]

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
        if messages= was passed explicitly and it *starts* with a system-role
        message, fall back to reading that. Later system-role messages are
        sent inline by _append_system_message instead, since the API forbids
        a system entry in first position.
        """
        if prompt.system:
            return prompt.system
        if prompt.messages and prompt.messages[0].role == "system":
            texts = [
                p.text
                for p in prompt.messages[0].parts
                if isinstance(p, TextPart)
            ]
            if texts:
                return "\n\n".join(texts)
        return None

    def build_kwargs(self, prompt, conversation):
        if prompt.schema and prompt.tools:
            raise ValueError(
                "llm-anthropic does not yet support using both schema and tools in the same prompt"
            )

        kwargs = {
            "model": self.claude_model_id,
            "messages": self.build_messages(prompt, conversation),
        }
        if prompt.options.user_id:
            kwargs["metadata"] = {"user_id": prompt.options.user_id}

        # anthropic>=1 removed temperature/top_p/top_k from the method
        # signatures; the API still accepts them, so send via extra_body
        extra_body = {}
        if prompt.options.top_p is not None:
            extra_body["top_p"] = prompt.options.top_p
        else:
            extra_body["temperature"] = (
                prompt.options.temperature
                if prompt.options.temperature is not None
                else DEFAULT_TEMPERATURE
            )

        if prompt.options.top_k:
            extra_body["top_k"] = prompt.options.top_k

        if extra_body:
            kwargs["extra_body"] = extra_body

        system = self._extract_system(prompt)
        if system:
            kwargs["system"] = system

        if prompt.options.stop_sequences:
            kwargs["stop_sequences"] = prompt.options.stop_sequences

        thinking_effort_enabled = (
            self.supports_thinking_effort and prompt.options.thinking_effort
        )

        # Thinking: Claude 5 models think by default (adaptive mode);
        # older models only think when it is explicitly requested.
        if self.supports_thinking:
            hide_reasoning = getattr(prompt, "hide_reasoning", False)
            if prompt.options.thinking is False:
                if self.always_thinks:
                    raise ValueError(
                        f"Thinking cannot be disabled for model {self.model_id}"
                    )
                kwargs["thinking"] = {"type": "disabled"}
            elif prompt.options.thinking or thinking_effort_enabled:
                if self.supports_adaptive_thinking or thinking_effort_enabled:
                    kwargs["thinking"] = {"type": "adaptive"}
                else:
                    # Pre-4.6 models: enabled with default budget
                    kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": DEFAULT_THINKING_TOKENS,
                    }
            elif self.thinks_by_default and hide_reasoning:
                # No thinking option set, but the model will think anyway -
                # send the param explicitly so display can be omitted below
                kwargs["thinking"] = {"type": "adaptive"}

            if (
                hide_reasoning
                and "thinking" in kwargs
                and kwargs["thinking"]["type"] != "disabled"
            ):
                # -R / hide_reasoning=True asks the API to leave the
                # thinking trace out of the response entirely
                kwargs["thinking"]["display"] = "omitted"

        # Handle effort in output_config
        if thinking_effort_enabled:
            kwargs.setdefault("output_config", {})[
                "effort"
            ] = prompt.options.thinking_effort.value

        max_tokens = self.default_max_tokens
        if prompt.options.max_tokens is not None:
            max_tokens = prompt.options.max_tokens
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
                kwargs.setdefault("extra_body", {})["thinking"] = kwargs.pop("thinking")

        # Check if we should use new structured outputs
        use_structured_outputs = prompt.schema and self.use_structured_outputs

        if use_structured_outputs:
            kwargs.setdefault("output_config", {})["format"] = {
                "type": "json_schema",
                "schema": transform_schema(prompt.schema),
            }

        # Fast mode for lower latency responses
        if prompt.options.fast:
            kwargs["speed"] = "fast"
            betas.append("fast-mode-2026-02-01")

        if betas:
            kwargs["betas"] = betas

        tools = []

        if prompt.schema and not use_structured_outputs:
            # Fall back to tools workaround for models that don't support structured outputs
            tools.append(
                {
                    "name": "output_structured_data",
                    "input_schema": prompt.schema,
                }
            )
            kwargs["tool_choice"] = {"type": "tool", "name": "output_structured_data"}

        server_side_tools = []
        if prompt.tools:
            function_tools, server_side_tools = _partition_tools(self, prompt.tools)
            tools.extend(tool.tool_spec(self) for tool in server_side_tools)
            tools.extend(
                [
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.input_schema,
                    }
                    for tool in function_tools
                ]
            )

        if tools:
            kwargs["tools"] = tools

        for tool in server_side_tools:
            tool.prepare_request(self, kwargs)

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

        # Always use Anthropic's streaming transport, even when LLM has been
        # asked to buffer the response for non-streaming presentation. The
        # Anthropic SDK rejects non-streaming requests with large max_tokens
        # values because they may take longer than ten minutes.
        with messages_client.stream(**kwargs) as stream_obj:
            current_block_id = None
            current_block_name = None
            is_server_tool = False
            container = None

            if prefill_text:
                # part_index 0 is reserved for the prefill so it can
                # never collide with a content block's explicit index.
                yield StreamEvent(type="text", chunk=prefill_text, part_index=0)

            for chunk in stream_obj:
                if chunk.type == "content_block_start":
                    block = chunk.content_block
                    block_type = getattr(block, "type", None)
                    block_part_index = self._block_part_index(chunk)
                    current_block_id = getattr(block, "id", None)
                    current_block_name = getattr(block, "name", None)
                    is_server_tool = block_type in (
                        "server_tool_use",
                        "mcp_tool_use",
                    ) or (block_type or "").endswith("_tool_result")

                    if block_type == "redacted_thinking":
                        # The opaque block arrives complete on
                        # content_block_start; preserve it as a
                        # metadata-only reasoning part so it replays
                        # unchanged in continuation requests.
                        yield StreamEvent(
                            type="reasoning",
                            chunk="",
                            part_index=block_part_index,
                            provider_metadata={
                                "anthropic": {
                                    "type": "redacted_thinking",
                                    "data": getattr(block, "data", ""),
                                }
                            },
                        )
                    elif block_type in (
                        "tool_use",
                        "server_tool_use",
                        "mcp_tool_use",
                    ):
                        yield StreamEvent(
                            type="tool_call_name",
                            chunk=current_block_name or "",
                            part_index=block_part_index,
                            tool_call_id=current_block_id,
                            server_executed=(block_type != "tool_use"),
                            provider_metadata=(
                                {
                                    "anthropic": {
                                        "mcp_server_name": getattr(
                                            block, "server_name", None
                                        )
                                    }
                                }
                                if block_type == "mcp_tool_use"
                                else None
                            ),
                        )
                    elif block_type and block_type.endswith("_tool_result"):
                        # Content is available inline on content_block_start
                        event = self._server_tool_result_event(block_type, block)
                        event.part_index = block_part_index
                        yield event

                elif chunk.type == "content_block_delta":
                    delta = chunk.delta
                    delta_type = getattr(delta, "type", None)
                    block_part_index = self._block_part_index(chunk)

                    if delta_type == "thinking_delta":
                        yield StreamEvent(
                            type="reasoning",
                            chunk=delta.thinking,
                            part_index=block_part_index,
                        )
                    elif delta_type == "signature_delta":
                        yield StreamEvent(
                            type="reasoning",
                            chunk="",
                            part_index=block_part_index,
                            provider_metadata={
                                "anthropic": {"signature": delta.signature}
                            },
                        )
                    elif delta_type == "text_delta":
                        yield StreamEvent(
                            type="text",
                            chunk=delta.text,
                            part_index=block_part_index,
                        )
                    elif delta_type == "input_json_delta":
                        yield StreamEvent(
                            type="tool_call_args",
                            chunk=delta.partial_json,
                            part_index=block_part_index,
                            tool_call_id=current_block_id,
                            server_executed=is_server_tool,
                        )

                elif chunk.type == "message_delta":
                    chunk_container = getattr(chunk, "container", None) or getattr(
                        getattr(chunk, "delta", None), "container", None
                    )
                    if chunk_container is not None:
                        container = chunk_container

            # This records usage and other data:
            last_message = self._model_dump_suppress_warnings(
                stream_obj.get_final_message()
            )
            self._apply_container(last_message, container)
            response.response_json = last_message

            if self.add_tool_usage(response, last_message):
                # Avoid "can have dragons.Now that I " bug
                yield StreamEvent(type="text", chunk=" ")
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

        # Always use Anthropic's streaming transport. LLM still controls
        # whether the yielded events are displayed incrementally or buffered.
        async with messages_client.stream(**kwargs) as stream_obj:
            current_block_id = None
            current_block_name = None
            is_server_tool = False
            container = None

            if prefill_text:
                # part_index 0 is reserved for the prefill so it can
                # never collide with a content block's explicit index.
                yield StreamEvent(type="text", chunk=prefill_text, part_index=0)

            async for chunk in stream_obj:
                if chunk.type == "content_block_start":
                    block = chunk.content_block
                    block_type = getattr(block, "type", None)
                    block_part_index = self._block_part_index(chunk)
                    current_block_id = getattr(block, "id", None)
                    current_block_name = getattr(block, "name", None)
                    is_server_tool = block_type in (
                        "server_tool_use",
                        "mcp_tool_use",
                    ) or (block_type or "").endswith("_tool_result")

                    if block_type == "redacted_thinking":
                        # The opaque block arrives complete on
                        # content_block_start; preserve it as a
                        # metadata-only reasoning part so it replays
                        # unchanged in continuation requests.
                        yield StreamEvent(
                            type="reasoning",
                            chunk="",
                            part_index=block_part_index,
                            provider_metadata={
                                "anthropic": {
                                    "type": "redacted_thinking",
                                    "data": getattr(block, "data", ""),
                                }
                            },
                        )
                    elif block_type in (
                        "tool_use",
                        "server_tool_use",
                        "mcp_tool_use",
                    ):
                        yield StreamEvent(
                            type="tool_call_name",
                            chunk=current_block_name or "",
                            part_index=block_part_index,
                            tool_call_id=current_block_id,
                            server_executed=(block_type != "tool_use"),
                            provider_metadata=(
                                {
                                    "anthropic": {
                                        "mcp_server_name": getattr(
                                            block, "server_name", None
                                        )
                                    }
                                }
                                if block_type == "mcp_tool_use"
                                else None
                            ),
                        )
                    elif block_type and block_type.endswith("_tool_result"):
                        event = self._server_tool_result_event(block_type, block)
                        event.part_index = block_part_index
                        yield event

                elif chunk.type == "content_block_delta":
                    delta = chunk.delta
                    delta_type = getattr(delta, "type", None)
                    block_part_index = self._block_part_index(chunk)

                    if delta_type == "thinking_delta":
                        yield StreamEvent(
                            type="reasoning",
                            chunk=delta.thinking,
                            part_index=block_part_index,
                        )
                    elif delta_type == "signature_delta":
                        yield StreamEvent(
                            type="reasoning",
                            chunk="",
                            part_index=block_part_index,
                            provider_metadata={
                                "anthropic": {"signature": delta.signature}
                            },
                        )
                    elif delta_type == "text_delta":
                        yield StreamEvent(
                            type="text",
                            chunk=delta.text,
                            part_index=block_part_index,
                        )
                    elif delta_type == "input_json_delta":
                        yield StreamEvent(
                            type="tool_call_args",
                            chunk=delta.partial_json,
                            part_index=block_part_index,
                            tool_call_id=current_block_id,
                            server_executed=is_server_tool,
                        )

                elif chunk.type == "message_delta":
                    chunk_container = getattr(chunk, "container", None) or getattr(
                        getattr(chunk, "delta", None), "container", None
                    )
                    if chunk_container is not None:
                        container = chunk_container

            # This records usage and other data:
            last_message = self._model_dump_suppress_warnings(
                await stream_obj.get_final_message()
            )
            self._apply_container(last_message, container)
            response.response_json = last_message

            if self.add_tool_usage(response, last_message):
                # Avoid "can have dragons.Now that I " bug
                yield StreamEvent(type="text", chunk=" ")
        self.set_usage(response)
