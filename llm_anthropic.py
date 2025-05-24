from anthropic import Anthropic, AsyncAnthropic
import llm
import json
from pydantic import Field, field_validator, model_validator
from typing import Optional, List, Union

DEFAULT_THINKING_TOKENS = 1024
DEFAULT_TEMPERATURE = 1.0


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
            "claude-3-5-sonnet-20241022", supports_pdf=True, default_max_tokens=8192
        ),
        AsyncClaudeMessages(
            "claude-3-5-sonnet-20241022", supports_pdf=True, default_max_tokens=8192
        ),
    )
    register(
        ClaudeMessages(
            "claude-3-5-sonnet-latest", supports_pdf=True, default_max_tokens=8192
        ),
        AsyncClaudeMessages(
            "claude-3-5-sonnet-latest", supports_pdf=True, default_max_tokens=8192
        ),
        aliases=("claude-3.5-sonnet", "claude-3.5-sonnet-latest"),
    )
    register(
        ClaudeMessages("claude-3-5-haiku-latest", default_max_tokens=8192),
        AsyncClaudeMessages("claude-3-5-haiku-latest", default_max_tokens=8192),
        aliases=("claude-3.5-haiku",),
    )
    # 3.7
    register(
        ClaudeMessages(
            "claude-3-7-sonnet-20250219",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-3-7-sonnet-20250219",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
    )
    register(
        ClaudeMessages(
            "claude-3-7-sonnet-latest",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-3-7-sonnet-latest",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-3.7-sonnet", "claude-3.7-sonnet-latest"),
    )
    register(
        ClaudeMessages(
            "claude-opus-4-0",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-opus-4-0",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-4-opus",),
    )
    register(
        ClaudeMessages(
            "claude-sonnet-4-0",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-sonnet-4-0",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-4-sonnet",),
    )


class ClaudeOptions(llm.Options):
    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate before stopping",
        default=None,
    )

    temperature: Optional[float] = Field(
        description="Amount of randomness injected into the response. Defaults to 1.0. Ranges from 0.0 to 1.0. Use temperature closer to 0.0 for analytical / multiple choice, and closer to 1.0 for creative and generative tasks. Note that even with temperature of 0.0, the results will not be fully deterministic.",
        default=None,
    )

    top_p: Optional[float] = Field(
        description="Use nucleus sampling. In nucleus sampling, we compute the cumulative distribution over all the options for each subsequent token in decreasing probability order and cut it off once it reaches a particular probability specified by top_p. You should either alter temperature or top_p, but not both. Recommended for advanced use cases only. You usually only need to use temperature.",
        default=None,
    )

    top_k: Optional[int] = Field(
        description="Only sample from the top K options for each subsequent token. Used to remove 'long tail' low probability responses. Recommended for advanced use cases only. You usually only need to use temperature.",
        default=None,
    )

    user_id: Optional[str] = Field(
        description="An external identifier for the user who is associated with the request",
        default=None,
    )

    prefill: Optional[str] = Field(
        description="A prefill to use for the response",
        default=None,
    )

    hide_prefill: Optional[bool] = Field(
        description="Do not repeat the prefill value at the start of the response",
        default=None,
    )

    stop_sequences: Optional[Union[list, str]] = Field(
        description=(
            "Custom text sequences that will cause the model to stop generating - "
            "pass either a list of strings or a single string"
        ),
        default=None,
    )
    cache: Optional[bool] = Field(
        description="Use Anthropic prompt cache for any attachments or fragments",
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

    @model_validator(mode="after")
    def validate_temperature_top_p(self):
        if self.temperature != 1.0 and self.top_p is not None:
            raise ValueError("Only one of temperature and top_p can be set")
        return self


class ClaudeOptionsWithThinking(ClaudeOptions):
    thinking: Optional[bool] = Field(
        description="Enable thinking mode",
        default=None,
    )
    thinking_budget: Optional[int] = Field(
        description="Number of tokens to budget for thinking", default=None
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

    supports_thinking = False
    supports_schema = True
    supports_tools = True
    default_max_tokens = 4096

    class Options(ClaudeOptions): ...

    def __init__(
        self,
        model_id,
        claude_model_id=None,
        supports_images=True,
        supports_pdf=False,
        supports_thinking=False,
        default_max_tokens=None,
    ):
        self.model_id = "anthropic/" + model_id
        self.claude_model_id = claude_model_id or model_id
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
        if default_max_tokens is not None:
            self.default_max_tokens = default_max_tokens

    def prefill_text(self, prompt):
        if prompt.options.prefill and not prompt.options.hide_prefill:
            return prompt.options.prefill
        return ""

    def build_messages(self, prompt, conversation) -> List[dict]:
        messages = []

        # Process existing conversation history
        if conversation:
            for response in conversation.responses:
                # Build user message
                user_content = []

                # Handle attachments in user message
                if response.attachments:
                    for attachment in response.attachments:
                        attachment_type = (
                            "document"
                            if attachment.resolve_type() == "application/pdf"
                            else "image"
                        )
                        user_content.append(
                            {
                                "type": attachment_type,
                                "source": source_for_attachment(attachment),
                            }
                        )

                # Add text content if it exists
                if response.prompt.prompt:
                    user_content.append(
                        {"type": "text", "text": response.prompt.prompt}
                    )

                # Handle tool results if present (add to the same user message)
                if response.prompt.tool_results:
                    for tool_result in response.prompt.tool_results:
                        user_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_result.tool_call_id,
                                "content": tool_result.output,
                            }
                        )

                # Only add the message if we have content
                if user_content:
                    messages.append({"role": "user", "content": user_content})

                # Build assistant message
                assistant_content = []
                text_content = response.text_or_raise()

                # Add text content if it exists
                if text_content:
                    assistant_content.append({"type": "text", "text": text_content})

                # Add tool calls if they exist
                tool_calls = response.tool_calls_or_raise()
                for tool_call in tool_calls:
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.tool_call_id,
                            "name": tool_call.name,
                            "input": tool_call.arguments,
                        }
                    )

                # Add assistant message if we have content
                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})

        # Handle current prompt's tool results and user content together
        user_content = []

        # Add attachments
        if prompt.attachments:
            for attachment in prompt.attachments:
                attachment_type = (
                    "document"
                    if attachment.resolve_type() == "application/pdf"
                    else "image"
                )
                user_content.append(
                    {
                        "type": attachment_type,
                        "source": source_for_attachment(attachment),
                    }
                )

            # Add cache control if needed
            if prompt.options.cache and user_content:
                user_content[-1]["cache_control"] = {"type": "ephemeral"}

        # Add current prompt's tool results
        if prompt.tool_results:
            for tool_result in prompt.tool_results:
                user_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_result.tool_call_id,
                        "content": tool_result.output,
                    }
                )

        # Add text content if it exists
        if prompt.prompt:
            user_content.append({"type": "text", "text": prompt.prompt})

        # Add the user message if we have content
        if user_content:
            messages.append({"role": "user", "content": user_content})

        # Handle cache control for messages without attachments
        elif prompt.options.cache and messages:
            last_message = messages[-1]
            if isinstance(last_message.get("content"), list):
                last_message["content"][-1]["cache_control"] = {"type": "ephemeral"}
            else:
                # This should not happen with our new structure, but keeping as a fallback
                last_message["cache_control"] = {"type": "ephemeral"}

        # Add prefill if specified
        if prompt.options.prefill:
            prefill_content = [{"type": "text", "text": prompt.options.prefill}]
            messages.append({"role": "assistant", "content": prefill_content})

        return messages

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

        if prompt.system:
            kwargs["system"] = prompt.system

        if prompt.options.stop_sequences:
            kwargs["stop_sequences"] = prompt.options.stop_sequences

        if self.supports_thinking and (
            prompt.options.thinking or prompt.options.thinking_budget
        ):
            prompt.options.thinking = True
            budget_tokens = prompt.options.thinking_budget or DEFAULT_THINKING_TOKENS
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}

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
        if max_tokens > 64000:
            kwargs["betas"] = ["output-128k-2025-02-19"]
            if "thinking" in kwargs:
                kwargs["extra_body"] = {"thinking": kwargs.pop("thinking")}

        tools = []
        if prompt.schema:
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
        # Only include usage details if prompt caching was on
        details = None
        if response.prompt.options.cache:
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
        client = Anthropic(api_key=self.get_key(key))
        kwargs = self.build_kwargs(prompt, conversation)
        prefill_text = self.prefill_text(prompt)
        if "betas" in kwargs:
            messages_client = client.beta.messages
        else:
            messages_client = client.messages
        if stream:
            with messages_client.stream(**kwargs) as stream:
                if prefill_text:
                    yield prefill_text
                for chunk in stream:
                    if hasattr(chunk, "delta"):
                        delta = chunk.delta
                        if hasattr(delta, "text"):
                            yield delta.text
                        elif hasattr(delta, "partial_json") and prompt.schema:
                            yield delta.partial_json
                # This records usage and other data:
                last_message = stream.get_final_message().model_dump()
                response.response_json = last_message
                if self.add_tool_usage(response, last_message):
                    # Avoid "can have dragons.Now that I " bug
                    yield " "
        else:
            completion = messages_client.create(**kwargs)
            text = "".join(
                [item.text for item in completion.content if hasattr(item, "text")]
            )
            yield prefill_text + text
            response.response_json = completion.model_dump()
            if self.add_tool_usage(response, response.response_json):
                yield " "
        self.set_usage(response)


class AsyncClaudeMessages(_Shared, llm.AsyncKeyModel):
    async def execute(self, prompt, stream, response, conversation, key):
        client = AsyncAnthropic(api_key=self.get_key(key))
        kwargs = self.build_kwargs(prompt, conversation)
        if "betas" in kwargs:
            messages_client = client.beta.messages
        else:
            messages_client = client.messages
        prefill_text = self.prefill_text(prompt)
        if stream:
            async with messages_client.stream(**kwargs) as stream_obj:
                if prefill_text:
                    yield prefill_text
                async for chunk in stream_obj:
                    if hasattr(chunk, "delta"):
                        delta = chunk.delta
                        if hasattr(delta, "text"):
                            yield delta.text
                        elif hasattr(delta, "partial_json") and prompt.schema:
                            yield delta.partial_json
            response.response_json = (await stream_obj.get_final_message()).model_dump()
            # TODO: Test this:
            self.add_tool_usage(response, response.response_json)
        else:
            completion = await messages_client.create(**kwargs)
            text = "".join(
                [item.text for item in completion.content if hasattr(item, "text")]
            )
            yield prefill_text + text
            response.response_json = completion.model_dump()
            # TODO: Test this:
            self.add_tool_usage(response, response.response_json)
        self.set_usage(response)
