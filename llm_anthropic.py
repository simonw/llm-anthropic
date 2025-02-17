from anthropic import Anthropic, AsyncAnthropic
import llm
import json
from pydantic import Field, field_validator, model_validator
from typing import Optional, List, Union


@llm.hookimpl
def register_models(register):
    # https://docs.anthropic.com/claude/docs/models-overview
    register(
        ClaudeMessages("claude-3-opus-20240229"),
        AsyncClaudeMessages("claude-3-opus-20240229"),
    ),
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
        ClaudeMessagesLong("claude-3-5-sonnet-20240620", supports_pdf=True),
        AsyncClaudeMessagesLong("claude-3-5-sonnet-20240620", supports_pdf=True),
    )
    register(
        ClaudeMessagesLong("claude-3-5-sonnet-20241022", supports_pdf=True),
        AsyncClaudeMessagesLong("claude-3-5-sonnet-20241022", supports_pdf=True),
    )
    register(
        ClaudeMessagesLong("claude-3-5-sonnet-latest", supports_pdf=True),
        AsyncClaudeMessagesLong("claude-3-5-sonnet-latest", supports_pdf=True),
        aliases=("claude-3.5-sonnet", "claude-3.5-sonnet-latest"),
    )
    register(
        ClaudeMessagesLong("claude-3-5-haiku-latest", supports_images=False),
        AsyncClaudeMessagesLong("claude-3-5-haiku-latest", supports_images=False),
        aliases=("claude-3.5-haiku",),
    )


class ClaudeOptions(llm.Options):
    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate before stopping",
        default=4_096,
    )

    temperature: Optional[float] = Field(
        description="Amount of randomness injected into the response. Defaults to 1.0. Ranges from 0.0 to 1.0. Use temperature closer to 0.0 for analytical / multiple choice, and closer to 1.0 for creative and generative tasks. Note that even with temperature of 0.0, the results will not be fully deterministic.",
        default=1.0,
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

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, max_tokens):
        real_max = cls.model_fields["max_tokens"].default
        if not (0 < max_tokens <= real_max):
            raise ValueError("max_tokens must be in range 1-{}".format(real_max))
        return max_tokens

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


long_field = Field(
    description="The maximum number of tokens to generate before stopping",
    default=4_096 * 2,
)


class _Shared:
    needs_key = "anthropic"
    key_env_var = "ANTHROPIC_API_KEY"
    can_stream = True

    class Options(ClaudeOptions): ...

    def __init__(
        self,
        model_id,
        claude_model_id=None,
        supports_images=True,
        supports_pdf=False,
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

    def prefill_text(self, prompt):
        if prompt.options.prefill and not prompt.options.hide_prefill:
            return prompt.options.prefill
        return ""

    def build_messages(self, prompt, conversation) -> List[dict]:
        messages = []
        if conversation:
            for response in conversation.responses:
                if response.attachments:
                    content = [
                        {
                            "type": (
                                "document"
                                if attachment.resolve_type() == "application/pdf"
                                else "image"
                            ),
                            "source": {
                                "data": attachment.base64_content(),
                                "media_type": attachment.resolve_type(),
                                "type": "base64",
                            },
                        }
                        for attachment in response.attachments
                    ]
                    content.append({"type": "text", "text": response.prompt.prompt})
                else:
                    content = response.prompt.prompt
                messages.extend(
                    [
                        {
                            "role": "user",
                            "content": content,
                        },
                        {"role": "assistant", "content": response.text()},
                    ]
                )
        if prompt.attachments:
            content = [
                {
                    "type": (
                        "document"
                        if attachment.resolve_type() == "application/pdf"
                        else "image"
                    ),
                    "source": {
                        "data": attachment.base64_content(),
                        "media_type": attachment.resolve_type(),
                        "type": "base64",
                    },
                }
                for attachment in prompt.attachments
            ]
            content.append({"type": "text", "text": prompt.prompt})
            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
        else:
            messages.append({"role": "user", "content": prompt.prompt})
        if prompt.options.prefill:
            messages.append({"role": "assistant", "content": prompt.options.prefill})
        return messages

    def build_kwargs(self, prompt, conversation):
        kwargs = {
            "model": self.claude_model_id,
            "messages": self.build_messages(prompt, conversation),
            "max_tokens": prompt.options.max_tokens,
        }
        if prompt.options.user_id:
            kwargs["metadata"] = {"user_id": prompt.options.user_id}

        if prompt.options.top_p:
            kwargs["top_p"] = prompt.options.top_p
        else:
            kwargs["temperature"] = prompt.options.temperature

        if prompt.options.top_k:
            kwargs["top_k"] = prompt.options.top_k

        if prompt.system:
            kwargs["system"] = prompt.system

        if prompt.options.stop_sequences:
            kwargs["stop_sequences"] = prompt.options.stop_sequences

        return kwargs

    def set_usage(self, response):
        usage = response.response_json.pop("usage")
        if usage:
            response.set_usage(
                input=usage.get("input_tokens"), output=usage.get("output_tokens")
            )

    def __str__(self):
        return "Anthropic Messages: {}".format(self.model_id)


class ClaudeMessages(_Shared, llm.KeyModel):

    def execute(self, prompt, stream, response, conversation, key):
        client = Anthropic(api_key=self.get_key(key))
        kwargs = self.build_kwargs(prompt, conversation)
        prefill_text = self.prefill_text(prompt)
        if stream:
            with client.messages.stream(**kwargs) as stream:
                if prefill_text:
                    yield prefill_text
                for text in stream.text_stream:
                    yield text
                # This records usage and other data:
                response.response_json = stream.get_final_message().model_dump()
        else:
            completion = client.messages.create(**kwargs)
            text = completion.content[0].text
            yield prefill_text + text
            response.response_json = completion.model_dump()
        self.set_usage(response)


class ClaudeMessagesLong(ClaudeMessages):
    class Options(ClaudeOptions):
        max_tokens: Optional[int] = long_field


class AsyncClaudeMessages(_Shared, llm.AsyncKeyModel):
    async def execute(self, prompt, stream, response, conversation, key):
        client = AsyncAnthropic(api_key=self.get_key(key))
        kwargs = self.build_kwargs(prompt, conversation)
        prefill_text = self.prefill_text(prompt)
        if stream:
            async with client.messages.stream(**kwargs) as stream_obj:
                if prefill_text:
                    yield prefill_text
                async for text in stream_obj.text_stream:
                    yield text
            response.response_json = (await stream_obj.get_final_message()).model_dump()
        else:
            completion = await client.messages.create(**kwargs)
            text = completion.content[0].text
            yield prefill_text + text
            response.response_json = completion.model_dump()
        self.set_usage(response)


class AsyncClaudeMessagesLong(AsyncClaudeMessages):
    class Options(ClaudeOptions):
        max_tokens: Optional[int] = long_field
