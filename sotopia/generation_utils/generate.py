import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import gin
from beartype import beartype
from litellm import acompletion
from rich import print
from rich.logging import RichHandler

from sotopia.database import EnvironmentProfile, RelationshipProfile
from sotopia.messages import ActionType, AgentAction, ScriptBackground
from sotopia.messages.message_classes import (
    ScriptInteraction,
    ScriptInteractionReturnType,
)
from sotopia.utils import format_docstring

from sotopia.generation_utils.output_parsers import (
    EnvResponse,
    OutputParser,
    OutputType,
    PydanticOutputParser,
    ScriptOutputParser,
    StrOutputParser,
)


def clean_json_response(text: str) -> str:
    """
    Strip markdown code blocks from JSON response.
    
    Some LLMs wrap JSON output in ```json ... ``` blocks.
    This function removes such wrappers to get clean JSON.
    """
    if not text:
        return text
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ``` wrapper
    pattern = r'^```(?:json)?\s*\n?(.*?)\n?```$'
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

# Configure logger
log = logging.getLogger("sotopia.generation")
log.setLevel(logging.INFO)

# Create console handler with rich formatting
console_handler = RichHandler(rich_tracebacks=True)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)

# Add handler to logger
log.addHandler(console_handler)

# subject to future OpenAI changes
DEFAULT_BAD_OUTPUT_PROCESS_MODEL = "gpt-4o-mini"

# ===================== LLM Call Logging =====================
# Global settings for LLM call logging
_llm_call_log_enabled: bool = False
_llm_call_log_file: Path | None = None
_llm_call_experiment_tag: str = ""


def enable_llm_call_logging(
    log_file: str | Path,
    experiment_tag: str = "",
) -> None:
    """
    Enable LLM call logging to track generation IDs and usage.

    Args:
        log_file: Path to the JSONL log file
        experiment_tag: Optional tag to identify the experiment
    """
    global _llm_call_log_enabled, _llm_call_log_file, _llm_call_experiment_tag
    _llm_call_log_enabled = True
    _llm_call_log_file = Path(log_file)
    _llm_call_experiment_tag = experiment_tag

    # Create parent directory if needed
    _llm_call_log_file.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"LLM call logging enabled: {_llm_call_log_file}")


def disable_llm_call_logging() -> None:
    """Disable LLM call logging."""
    global _llm_call_log_enabled, _llm_call_log_file, _llm_call_experiment_tag
    _llm_call_log_enabled = False
    _llm_call_log_file = None
    _llm_call_experiment_tag = ""
    log.info("LLM call logging disabled")


def _log_llm_call(response: Any, model_name: str, caller: str = "") -> None:
    """
    Log a single LLM call to the log file.

    Args:
        response: The LiteLLM response object
        model_name: The model name used
        caller: Optional caller function name for context
    """
    if not _llm_call_log_enabled or _llm_call_log_file is None:
        return

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment_tag": _llm_call_experiment_tag,
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", model_name),
        "caller": caller,
        "usage": {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
            "completion_tokens": getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
            "total_tokens": getattr(response.usage, "total_tokens", 0) if response.usage else 0,
        },
    }

    with open(_llm_call_log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def get_llm_call_log_path() -> Path | None:
    """Get the current LLM call log file path."""
    return _llm_call_log_file


# ============================================================


@beartype
async def format_bad_output(
    ill_formed_output: str,
    format_instructions: str,
    model_name: str,
    use_fixed_model_version: bool = True,
) -> str:
    template = """
    Given the string that can not be parsed by json parser, reformat it to a string that can be parsed by json parser.
    Original string: {ill_formed_output}

    Format instructions: {format_instructions}

    Please only generate the pure JSON without any additional text.  
    IMPORTANT:Do not Reapet the format instructions. Output ONLY the JSON object. Do not include any other text, markdown formatting.
    """

    input_values = {
        "ill_formed_output": ill_formed_output,
        "format_instructions": format_instructions,
    }
    content = template.format(**input_values)
    response = await acompletion(
        model=model_name,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": content}],
    )
    # Log LLM call for cost tracking
    _log_llm_call(response, model_name, caller="format_bad_output")

    reformatted_output = response.choices[0].message.content
    assert isinstance(reformatted_output, str)
    # Clean markdown code blocks if present
    reformatted_output = clean_json_response(reformatted_output)
    log.warning(f"ill_formed_output: {ill_formed_output} \n LLM Reformated output: {reformatted_output}")
    return reformatted_output


@gin.configurable
@beartype
async def agenerate(
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: OutputParser[OutputType],
    temperature: float = 0.7,
    max_tokens: int | None = 4096,
    structured_output: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> OutputType:
    """Generate text using LiteLLM instead of Langchain."""
    # Format template with input values
    if "format_instructions" not in input_values:
        input_values["format_instructions"] = output_parser.get_format_instructions()

    # Process template
    template = format_docstring(template)

    # Replace template variables
    for key, value in input_values.items():
        template = template.replace(f"{{{key}}}", str(value))

    if model_name.startswith("custom"):
        base_url, api_key = (
            model_name.split("@")[1],
            os.environ.get("CUSTOM_API_KEY", "EMPTY"),
        )
        model_name = model_name.split("@")[0].replace("custom/", "openai/")
    else:
        base_url = None
        api_key = None

    # Build common kwargs for acompletion
    completion_kwargs: dict = {
        "model": model_name,
        "temperature": temperature,
        "drop_params": True,
    }
    if max_tokens is not None:
        completion_kwargs["max_tokens"] = max_tokens
    if base_url is not None:
        completion_kwargs["base_url"] = base_url
        completion_kwargs["api_key"] = api_key

    # Fix provider for deepseek-v3.2 on OpenRouter to use siliconflow/fp8
    if "openrouter" in model_name.lower() and "deepseek" in model_name.lower():
        completion_kwargs["extra_body"] = {
            "provider": {
                "order": ["siliconflow"],
                "quantizations": ["fp8"],
                'preferred_min_throughput': {
                    'p90': 100, # Prefer providers with >100 tokens/sec for 90% of requests in last 5 minutes
                },
            }
        }
        log.debug(f"Fixed provider to SiliconFlow/fp8 for model: {model_name}")

    if structured_output:
        assert (
            model_name.startswith("gpt-4o")
            or model_name.startswith("openai/")
            or model_name.startswith("o1")
        ), "Structured output is only supported in limited models"
        messages = [{"role": "user", "content": template}]

        assert isinstance(
            output_parser, PydanticOutputParser
        ), "structured output only supported in PydanticOutputParser"
        response = await acompletion(
            messages=messages,
            response_format=output_parser.pydantic_object,
            **completion_kwargs,
        )
        # Log LLM call for cost tracking
        _log_llm_call(response, model_name, caller="agenerate_structured")

        result = response.choices[0].message.content
        log.info(f"Generated result: {result}")
        assert isinstance(result, str)
        # Clean markdown code blocks if present
        result = clean_json_response(result)
        return cast(OutputType, output_parser.parse(result))

    messages = [{"role": "user", "content": template}]
    # log.info(f"Generated messages to Model: {messages}")
    # For non-structured, use api_base instead of base_url
    if "base_url" in completion_kwargs:
        completion_kwargs["api_base"] = completion_kwargs.pop("base_url")

    response = await acompletion(
        messages=messages,
        **completion_kwargs,
    )
    # Log LLM call for cost tracking
    _log_llm_call(response, model_name, caller="agenerate")

    result = response.choices[0].message.content
    # Clean markdown code blocks if present
    result = clean_json_response(result)

    try:
        parsed_result = output_parser.parse(result)
    except Exception as e:
        if isinstance(output_parser, ScriptOutputParser):
            raise e
        log.debug(
            f"[red] Failed to parse result: {result}\nEncounter Exception {e}\nstart to reparse",
            extra={"markup": True},
        )
        # Handle bad output reformatting
        reformat_result = await format_bad_output(
            result,
            output_parser.get_format_instructions(),
            bad_output_process_model or model_name,
            use_fixed_model_version,
        )
        parsed_result = output_parser.parse(reformat_result)

    log.debug(f"Generated result: {parsed_result}")
    return parsed_result


@gin.configurable
@beartype
async def agenerate_env_profile(
    model_name: str,
    inspiration_prompt: str = "asking my boyfriend to stop being friends with his ex",
    examples: str = "",
    temperature: float = 0.7,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> EnvironmentProfile:
    """
    Using langchain to generate the background
    """
    return await agenerate(
        model_name=model_name,
        template="""Please generate scenarios and goals based on the examples below as well as the inspirational prompt, when creating the goals, try to find one point that both sides may not agree upon initially and need to collaboratively resolve it.
        Examples:
        {examples}
        Inspirational prompt: {inspiration_prompt}
        Please use the following format:
        {format_instructions}
        """,
        input_values=dict(
            inspiration_prompt=inspiration_prompt,
            examples=examples,
        ),
        output_parser=PydanticOutputParser(pydantic_object=EnvironmentProfile),
        temperature=temperature,
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


@beartype
async def agenerate_relationship_profile(
    model_name: str,
    agents_profiles: list[str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> tuple[RelationshipProfile, str]:
    """
    Using langchain to generate the background
    """
    agent_profile = "\n".join(agents_profiles)
    return await agenerate(
        model_name=model_name,
        template="""Please generate relationship between two agents based on the agents' profiles below. Note that you generate
        {agent_profile}
        Please use the following format:
        {format_instructions}
        """,
        input_values=dict(
            agent_profile=agent_profile,
        ),
        output_parser=PydanticOutputParser(pydantic_object=RelationshipProfile),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


@gin.configurable
@beartype
async def agenerate_action(
    model_name: str,
    history: str,
    turn_number: int,
    action_types: list[ActionType],
    agent: str,
    goal: str,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    script_like: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> AgentAction:
    """
    Using langchain to generate an example episode.

    Args:
        max_retries: Maximum number of retries on API errors (default: 10)
        retry_delay: Base delay between retries in seconds, uses exponential backoff (default: 1.0)
    """
    import asyncio
    from litellm import APIError, Timeout

    # Build template based on script_like mode
    if script_like:
        # model as playwright
        template = """
            Now you are a famous playwright, your task is to continue writing one turn for agent {agent} under a given background and history to help {agent} reach social goal. Please continue the script based on the previous turns. You can only generate one turn at a time.
            You can find {agent}'s background and goal in the 'Here is the context of the interaction' field.
            You should try your best to achieve {agent}'s goal in a way that align with their character traits.
            Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
            {history}.
            The script has proceeded to Turn #{turn_number}. Current available action types are
            {action_list}.
            Note: The script can be ended if 1. one agent have achieved social goals, 2. this conversation makes the agent uncomfortable, 3. the agent find it uninteresting/you lose your patience, 4. or for other reasons you think it should stop.
            IMPORTANT:Do not Reapet the format instructions. Output ONLY the JSON object. Do not include any other text, markdown formatting.
            Please only generate a pure JSON string including the action type and the argument without any additional text.
            Your action should follow the given format:
            {format_instructions}
        """
    else:
        # Normal case, model as agent
        template = """
            Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
            You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
            Note that {agent}'s goal is only visible to you.
            You should try your best to achieve {agent}'s goal in a way that align with their character traits.
            Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
            {history}.
            You are at Turn #{turn_number}. Your available action types are
            {action_list}.
            Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

            Please only generate a pure JSON string including the action type and the argument without any additional text.

            IMPORTANT:Do not Reapet the format instructions. Output ONLY the JSON object. Do not include any other text, markdown formatting.

            Your action should follow the given format:
            {format_instructions}
        """

    last_exception: Exception | None = None

    for attempt in range(max_retries):
        try:
            return await agenerate(
                model_name=model_name,
                template=template,
                input_values=dict(
                    agent=agent,
                    turn_number=str(turn_number),
                    history=history,
                    action_list=" ".join(action_types),
                ),
                output_parser=PydanticOutputParser(pydantic_object=AgentAction),
                temperature=temperature,
                max_tokens=max_tokens,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
            )
        except (APIError, Timeout) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                log.warning(
                    f"[Attempt {attempt + 1}/{max_retries}] API error in agenerate_action: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                log.error(
                    f"[Attempt {attempt + 1}/{max_retries}] API error in agenerate_action: {e}. "
                    f"All retries exhausted."
                )
        except Exception as e:
            # For non-API errors (e.g., parsing errors), fail immediately
            log.error(f"Failed to generate action due to non-retryable error: {e}")
            raise

    # All retries exhausted - raise the last exception instead of returning "none"
    assert last_exception is not None
    raise last_exception


@gin.configurable
@beartype
async def agenerate_script(
    model_name: str,
    background: ScriptBackground,
    temperature: float = 0.7,
    agent_names: list[str] = [],
    agent_name: str = "",
    history: str = "",
    single_step: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> tuple[ScriptInteractionReturnType, str]:
    """
    Using langchain to generate an the script interactions between two agent
    The script interaction is generated in a single generation process.
    Note that in this case we do not require a json format response,
    so the failure rate will be higher, and it is recommended to use at least llama-2-70b.
    """
    try:
        if single_step:
            return await agenerate(
                model_name=model_name,
                template="""Now you are a famous playwright, your task is to continue writing one turn for agent {agent} under a given background and history to help {agent} reach social goal. Please continue the script based on the previous turns. You can only generate one turn at a time.

                Here are the conversation background and history:
                {background}
                {history}

                Remember that you are an independent scriptwriter and should finish the script by yourself.
                The output should only contain the script following the format instructions, with no additional comments or text.

                Here are the format instructions:
                {format_instructions}""",
                input_values=dict(
                    background=background.to_natural_language(),
                    history=history,
                    agent=agent_name,
                ),
                output_parser=ScriptOutputParser(  # type: ignore[arg-type]
                    agent_names=agent_names,
                    background=background.to_natural_language(),
                    single_turn=True,
                ),
                temperature=temperature,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
            )

        else:
            return await agenerate(
                model_name=model_name,
                template="""
                Please write the script between two characters based on their social goals with a maximum of 20 turns.

                {background}
                Your action should follow the given format:
                {format_instructions}
                Remember that you are an independent scriptwriter and should finish the script by yourself.
                The output should only contain the script following the format instructions, with no additional comments or text.""",
                input_values=dict(
                    background=background.to_natural_language(),
                ),
                output_parser=ScriptOutputParser(  # type: ignore[arg-type]
                    agent_names=agent_names,
                    background=background.to_natural_language(),
                    single_turn=False,
                ),
                temperature=temperature,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
            )
    except Exception as e:
        # TODO raise(e) # Maybe we do not want to return anything?
        print(f"Exception in agenerate {e}")
        return_default_value: ScriptInteractionReturnType = (
            ScriptInteraction.default_value_for_return_type()
        )
        return (return_default_value, "")


@beartype
def process_history(
    script: ScriptBackground | EnvResponse | dict[str, AgentAction],
) -> str:
    """
    Format the script background
    """
    result = ""
    if isinstance(script, ScriptBackground | EnvResponse):
        script = script.dict()
        result = "The initial observation\n\n"
    for key, value in script.items():
        if value:
            result += f"{key}: {value} \n"
    return result


@beartype
async def agenerate_init_profile(
    model_name: str,
    basic_info: dict[str, str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    """
    Using langchain to generate the background
    """
    return await agenerate(
        model_name=model_name,
        template="""Please expand a fictional background for {name}. Here is the basic information:
            {name}'s age: {age}
            {name}'s gender identity: {gender_identity}
            {name}'s pronouns: {pronoun}
            {name}'s occupation: {occupation}
            {name}'s big 5 personality traits: {bigfive}
            {name}'s moral Foundation: think {mft} is more important than others
            {name}'s Schwartz portrait value: {schwartz}
            {name}'s decision-making style: {decision_style}
            {name}'s secret: {secret}
            Include the previous information in the background.
            Then expand the personal backgrounds with concrete details (e.g, look, family, hobbies, friends and etc.)
            For the personality and values (e.g., MBTI, moral foundation, and etc.),
            remember to use examples and behaviors in the person's life to demonstrate it.
            """,
        input_values=dict(
            name=basic_info["name"],
            age=basic_info["age"],
            gender_identity=basic_info["gender_identity"],
            pronoun=basic_info["pronoun"],
            occupation=basic_info["occupation"],
            bigfive=basic_info["Big_Five_Personality"],
            mft=basic_info["Moral_Foundation"],
            schwartz=basic_info["Schwartz_Portrait_Value"],
            decision_style=basic_info["Decision_making_Style"],
            secret=basic_info["secret"],
        ),
        output_parser=StrOutputParser(),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


@beartype
async def convert_narratives(
    model_name: str,
    narrative: str,
    text: str,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    if narrative == "first":
        return await agenerate(
            model_name=model_name,
            template="""Please convert the following text into a first-person narrative.
            e.g, replace name, he, she, him, her, his, and hers with I, me, my, and mine.
            {text}""",
            input_values=dict(text=text),
            output_parser=StrOutputParser(),
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
        )
    elif narrative == "second":
        return await agenerate(
            model_name=model_name,
            template="""Please convert the following text into a second-person narrative.
            e.g, replace name, he, she, him, her, his, and hers with you, your, and yours.
            {text}""",
            input_values=dict(text=text),
            output_parser=StrOutputParser(),
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
        )
    else:
        raise ValueError(f"Narrative {narrative} is not supported.")


@beartype
async def agenerate_goal(
    model_name: str,
    background: str,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    """
    Using langchain to generate the background
    """
    return await agenerate(
        model_name=model_name,
        template="""Please generate your goal based on the background:
            {background}
            """,
        input_values=dict(background=background),
        output_parser=StrOutputParser(),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )
