"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""


"""
# Verification
# Tasks 1 to 5 and Bonus A to C passed the provided pytest suite.
# Tasks 2 and 5 were also verified with the live Gemini API.
# Live verification for Tasks 1, 3, and 4 was blocked by insufficient API credits.
"""


import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI

    # Initialize the client using the API key from the environment.
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Start measuring the API request duration.
    start = time.perf_counter()

    # Send the prompt to the selected model.
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Calculate the elapsed time in seconds.
    latency = time.perf_counter() - start

    # Extract the response text.
    response_text = response.choices[0].message.content or ""

    # Extract input and output token counts.
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }

    return response_text, latency, usage

# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 3.5 (Flash Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from google import genai
    from google.genai import types

    # Initialize the client using the API key from the environment.
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Configure generation parameters.
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    # Start measuring the API request duration.
    start = time.perf_counter()

    # Send the prompt to the selected model.
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    # Calculate the elapsed time in seconds.
    latency = time.perf_counter() - start

    # Extract the response text.
    response_text = response.text or ""

    # Extract input and output token counts.
    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count or 0,
        "output_tokens": response.usage_metadata.candidates_token_count or 0,
    }

    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    import anthropic

    # Initialize the client using the API key from the environment.
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    # Start measuring the API request duration.
    start = time.perf_counter()

    # Send the prompt to the selected model.
    response = client.messages.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
    )

    # Calculate the elapsed time in seconds.
    latency = time.perf_counter() - start

    # Combine text from the response content blocks.
    response_text = "".join(
        block.text
        for block in response.content
        if hasattr(block, "text")
    )

    # Extract input and output token counts.
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    # Define the models and their corresponding API functions.
    gemini_model = os.getenv("LAB_GEMINI_COMPARE_MODEL", GEMINI_MODEL)
    models = [
        ("gpt4o", OPENAI_MODEL, call_openai),
        ("gpt4o_mini", OPENAI_MINI_MODEL, call_openai),
        ("gemini_flash", gemini_model, call_gemini),
    ]

    results = {}

    for result_key, model_name, call_fn in models:
        # Send the same prompt to each model.
        response_text, latency, usage = call_fn(
            prompt,
            model=model_name,
        )

        # Retrieve token prices from the lab pricing table.
        pricing = PRICING_1M_TOKENS[model_name]

        # Calculate the input and output token cost in USD.
        cost = (
            usage["input_tokens"] * pricing["input"]
            + usage["output_tokens"] * pricing["output"]
        ) / 1_000_000

        # Store the comparison metrics for this model.
        results[result_key] = {
            "response": response_text,
            "latency": latency,
            "cost": cost,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        }

    return results

# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    from google import genai
    from google.genai import types

    # Initialize the client using the API key from the environment.
    client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        timeout=20_000,
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)

    # Store completed user and model messages.
    history = []

    print("Gemini chatbot — type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        # End the session when requested.
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        # Skip empty messages.
        if not user_input:
            continue

        user_message = types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        )

        # Include recent conversation history and the current message.
        contents = history + [user_message]
        response_parts = []

        print("Gemini: ", end="", flush=True)

        try:
            # Display response chunks as they arrive.
            response_stream = client.models.generate_content_stream(
            model="gemini-3.5-flash-lite",
            contents=contents,
            )

            for chunk in response_stream:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response_parts.append(chunk.text)

            print()

        except Exception as exc:
            print(f"\nRequest failed: {exc}")
            continue

        # Save the completed conversation turn.
        response_text = "".join(response_parts)

        history.extend([
            user_message,
            types.Content(
                role="model",
                parts=[types.Part(text=response_text)],
            ),
        ])

        # Keep the last three completed turns: six messages.
        history = history[-6:]

# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    # Attempt the initial call followed by the allowed retries.
    for attempt in range(max_retries + 1):
        try:
            return fn()

        except Exception:
            # Raise the last exception when no retries remain.
            if attempt == max_retries:
                raise

            # Double the waiting time before each subsequent retry.
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)

# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    results = []

    for prompt in prompts:
        # Compare all models using the current prompt.
        comparison = compare_models(prompt)

        # Store the original prompt with its comparison results.
        results.append({
            **comparison,
            "prompt": prompt,
        })

    return results
# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    # Map result keys to readable model names.
    model_names = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }

    # Escape characters that could break Markdown table cells.
    def escape_cell(text: str) -> str:
        return (
            text.replace("\r\n", " ")
            .replace("\n", " ")
            .replace("\r", " ")
            .replace("|", "\\|")
        )

    rows = [
        "| Prompt | Model | Response (truncated) | Latency | "
        "Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for result in results:
        prompt = escape_cell(result["prompt"])

        for model_key, model_name in model_names.items():
            stats = result[model_key]

            # Limit the response preview to 50 characters.
            response = stats["response"]
            if len(response) > 50:
                response = response[:47] + "..."

            response = escape_cell(response)

            # Format one table row for each model.
            rows.append(
                f"| {prompt} "
                f"| {model_name} "
                f"| {response} "
                f"| {stats['latency']:.3f}s "
                f"| {stats['input_tokens']} / {stats['output_tokens']} "
                f"| ${stats['cost']:.8f} |"
            )

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
