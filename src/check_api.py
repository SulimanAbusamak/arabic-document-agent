"""
One-off check that the API key works. Run this before anything else.

    python src/check_api.py

Costs a fraction of a cent.
"""

import anthropic

import config


def main() -> None:
    key = config.require_api_key()
    print(f"key found: {key[:12]}...{key[-4:]}")
    print(f"model    : {config.MODEL}")
    print("sending a test message...\n")

    client = anthropic.Anthropic(api_key=key)

    response = client.messages.create(
        model=config.MODEL,
        max_tokens=100,
        messages=[
            {"role": "user", "content": "Reply with exactly: connection ok"}
        ],
    )

    print("reply:", response.content[0].text)
    print(
        f"\ntokens -> in: {response.usage.input_tokens}  "
        f"out: {response.usage.output_tokens}"
    )
    print("\nIf you see 'connection ok' above, you are ready.")


if __name__ == "__main__":
    main()
