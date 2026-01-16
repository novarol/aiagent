import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description="Gemini Agent")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    user_prompt = args.user_prompt
    is_verbose = args.verbose

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("API key not found! Make sure GEMINI_API_KEY is set as an environment variable!")

    client = genai.Client(api_key=api_key)

    messages = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages
    )

    if not response.usage_metadata:
        raise RuntimeError("Gemini failed to respond. Try again.")

    if is_verbose:
        print(f"User prompt: {user_prompt}")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")

    print("Response:")
    print(response.text)


if __name__ == "__main__":
    main()
