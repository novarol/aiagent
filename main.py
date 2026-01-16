import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from argparse import ArgumentParser

def main():
    user_prompt, is_verbose = get_cli_arguments()

    prompt_client()(user_prompt, is_verbose)


def get_cli_arguments() -> tuple[str, bool]:
    parser = ArgumentParser(description="Gemini Agent")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    return args.user_prompt, args.verbose


def get_environment_variable(name: str):
    load_dotenv()

    env_var = os.environ.get("GEMINI_API_KEY")

    if not env_var:
        raise RuntimeError(f"Environment variable not found! Make sure {name} is set!")
    
    return env_var


def create_client():
    api_key = get_environment_variable("GEMINI_API_KEY")

    return genai.Client(api_key=api_key)


def prompt_client():
    client = create_client()
    messages = []

    def add_content(role: str, text: str):
        messages.append(types.Content(role=role, parts=[types.Part(text=text)]))

    def generate_content(user_prompt: str, is_verbose: bool):
        if is_verbose:
            print(f"User prompt: {user_prompt}")

        add_content("user", user_prompt)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages
        )

        if not response.usage_metadata:
            raise RuntimeError("Gemini failed to respond. Try again.")

        if is_verbose:
            print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
            print(f"Response tokens: {response.usage_metadata.candidates_token_count}")

        print("Response:")
        print(response.text)

    return generate_content



if __name__ == "__main__":
    main()
