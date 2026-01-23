import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from argparse import ArgumentParser
from prompts import system_prompt
from call_function import available_functions, call_function

def main():
    user_prompt, is_verbose = get_cli_arguments()


    if is_verbose:
        print(f"User prompt: {user_prompt}")

    generate_content = prompt_client()

    parts = [types.Part(text=user_prompt)]

    max_iterations = 20

    for _ in range(max_iterations):
        parts = generate_content(parts, is_verbose)

        if not parts:
            break

        if _ == max_iterations - 1:
            print("Gemini agent reached maximum iterations without a result")
            sys.exit(1)


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

    def add_content(role: str, parts: list[types.Part]):
        messages.append(types.Content(role=role, parts=parts))

    def generate_content(parts: list[types.Part], is_verbose: bool):
        add_content("user", parts)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[available_functions],
            ),
        )

        if not response.usage_metadata:
            raise RuntimeError("Gemini failed to respond. Try again.")

        if is_verbose:
            print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
            print(f"Response tokens: {response.usage_metadata.candidates_token_count}")

        function_results: list[types.Part] = []

        if response.function_calls:
            print("Function calls:")
            for function_call in response.function_calls:
                function_call_result = call_function(function_call, is_verbose)

                if not function_call_result.parts or len(function_call_result.parts) == 0:
                    raise Exception("Function call result does not contain parts")

                function_response = function_call_result.parts[0].function_response

                if not function_response:
                    raise Exception("Function call returned a part with no function response")

                if not function_response.response:
                    raise Exception("Function response does not contain response")

                function_results.append(function_call_result.parts[0])

                if is_verbose:
                    print(f"-> {function_response.response}")
            
            if response.candidates:
                for candidate in response.candidates:
                    messages.append(candidate.content)

            return function_results

        else:
            print("Response:")
            print(response.text)

    return generate_content



if __name__ == "__main__":
    main()
