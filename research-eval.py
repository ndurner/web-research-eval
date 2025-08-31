import argparse
import concurrent.futures
import json
import os
import sys
import time

from pathlib import Path

import anthropic
import mistralai
import openai

from cryptography.fernet import Fernet
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm


ENCRYPTION_KEY = b'8FElACa70l2pKOQ3F5v0ujTRqo6yX3bmO8ZWBFikQcQ='


def decrypt(encrypted_text):
    return Fernet(ENCRYPTION_KEY).decrypt(encrypted_text.encode('utf-8')).decode('utf-8')


def read_jsonl(path):
    data = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    with open(path, mode='w', encoding='utf-8') as f:
        for entry in data:
            print(json.dumps(entry, ensure_ascii=False), file=f)


def generate_openai(prompt, model, api_key, base_url=None, extra_body=None):
    kwargs = {} if base_url is None else {'base_url': base_url}
    client = openai.OpenAI(api_key=api_key, **kwargs)
    messages = [
        {
            'role': 'user',
            'content': prompt,
        }
    ]
    kwargs = {} if extra_body is None else {'extra_body': extra_body}
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    message = response.choices[0].message.content
    if message is None:
        raise ValueError('Missing text response')
    return message


def generate_openai_responses(prompt, model, api_key, use_web_search=False, base_url=None):
    """Generate using OpenAI Responses API, with optional web_search tool.

    Args:
        prompt: User input string.
        model: OpenAI model name.
        api_key: API key string.
        use_web_search: If True, enable the web_search tool.
        base_url: Optional custom base URL (not typically used for OpenAI proper).

    Returns:
        The assistant's text output.
    """
    kwargs = {} if base_url is None else {'base_url': base_url}
    client = openai.OpenAI(api_key=api_key, **kwargs)
    create_kwargs = {
        'model': model,
        'input': prompt,
    }
    if use_web_search:
        create_kwargs['tools'] = [{'type': 'web_search'}]
        # Let the model decide when to use tools
        create_kwargs['tool_choice'] = 'auto'
    response = client.responses.create(**create_kwargs)

    # Prefer the convenience accessor when available
    text = getattr(response, 'output_text', None)
    if text:
        return text

    # Fallback: try to concatenate any text parts from the response
    try:
        parts = []
        for item in getattr(response, 'output', []) or []:
            for c in getattr(item, 'content', []) or []:
                if getattr(c, 'type', '') == 'output_text' and getattr(c, 'text', None):
                    parts.append(c.text)
                elif getattr(c, 'type', '') == 'input_text' and getattr(c, 'text', None):
                    # ignore user echo; only collect assistant output
                    pass
        if parts:
            return ''.join(parts)
    except Exception:
        pass
    raise ValueError('Missing text response from Responses API')


def generate_reka(prompt, model, api_key):
    return generate_openai(prompt, model, api_key, 'https://api.reka.ai/v1')


def generate_perplexity(prompt, model, api_key):
    response = generate_openai(prompt, model, api_key, 'https://api.perplexity.ai')
    if 'reasoning' in model:
        response = response.split('</think>', 1)[1].lstrip()
    return response


def generate_xai(prompt, model, api_key):
    return generate_openai(prompt, model, api_key, 'https://api.x.ai/v1', extra_body={'search_parameters': {'mode': 'on'}})


def generate_fireworks(prompt, model, api_key):
    if not model.startswith('accounts/'):
        model = 'accounts/fireworks/models/' + model
    return generate_openai(prompt, model, api_key, 'https://api.fireworks.ai/inference/v1')


def generate_google(prompt, model, api_key):
    client = genai.Client(api_key=api_key)
    tools = [
        genai.types.Tool(google_search=genai.types.GoogleSearch),
    ]
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            tools=tools,
            response_modalities=['TEXT'],
        )
    )
    if response.text is None:
        raise ValueError('Missing text response')
    return response.text


def generate_anthropic(prompt, model, api_key):
    client = anthropic.Anthropic(api_key=api_key)
    messages = [
        {
            'role': 'user',
            'content': prompt,
        }
    ]
    tools = [
        {
            'type': 'web_search_20250305',
            'name': 'web_search',
        }
    ]
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=messages,
        tools=tools
    )
    ans = []
    for chunk in response.content:
        if chunk.type == 'text':
            ans.append(chunk.text)
        else:  # Reset answer if a model calls a tool, which means previous messages were intermediate CoT
            ans = []
    return ''.join(ans)


def generate_mistral(prompt, model, api_key):
    # From https://docs.mistral.ai/agents/connectors/websearch/
    client = mistralai.Mistral(api_key=api_key)
    websearch_agent = client.beta.agents.create(
        model=model,
        description='Agent able to search information over the web, such as news, weather, sport results...',
        name='Websearch Agent',
        instructions='You have the ability to perform web searches with `web_search` to find up-to-date information.',
        tools=[{'type': 'web_search'}],
        completion_args={
            'temperature': 0.3,
            'top_p': 0.95,
        },
    )
    response = client.beta.conversations.start(
        agent_id=websearch_agent.id,
        inputs=prompt,
    )
    content = response.outputs[-1].content
    if isinstance(content, str):
        return content
    else:
        return ''.join([chunk.text for chunk in content if chunk.type == 'text'])


def generate_with_retry(generate, prompt, model, api_key, trials, validator=None):
    seconds_to_wait = 1
    # exc = None
    for _ in range(trials):
        try:
            generation = generate(prompt, model, api_key)
            if validator is not None and not validator(generation):
                raise ValueError(f'Invalid generation: {generation}')
            return generation
        except Exception as e:
            # exc = e
            # print('>>> Trying again:', exc)
            time.sleep(seconds_to_wait)
            seconds_to_wait *= 2
    # raise RuntimeError from exc
    print(f'WARNING: Generation failed after {trials} trials, returning dummy response', file=sys.stderr)
    return f'Generation failed after {trials} trials'


def generate_prompts(generate, prompts, model, api_key, trials, concurrency, validator=None):
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(generate_with_retry, generate, prompt, model, api_key, trials, validator) for prompt in prompts]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()
        return [future.result() for future in futures]


def format_judge(prompt, generation, requirements) -> str:
    formatted_requirements = []
    answer_format = []
    for i, requirement in enumerate(requirements):
        formatted_requirements.append(f"{i+1}) {requirement.strip()}")
        answer_format.append(
            f"Analysis of requirement {i+1}: {{Your brief analysis of whether requirement {i+1} is met. Think step by step before committing to any conclusion.}}"
        )
    answer_format.append(
        "Final answer: {REQUIREMENTS_MET if the response meets all of the requirements, or REQUIREMENTS_NOT_MET if the response violates one or more requirements}"
    )
    requirements_text = "\n\n".join(formatted_requirements)
    answer_format_text = "\n".join(answer_format)
    prompt = (
        "[User Prompt]\n"
        f"{prompt.strip()}\n\n"
        "[Assistant Response]\n"
        f"{generation.strip()}\n\n"
        "[Requirements]\n"
        f"{requirements_text}\n\n"
        "[System]\n"
        "Assess if the assistant response meets all of the requirements above. You should not evaluate the correctness of the response, simply check if it meets the requirements given. "
        "Your response should be in the following format:\n\n"
        f"{answer_format_text}"
    )
    return prompt


def add_generate_args(parser, default_api=None, default_model=None):
    parser.add_argument('--api', default=default_api, required=default_api is None, choices=API2CONFIG.keys(), help='The API provider')
    parser.add_argument('--model', default=default_model, required=default_model is None, help='The model to be used with the selected API provider')
    parser.add_argument('--concurrency', default=30, type=int, help='Number of concurrent calls')
    parser.add_argument('--trials', default=6, type=int, help='How many times to retry for failed requests')
    parser.add_argument('--web-search', action='store_true', help='Enable web_search tool when supported (OpenAI Responses)')


def add_io_args(parser, input=True, output=True, default_input=None, default_output=None):
    if input:
        parser.add_argument('-i', '--input', default=default_input, required=default_input is None, help='Path to the input file')
    if output:
        parser.add_argument('-o', '--output', default=default_output, required=default_output is None, help='Path to the output file')


API2CONFIG = {
    'reka': [generate_reka, 'REKA_API_KEY'],
    'openai': [generate_openai, 'OPENAI_API_KEY'],
    'openai-responses': [generate_openai_responses, 'OPENAI_API_KEY'],
    'fireworks': [generate_fireworks, 'FIREWORKS_API_KEY'],
    'perplexity': [generate_perplexity, 'PERPLEXITY_API_KEY'],
    'xai': [generate_xai, 'XAI_API_KEY'],
    'google': [generate_google, 'GOOGLE_API_KEY'],
    'anthropic': [generate_anthropic, 'ANTHROPIC_API_KEY'],
    'mistral': [generate_mistral, 'MISTRAL_API_KEY'],
}


def main():
    parser = argparse.ArgumentParser(description='Research-Eval evaluation script')
    subparsers = parser.add_subparsers(dest='command', required=True, help='The command to run')
    parser_generate = subparsers.add_parser('generate', help='Generate responses')
    add_generate_args(parser_generate)
    add_io_args(parser_generate, default_input=Path(__file__).parent / 'research-eval-v1.0.jsonl')
    parser_judge = subparsers.add_parser('judge', help='Judge a set of generations')
    add_generate_args(parser_judge, default_api='fireworks', default_model='deepseek-v3-0324')
    add_io_args(parser_judge)
    parser_report = subparsers.add_parser('report', help='Report results from a set of judged generations')
    add_io_args(parser_report, output=False)
    args = parser.parse_args()

    load_dotenv()

    if args.command != 'report':
        base_generate, api_var = API2CONFIG[args.api]
        api_key = os.getenv(api_var)
        if api_key is None:
            print(f'API key not found, please set the following environment variable: {api_var}', file=sys.stderr)
            sys.exit(-1)

        # Wire optional web_search for OpenAI Responses; warn if requested for Chat Completions
        if args.api == 'openai-responses':
            if args.web_search:
                def generate(prompt, model, api_key):
                    return generate_openai_responses(prompt, model, api_key, use_web_search=True)
            else:
                generate = base_generate
        else:
            if args.web_search and args.api == 'openai':
                print('Note: --web-search is only supported with OpenAI Responses; ignoring for Chat Completions', file=sys.stderr)
            generate = base_generate

    if args.command == 'generate':
        examples = read_jsonl(args.input)
        prompts = [decrypt(example['prompt']) for example in examples]
        generations = generate_prompts(generate, prompts, args.model, api_key, args.trials, args.concurrency)
        for example, generation in zip(examples, generations):
            example['generation'] = generation
            if 'metadata' not in example:
                example['metadata'] = {}
            example['metadata'].update({
                'generation_api': args.api,
                'generation_model': args.model,
            })
        save_jsonl(examples, args.output)
    elif args.command == 'judge':
        examples = read_jsonl(args.input)
        validator = lambda generation: 'REQUIREMENTS_MET' in generation or 'REQUIREMENTS_NOT_MET' in generation
        prompts = [format_judge(decrypt(example['prompt']), example['generation'], map(decrypt, example['requirements'])) for example in examples]
        generations = generate_prompts(generate, prompts, args.model, api_key, args.trials, args.concurrency, validator)
        for example, generation in zip(examples, generations):
            example['judge_output'] = generation
            example['metadata'].update({
                'judge_api': args.api,
                'judge_model': args.model,
            })
        save_jsonl(examples, args.output)
    elif args.command == 'report':
        examples = read_jsonl(args.input)
        score = 100 * sum(['REQUIREMENTS_MET' in example['judge_output'] for example in examples]) / len(examples)
        api = examples[0]['metadata']['generation_api']
        model = examples[0]['metadata']['generation_model']
        api_model = f'{api}/{model}'
        print(f'{api_model:50s} {score:.1f}')


if __name__ == '__main__':
    main()