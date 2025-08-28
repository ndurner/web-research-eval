# Research-Eval

A benchmark to evaluate search-augmented LLMs.

In addition to the dataset, the repository provides a simple evaluation script that supports most frontier models out of the box (including Reka, Google, OpenAI, Perplexity, Anthropic, XAI, and Mistral).

Check out our [blogpost](https://reka.ai/news/introducing-research-eval-a-benchmark-for-search-augmented-llms) for more details.


## Setup

Install dependencies as follows:
```bash
pip install -r requirements.txt
```

API keys are read from environment variables, set them as needed:
```bash
export REKA_API_KEY='YOUR_API_KEY'
export FIREWORKS_API_KEY='YOUR_API_KEY'
export GOOGLE_API_KEY='YOUR_API_KEY'
export OPENAI_API_KEY='YOUR_API_KEY'
export PERPLEXITY_API_KEY='YOUR_API_KEY'
export ANTHROPIC_API_KEY='YOUR_API_KEY'
export XAI_API_KEY='YOUR_API_KEY'
export MISTRAL_API_KEY='YOUR_API_KEY'
```
We recommend setting all relevant environment variables in `.env`, which the evaluation code loads automatically.

Not all API keys are required, only set the relevant ones for the models you want to evaluate. As the only exception, `FIREWORKS_API_KEY` is always required for scoring, as it is used to run the recommended `deepseek-v3-0324` judge.


## Usage

#### 1. Generate responses

```bash
python research-eval.py generate \
    --api reka \
    --model reka-flash-research \
    --concurrency 20 \
    -o experiments/reka-flash-research.out.jsonl
```

#### 2. Judge responses

```bash
python research-eval.py judge \
    -i experiments/reka-flash-research.out.jsonl \
    -o experiments/reka-flash-research.eval.jsonl
```

This uses the `deepseek-v3-0324` model through Fireworks by default, which is our recommended judge. While other judge models can be used, we strongly discourage publishing results obtained with them, as this would make numbers non-comparable.


#### 3. Report results

```bash
python research-eval.py report \
    -i experiments/reka-flash-research.eval.jsonl
```


## Leaderboard

| Model                                     | Score |  Cost  |
|-------------------------------------------|-------|--------|
| Reka Research                             |  59.1 |  25.00 |
| Gemini 2.0 Flash (w/ Google Search)	    |  54.2 |  35.05 |
| GPT-4o search preview                     |  53.0 |  37.14 |
| Claude Sonnet 4 (w/ web search)           |  44.8 | 162.01 |
| Sonar Reasoning Pro                       |  44.6 |  14.75 |
| Mistral Medium 2505 (w/ web search agent) |  31.5 |  32.51 |
| Grok 4 (w/ live search)                   |  26.7 |  85.88 |

We report average accuracy across 5 runs. The cost refers to USD per 1,000 requests in the Research-Eval distribution.


## Reproducing results

To reproduce the results in the leaderboard, run the following script:

```bash
./reproduce-results.sh
```


## License

The content of this repository is licensed under a [Modified MIT License](LICENSE).

The dataset is distributed in encrypted form to prevent contamination, and our license prohibits redistributing it in decrypted or plain-text form.