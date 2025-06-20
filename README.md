
# AI Policy Violation Detector

## Installation

### Setup
1. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Run the application:
```bash
poetry run python src/main.py
```


## Current Functionality

The application analyzes code diffs against policy guidelines to detect potential violations. The main workflow:

1. Reads a diff file (`tests/pr_diff.txt`) and policy file (`tests/violation_prompt.txt`) from the specified paths
2. Initializes an AI agent to analyze the diff content
3. The agent evaluates the diff against the policy rules
4. Returns a list of detected violations

## Usage

The application currently uses the paths defined in `main.py` to analyze a sample diff file against policy guidelines.



## Your task: LLM Evaluation Implementation

This is a take-home assessment focused on implementing LLM evaluation. Your task is to:

1. Implement an evaluation system for the LLM-based policy violation detector
2. Use any platform or library of your choice
3. Create metrics to measure the effectiveness of the violation detection
4. Demo your approach with a quick video along with your code.

The evaluation should help assess how well the AI agent identifies policy violations in code diffs.

Note: You will need your own Anthropic API key to run the code (let us know if you need help with this). Feel free to contact us for any questions you may have. :)

## Bonus

We love AI coding agents. If you think you're particularly skilled at using and getting the best results from them, try using [Amp](https://ampcode.com) and [sharing your threads publicly](https://ampcode.com/manual#privacy-and-permissions) with us. We'd love to see what you come up with!

