# Text-to-SQL Agent Comparison

## Project Overview

This project implements a text-to-SQL agent and evaluates different chain of thoughts prompting approaches. The goal is to evaluate the effectiveness of these approaches in converting natural language queries into SQL statements.


## Installation

1. Clone the repository:

```bash
git clone https://github.com/dawidpaluszkiewicz/sql_agent_interview.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the Dev Set dataset used for evaluation from here https://bird-bench.github.io/

4. Unzip the file and place it in the `data` directory.

## Usage

1. Setup the environment:

```bash
export OPENAI_API_KEY=<your_openai_api_key>
source .venv/bin/activate
```

2. Run the evaluation script:

```bash
python run_benchmark.py
```

2. To inspect the results, run:

```bash
mlflow server
```

and navigate to the provided URL to view the results.