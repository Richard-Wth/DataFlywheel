1. select benchmark from fixed datasets and dynamic datasets.
2. train model in a basline training dataset(DataFlywheel/data/training/deepseek_math_converted.json), then get a base model.
3. inference on benchmark, judge by LLM and get bad cases.
4. generate training data based on bad case attributions.
5. train a model with LLM generated data and raw training data.
6. inference on benchmark and compare performance.

Notes:
- For `src/judge_inference.py`, put `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL`) into a `.env` file under `DataFlywheel/`.
  You can start from `env.template`:
  - `cp env.template .env`
- (Optional) If you want to avoid committing `.env`, you can start from `gitignore.template`:
  - `cp gitignore.template .gitignore`
