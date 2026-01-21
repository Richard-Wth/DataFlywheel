import openai
from openai import OpenAI

api_key = "sk-6psVESKMGkzC4lPHHI46FIdkacdT7rWOpijx5rFhXlaZzydw"
base_url = "https://zjuapi.com/v1"

client = OpenAI(api_key=api_key, base_url=base_url)

prompt = """
Let $ABCDE$ be a convex pentagon with $AB=14, BC=7, CD=24, DE=13, EA=26,$ and $\\angle B=\\angle E=60^\\circ$. For each point $X$ in the plane, define $f(X)=AX+BX+CX+DX+EX$. The least possible value of $f(X)$ can be expressed as $m+n\\sqrt{p}$, where $m$ and $n$ are positive integers and $p$ is not divisible by the square of any prime. Find $m+n+p$.
"""

_req = dict(
    model="gpt-5.2",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.6,
)

# "think/推理模式" 在不同 OpenAI 兼容网关里字段名不统一：
# - 这里优先尝试 OpenAI 生态更常见的 reasoning/effort 写法
# - 若网关不支持该参数，则自动降级为普通请求（保证不 400）
try:
    completion = client.chat.completions.create(
        **_req,
        extra_body={"reasoning": {"effort": "high"}},
    )
except openai.BadRequestError as e:
    msg = str(e)
    if "Unknown parameter" in msg and ("reasoning" in msg or "reasoning_effort" in msg):
        completion = client.chat.completions.create(**_req)
    else:
        raise

print(completion.choices[0].message.content)