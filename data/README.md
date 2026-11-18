# Calibration and test splits

These files preserve the questions, answers, membership, and ordering of the supplied experiment splits. The original evaluation files are named `test.jsonl.gz` in this repository. The runner reads them directly.

| Benchmark | Calibration | Test | Upstream |
| --- | ---: | ---: | --- |
| MATH | 2,500 | 2,500 | [Hendrycks et al.](https://github.com/hendrycks/math) · [MIT notice](math/LICENSE) |
| GSM8K | 660 | 659 | [OpenAI grade-school-math](https://github.com/openai/grade-school-math) · [MIT notice](gsm8k/LICENSE) |

MATH records contain `problem`, `answer`, `solution`, and the original metadata. GSM8K records contain `question` and `answer`. The final reference answer follows the `####` marker.

Score both splits using the same checkpoint, prompting, decoding settings, and uncertainty measure. Calibration uses the scores of examples whose generated final answers fail the benchmark's answer check. Use test scores only to report held-out metrics.

SHA-256 of the uncompressed JSONL files:

| File | SHA-256 |
| --- | --- |
| `math/calibration.jsonl` | `3be46485824734d2e081e8fe70fd5ec4e612554df703dde924d5084350ca8375` |
| `math/test.jsonl` | `f2a90207302cf42711fdbc9492298ca33ddbdb0dc37f89616b5d3529ca45d523` |
| `gsm8k/calibration.jsonl` | `accfc1e74466fa0a915a344a8099bf9711a0481059ee9cba73db5dde8c61215e` |
| `gsm8k/test.jsonl` | `88d4e1f889bd9fd85ab373574f6425426411fb2eb6f9a7fc251fad8675d74d2f` |
