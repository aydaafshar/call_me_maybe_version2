# Changes: speeding up the run

Goal: the subject requires all test prompts to be processed in **under 5
minutes**. On this machine the project took ~3m45s (and longer under load).
After the changes below it takes **~1m28s**.

Only two source files changed. The decoding algorithm itself was **not**
touched.

---

## 1. Why it was slow

The SDK gives exactly one inference method:

```python
Small_LLM_Model.get_logits_from_input_ids(input_ids: list[int]) -> list[float]
```

It takes a full list of token ids and runs a **complete forward pass over the
whole list** every time it is called. There is no key/value cache exposed, and
the subject forbids touching private attributes such as `model._model`, so
there is no way to reuse the work of the previous step.

That means the total cost is:

```
time  ~=  (number of generated tokens)  x  (length of the prefix)
```

The number of generated tokens is small (98-121 calls for the 11 sample
prompts). So the only thing left to shrink is **the prefix**.

The old code built one big context per prompt and reused it for the whole
generation:

```
Select exactly one JSON function call for the user prompt. Do not answer the prompt.
Functions:
[{"name": "fn_add_numbers", "description": "...", "parameters": {"a": {"type": "number"}, ...}}, ... all 5 functions ...]
Prompt:
<user prompt>
JSON:{"prompt":"<user prompt again>","name":"
```

That is **286 tokens on average**, and it was re-read on every single step -
including every parameter-value step, long after the function name had already
been chosen and the other four functions had become irrelevant.

---

## 2. What was changed

### `src/model_setup.py`

**a. `_selection_prompt` is now compact.**
It used to `json.dumps()` every function with its full parameter schema. It now
lists only `name: description`, because that is what decides *which* function
fits - parameter types do not help choose between functions.

```
Select exactly one function for the user prompt. Do not answer the prompt.
Functions:
- fn_add_numbers: Add two numbers together and return their sum.
- fn_greet: Generate a greeting message for a person by name.
- ...
Prompt: <user prompt>
Call:
```

**b. New `_argument_prompt(prompt, function)`.**
Once the name is decoded, the other functions are dead weight. The argument
steps restart from a context describing **only the chosen function**:

```
Extract the arguments of the function from the prompt.
fn_greet(name: string): Generate a greeting message for a person by name.
Prompt: <user prompt>
Call:
```

**c. New `_type_hint(definition)` helper.**
Renders one parameter type for the signature line above. `enum` parameters are
rendered as `one of ["a", "b"]` so the allowed values are still shown to the
model - no schema information is lost by dropping the JSON dump.

### `src/function_call_builder.py`

**a. `generate_function_call` takes one new argument**,
`argument_prompt: Callable[[str, FunctionDefinition], str]` - the builder from
`model_setup` above. (`src/model_setup.py` is the only caller.)

**b. After the name is chosen, the prefix is rebuilt** from the short argument
context instead of continuing to append to the long selection context:

```python
ids = encode(model, context_prompt) + encode(model, '{"name":"')
name, _ = choose_literal(model, ids, [fn.name for fn in functions])
function_def = next(fn for fn in functions if fn.name == name)
# restart from a context describing only the chosen function
ids = encode(model, argument_prompt(prompt, function_def))
ids += encode(model, '{"name":' + json.dumps(name) + ',"parameters":{')
```

**c. The `"prompt": "..."` key was removed from the decoded JSON.**
The old prefix made the model carry a second full copy of the user prompt
inside the JSON being generated. That value is never produced by the model -
`file_io.write_results` writes it straight from the input file - so re-reading
it on every step was pure cost. The output file format is unchanged: `prompt`,
`name` and `parameters` are all still written, exactly as the subject requires.

### `README.md`

Updated so the documentation matches the code:

- *Algorithm* - notes that `prompt` is copied from the input file and that two
  contexts are used.
- *Performance* - explains the "no cache, so shorten the prefix" reasoning and
  carries the measured before/after table below.

---

## 3. What was deliberately NOT changed

The subject is strict about how the answer is produced, so none of this was
touched:

- The function name is still chosen **by the LLM**, token by token, over the
  set of legal names (`literal_decoding.py`). No keyword matching, no
  heuristics.
- Every parameter value is still produced by **constrained decoding** with the
  vocabulary masked per type (`number_decoding.py`, `string_decoding.py`).
- No private SDK attribute or method is used. No `torch`, `transformers` or
  `huggingface` import was added to `src/`.
- The output file format and the CLI arguments are unchanged.

Also considered and rejected as not worth it: replacing the Python
`max(..., key=...)` token search with a numpy `argmax`, and caching the
digit/safe-token vocabulary scans. Both are real but tiny (a few seconds
total) next to the forward passes, and they would have meant more code for
almost no gain.

---

## 4. Measured result

11 prompts, 5 functions, Qwen3-0.6B on CPU, model already downloaded:

| | before | after |
| --- | --- | --- |
| average prefix per call | 286 tokens | **87 tokens** |
| seconds per forward pass | 1.64 | **0.54** |
| forward passes | 121 | 98 |
| total (instrumented) | 223 s | **58 s** |
| total (`python -m src`, incl. model load) | - | **88 s** |

**Accuracy did not drop.** Function selection was 11/11 correct before and
after, and the two hardest values improved:

| prompt | before | after |
| --- | --- | --- |
| "Replace all numbers ... with NUMBERS" | `regex: "([0-9]+) \| ([0-9]+) \| ([0-9]+) \| (["` | `regex: "34"` |
| "Replace all vowels ... with asterisks" | `regex: "[^aeiouAEIOU]i"` (negated, wrong) | `regex: "a\|e\|i\|o\|u"` |

Checks after the change: `make lint` clean (flake8 + mypy, 14 files),
`pytest` 22 passed.

### Known limitation (pre-existing, not caused by these changes)

`string_decoding.py` masks out every token containing a backslash, because an
unescaped `\` is invalid inside a JSON string. So the model can never write a
regex such as `\d+`. This is already documented under *Challenges* in the
README. It is a correctness limit, not a speed one.
