# Fixtures of the core

| File | sha256 | Origin |
|---|---|---|
| `lower_reference.json` | `3681b59a793de52dd4184d0c0ee3ce3e4b3815323aa066b5264b3be0a0bbef4a` | Python 3.11.14 `str.lower()` for all code points (Unicode 14.0.0); input of `tools/gen_python_text_data.py` |
| `text_cases.json` | `5c7b31ac3d6cf1078a5ce91f620b0bdc06334f8d4110a8912814e45477de79e7` | text cases (input bytes, text after Python's universal-newline read, Python's lowercase) used by the `PythonText` tests |
| `rope_hf.safetensors` | `e606990bde54e2042884d3ed5fdf8bebd266d73037691bd7a8ee72c7d6cbd1bb` | HuggingFace Llama rotary embedding (`transformers` 4.47.1, torch 2.7.0 CPU, fp32): two cases `a`/`b` with `x`, `pos`, `out`, `params`; generated once by `tools/dump_rope_hf.py` (recipe in its docstring) |

The first two files are **byte-identical copies** of the ones in HeartMuLa.Net's `tests/fixtures`, which produced them with
`tools/dump_text_reference.py` (recipe in that repository's `tests/fixtures/MANIFEST.md`: minimal venv with Python 3.11 and
`tokenizers==0.22.1`). `text_cases.json` also holds tokenizer ids that only HeartMuLa.Net's tests use. If either file is
regenerated there, copy it here, update the sha256 above, and run `python3 tools/gen_python_text_data.py` (standard library only);
the exhaustive `PythonTextLowerTests` fails if `PythonTextData.cs` is stale.
