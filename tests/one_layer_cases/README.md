# One-Layer Forward/Backward Test Cases

This directory contains generated test cases used by `tests/test_one_layer_forward_backward.py`.

## Generate test cases
- Default location: The generator writes here by default.

```bash
python scripts/generate_one_layer_tests.py
```

- Custom options (optional):

```bash
python scripts/generate_one_layer_tests.py --Ns 2,4 --Ds 3,5 --Ks 2,3
```

- Explicit output directory (optional):

```bash
python scripts/generate_one_layer_tests.py --out-dir tests/one_layer_cases
```

## Run the tests
- Quiet mode (default):

```bash
pytest -q tests/test_one_layer_forward_backward.py
```

- Debug output (recommended to run via python so the environment variable is honored):

```bash
ONE_LAYER_DEBUG=1 python tests/test_one_layer_forward_backward.py
ONE_LAYER_DEBUG=2 python tests/test_one_layer_forward_backward.py
```

- On Windows PowerShell:

```powershell
$Env:ONE_LAYER_DEBUG=1; python tests/test_one_layer_forward_backward.py
$Env:ONE_LAYER_DEBUG=2; python tests/test_one_layer_forward_backward.py
```

- Note on pytest and environment variables:
  Some environments may ignore inline environment variables when invoking pytest directly. If you do not see debug output with pytest, use the python invocation shown above.

## Notes
- If no `manifest.json` is present in this directory, the test will be skipped. Generate the cases first using the commands above.
- The debug levels are:
  - 0 or unset: no extra output (default)
  - 1: per-case info and per-assert summary (names, shapes, max|diff|)
  - 2: includes tensor contents for small tensors (<= 200 elements)
