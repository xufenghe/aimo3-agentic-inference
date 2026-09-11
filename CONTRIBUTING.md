# Contributing

Focused, test-backed contributions are welcome.

## Development loop

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make check
```

Before opening a pull request:

1. Add or update a deterministic test for behavioral changes.
2. Run `make check` on Python 3.11 or newer.
3. Keep model weights, datasets, credentials, generated outputs, and private problem text out of the commit.
4. Update the relevant documentation when changing a public interface or CLI flag.
5. State exactly what was tested. Do not imply a GPU, benchmark, or competition result that was not run.

## Design preferences

- Keep the core dependency-light and backend-neutral.
- Prefer typed records and small protocol boundaries over provider-specific state.
- Keep deadlines, cancellation, failure behavior, and security assumptions explicit.
- Treat chain-of-thought as private by default; log aggregate outcomes instead.
- Avoid weakening strict parsing to make a single benchmark look better.

For a larger proposal, open a design issue before implementing it.
