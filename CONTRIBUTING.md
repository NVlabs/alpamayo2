# Contributing To Alpamayo 2 Super

Thank you for your interest in Alpamayo 2 Super. This repository contains the public inference
package, examples, notebook, and tests for the released expert model.

## Contribution Guidelines

- Keep changes focused on one concern at a time.
- Follow the existing Python and documentation style in the surrounding files.
- Run `pytest -q` before sending a change.
- Update `README.md`, examples, or tests when user-facing behavior changes.
- Do not commit model weights, generated inference outputs, caches, credentials, or dataset files.

## Pull Requests

Open a pull request from a fork or feature branch and describe:

- the problem being fixed or feature being added
- commands you ran for validation
- any model, dataset, GPU, or environment assumptions needed to reproduce the result

Use sign-off if your project workflow requires Developer Certificate of Origin tracking:

```bash
git commit --signoff -m "Describe the change"
```

## Large Artifacts

Model checkpoints and dataset assets are distributed separately from this source repository. Keep
large artifacts in Hugging Face or another approved artifact store and reference them by model id,
dataset id, or local path in examples.
