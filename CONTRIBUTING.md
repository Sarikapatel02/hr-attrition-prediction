# Contributing

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the checks before opening a pull request:

```powershell
py -3 -m py_compile app.py src\*.py
py -3 -m pytest -q
```

Keep changes focused, add a regression test for behavior changes, and do not commit raw employee data, credentials, or generated model artifacts. Pull requests should explain any changes to model inputs, evaluation metrics, or deployment behavior.