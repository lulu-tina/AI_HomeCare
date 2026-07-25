Deploy this project to GitHub and output Streamlit Cloud deployment instructions.

# Prepare & Push Streamlit App to GitHub

This command sets up Git tracking, cleans up sensitive files, creates necessary configuration files, 
and pushes the project to GitHub so it is ready for deployment on Streamlit Community Cloud (https://share.streamlit.io).

## Steps

### 1. Pre-flight checks & Environment context

Verify CLI tools are available:
- Check `git --version` and `gh --version`
- Check GitHub auth status: `gh auth status`. If not authenticated, prompt the user to run `gh auth login`.

Retrieve variables:
- `GITHUB_USER`: via `gh api user --jq .login`
- `REPO_NAME`: from the current working directory name (sanitized for URL compatibility)

### 2. Smart File Generation (if missing or incomplete)

1. **`requirements.txt`**:
   - Parse all `.py` files in the workspace to discover imported third-party libraries (e.g., streamlit, pandas, etc.).
   - Generate or update `requirements.txt` with these dependencies, ignoring Python built-in modules.

2. **`.gitignore`**:
   - Create or update `.gitignore` to strictly exclude:
     - Sensitive data & local databases: `00_input/`, `00_DB/`, `*.db`, `*.sqlite`, `*.sqlite3`
     - Local export/raw data files: `*.xlsx`, `*.csv` (unless explicitly flagged as static reference files)
     - Caches & Virtual Environments: `__pycache__/`, `*.pyc`, `*.pyo`, `.venv/`, `venv/`, `.env`, `*.pem`
     - IDE & OS metadata: `.vscode/`, `.idea/`, `.DS_Store`

### 3. Git Initialization & Remote Handling

- If `.git` directory does not exist: run `git init`
- Ensure the primary branch is set to `main`: `git branch -M main`
- Check if remote `origin` exists:
  - If NOT present, create a public GitHub repository and link it:
    `gh repo create "$REPO_NAME" --public --source=. --remote=origin`

### 4. Staging Safety Verification & Commit

- Stage tracked files: run `git add .`
- **CRITICAL PRIVACY CHECK**: Run `git status` and inspect all staged files. Verify that NO database directories (`00_DB/`), input folders (`00_input/`), credentials (`.env`), or private data files are staged.
- If any sensitive file is staged, abort immediately, unstage the file, and update `.gitignore`.
- Commit changes: `git commit -m "chore: prepare repository for Streamlit app deployment"`
- Push to GitHub: `git push -u origin main`

### 5. Final Report & Next Steps

Print a clear summary containing:
1. **GitHub Repository URL**: `https://github.com/$GITHUB_USER/$REPO_NAME`
2. **Streamlit Deployment Instructions**:
   - Go to https://share.streamlit.io and sign in with GitHub.
   - Click **"New app"** → select Repository `$GITHUB_USER/$REPO_NAME`, Branch `main`, Main file path (e.g., `app.py`).
   - Click **Deploy**.
3. **Data Security Reminder**: Explicitly highlight that local databases (`00_DB/`) and raw inputs (`00_input/`) are intentionally excluded for privacy/data safety, and should be loaded dynamically via user interface or cloud database integrations at runtime.