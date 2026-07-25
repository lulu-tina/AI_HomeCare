# Update and Sync Project to GitHub

You are an expert in Git workflow. I have made changes to the project and need to push them safely.

## 1. Pre-sync Checks
- Scan the project for any new Python imports and update `requirements.txt` if necessary.
- Ensure no sensitive data or local databases in `00_input/`, `00_DB/`, or `.env` are staged (check `.gitignore`).

## 2. Commit Strategy & Interactive Confirmation
1. **Analyze & Draft**: Thoroughly analyze my changes and prepare three things:
   - **High-Level Summary**: A concise 1-2 sentence overview of the overall changes.
   - **Detailed Change Log**: A clear, bulleted list detailing the specific technical modifications (e.g., files modified, functions updated, or logic changed).
   - **Commit Message**: A drafted message following Conventional Commits standards (e.g., "fix: resolve data processing bug").
2. **[CRITICAL PAUSE] Review Request**: You **MUST** pause the execution here. Present the High-Level Summary, the Detailed Change Log, and the drafted Commit Message to me for review.
3. **Wait for Approval**: Explicitly ask me: *"Is this summary, change log, and commit message sufficient? Do I have your permission to proceed with the commit and push?"*
4. **No Auto-Action**: Do **NOT** run `git add`, `git commit`, or any push commands until I provide explicit confirmation (e.g., "Yes", "OK", "Proceed").

## 3. Push and Deploy
- Push current changes to the `develop` branch: `git push origin develop`.
- If I confirm everything is stable, merge `develop` into `main` to trigger Streamlit Cloud update:
  1. `git checkout main`
  2. `git merge develop`
  3. `git push origin main`
  4. `git checkout develop` (return to dev branch)

## 4. Status Report
- Show the latest commit hash for my reference (in case I need to revert).
- Confirm the live URL is updating.