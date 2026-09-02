# Project Rules & Context

1. **Virtual Environment (venv)**: 
   - The project uses a Python 3 virtual environment located at `./venv` in the root directory. 
   - Any package installations or terminal commands must be executed within this environment (e.g., using `source venv/bin/activate` before installing, or using `./venv/bin/pip`).

2. **Models Location**: 
   - All trained models are located in the `./models` directory.

3. **Git Version Control**: 
   - When asked to commit, **ONLY** use `git commit`. 
   - **NEVER** use `git push`. The user will handle pushing the code manually.

4. **Frontend Styling & CSS Guidelines**:
   - **Strictly NO Inline CSS (`style="..."`)**: Never write inline CSS attributes in HTML templates or UI components.
   - **Always Use External CSS**: Define all styling in external CSS files (e.g. `assets/styles.css`).
   - **CSS Class & ID Selectors**: Link styles exclusively through semantic class names and ID selectors.
   - **No Unicode Emojis**: Use clean textual tags or icons instead of emojis in UI strings, comments, and console prints.
