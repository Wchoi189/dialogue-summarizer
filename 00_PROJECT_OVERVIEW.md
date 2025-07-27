
# Project Overview - Dialogue Summarization

> **IMPORTANT**: Read files in order: 00→01→02→03→04 for complete context

## **Quick Context**
- **Timeline**: July 27 - August 6, 2025
- **Goal**: Improve upon 47.1244 ROUGE-F1 baseline
- **Status**: Core infrastructure complete, evaluation phase next
- **Current Phase**: 2D-2E (Evaluation & Inference)
----
# **Project Prompt**

**Project Title:** Modular Dialogue Summarization with PyTorch Lightning & Hydra

**1. Overall Goal**

My objective is to refactor a notebook-based dialogue summarization project into a modular, production-ready application **within the project window of July 27 to August 6, 2025**.

The goal is to **improve upon or take ideas from the existing baseline codes** (`baseline.ipynb` and `solar_api.ipynb`) to create a system that is highly configurable using Hydra and uses PyTorch Lightning for training. All project context is detailed in the `data_overview.pdf` and `dialogue_summarization_overview.pdf` files.

**2. Phase 1: Assessment and Project Scaffolding (No Code Generation Yet)**

Before writing any implementation code, I need you to provide a detailed project plan. This plan should include:
* **Proposed Project Structure:** A clear directory tree for a modern NLP project.
* **Component Breakdown:** A description of the key Python modules and their responsibilities.
* **Configuration Strategy:** An outline of how we will use Hydra to manage datasets, model hyperparameters, and training settings.
* **Initial Model Choice:** Based on the project documents, confirm if `KoBART` is the best starting point.
* **Utility Script Assessment:** Evaluate my potentially reusable scripts (`config_utils.py`, `wandb_utils.py`, `project_setup.py`) and suggest how they can be integrated or adapted for this project.

**Do not proceed to Phase 2 until I approve this plan.**

**3. Phase 2: Code Implementation (Module by Module)**

Once the plan is approved, we will build the project one component at a time. Your task is to generate the code for each part, following this technology stack:
* **Environment Manager:** `micromamba`
* **Dependencies File:** `environment.yml` (clearly separating conda and pip packages).
* **Core Libraries:** `Python 3.10.13`, `PyTorch`, `PyTorch Lightning`, `Hydra`, `WandB`, `Hugging Face Transformers`.

**4. Code & Collaboration Style**

Please adhere to the following rules throughout our collaboration:
* **Work Sequentially:** Generate code for only one module at a time, as we agree upon it.
* **Label Clearly:** Use markdown headings to label every file or code block (e.g., `### File: src/utils/config_utils.py`).
* **Concise Code:** Keep Python files focused and ideally under 300 lines. Use a central `src/utils/` module for reusable functions.
* **Validate Code:** Periodically generate necessary test scripts to validate new features.
* **Context Window:** Keep context window usage to less than 50,000 tokens to maximize context awareness.
* **Precise Fixes:** When fixing code, provide only the line(s) that need to be changed and specify where the change should occur. Do not reprint the entire file.
* **Context Handover:** If we are nearing the context limit(50,000 tokens), please generate a summary of the project's current state and a prompt to continue our work in a new session.
* **Resource Management Tips:** I would like suggestions on how I can reduce token usage and which files to exclude in the project knowledge when planning for new features.
* **Update Docs:** Markdown files in the root directory may need updates after new features. Suggest snippets of updates that should be added and I will manually copy and paste. Request to add/remove additional docs as necessary. I will provide information.  

**5. Additional Information**
* **Additional contexts:** Additional contexts such as system information and data overview can be found in docs/