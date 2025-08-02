
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

----
## Competition Overview

**Dialogue Summarization Competition** challenges participants to create models that can summarize everyday conversations on topics like school, work, healthcare, and travel. The competition runs from **July 25, 2025, to August 6, 2025**.

---

## Dataset Details

The data is provided in `.csv` format and is structured with the following columns:

- **fname**: A unique ID for each conversation.
- **dialogue**: The text of the conversation, which involves between 2 and 7 speakers and consists of 2 to 60 turns. Speakers are identified with labels like `#Person1#`.
- **summary**: The reference summary for the dialogue.

The dataset includes various types of **noise**, such as typos, inconsistent punctuation, speaker labels, and formatting issues like `\\n` characters or `<br>` tags.

The data is split into:
- **Test Set**: 250 dialogues.
- **Hidden Test Set**: 249 dialogues.

---

## Modeling and Baseline

A baseline solution is provided using **KoBART**, a model based on BART that is specifically tailored for the Korean language. The baseline code is organized into several key files:

- `baseline.ipynb`: Contains the primary code for data processing, model training, and inference.
- `config.yaml`: A configuration file for model parameters.
- `solar_api.ipynb`: An alternative approach that demonstrates how to use the Solar Chat API for summarization, including prompt engineering techniques.

The training process involves preprocessing the data for the model's encoder and decoder and then using the Hugging Face `Trainer` class for training.

---

## Evaluation

The performance of the summarization models is evaluated using the **ROUGE score**. To ensure a fair and comprehensive assessment, each dialogue in the test set is evaluated against **three different reference summaries**. This approach acknowledges that a single conversation can have multiple valid summaries. The baseline model achieved a **ROUGE-F1 score of 47.1244** on the public test data.

---

## Introduction (Original Content)

This introduction provides context for the competition and the nature of the dialogues being summarized, covering a broad range of everyday situations such as school life, work, healthcare, shopping, leisure, and travel. Participants are tasked with creating models that effectively generate summaries of these dialogues using the provided dataset, which includes only dialogue and summary pairs. The competition aims to develop accurate and generalized models that can summarize key parts of conversations, improving work efficiency and interpersonal relationships.

---

## Sample Dialogue and Summary

### Sample 1: Health Check-up

- **Dialogue**:
  - `#Person1#`: 안녕하세요, 스미스씨, 저는 호킨스 의사입니다. 오늘 왜 오셨나요? (Hello, Mr. Smith, I'm Dr. Hawkins. Why are you here today?)
  - `#Person2#`: 건강검진을 받는 것이 좋을 것 같아서요. (I thought it would be good to get a check-up.)
  - `#Person1#`: 그렇군요, 당신은 5년 동안 건강검진을 받지 않았습니다. 매년 받아야 합니다. (I see, you haven't had a check-up in 5 years. You should have one every year.)
  - `#Person2#`: 알고 있습니다. 하지만 아무 문제가 없다면 왜 의사를 만나러 가야 하나요? (I know. But why should I go to the doctor if there are no problems?)
  - `#Person1#`: 심각한 질병을 피하는 가장 좋은 방법은 이를 조기에 발견하는 것입니다. 그러니 당신의 건강을 위해 최소한 매년 한 번은 오세요. (The best way to avoid serious illness is to detect it early. So please come at least once a year for your health.)
  - `#Person1#`: 여기 보세요. 당신의 눈과 귀는 괜찮아 보입니다. 깊게 숨을 들이쉬세요. 스미스씨, 담배 피우시나요? (Look here. Your eyes and ears look fine. Take a deep breath. Mr. Smith, do you smoke?)
  - `#Person2#`: 네. (Yes.)
  - `#Person1#`: 당신도 알다시피, 담배는 폐암과 심장병의 주요 원인입니다. 정말로 끊으셔야 합니다. (As you know, smoking is a major cause of lung cancer and heart disease. You really need to quit.)
  - `#Person2#`: 수백 번 시도했지만, 습관을 버리는 것이 어렵습니다. (I've tried hundreds of times, but it's hard to break the habit.)
  - `#Person1#`: 우리는 도움이 될 수 있는 수업과 약물들을 제공하고 있습니다. 나가기 전에 더 많은 정보를 드리겠습니다. (We offer classes and medications that can help. I'll give you more information before you leave.)
- **Gold Summary**: 스미스씨가 건강검진을 받고 있고, 호킨스 의사는 매년 건강검진을 받는 것을 권장합니다. 호킨스 의사는 스미스씨가 담배를 끊는 데 도움이 될 수 있는 수업과 약물에 대한 정보를 제공할 것입니다. (Mr. Smith is getting a check-up, and Dr. Hawkins recommends getting one every year. Dr. Hawkins will provide information on classes and medication to help Mr. Smith quit smoking.)
- **Predicted Summary**: 호킨스 의사는 스미스씨에게 매년 건강검진을 받는 것이 심각한 질병을 조기에 발견하여 예방하는 데 중요하다고 조언합니다. 의사는 스미스씨의 눈과 귀를 검사하고, 폐암과 심장병의 위험을 증가시키는 흡연 습관을 끊을 것을 권장합니다. 의사는 스미스씨에게 도움을 줄 수 있는 수업과 약물에 대한 정보를 제공합니다. (Dr. Hawkins advises Mr. Smith that getting an annual check-up is important for preventing serious diseases by detecting them early. The doctor checks Mr. Smith's eyes and ears and recommends he quit his smoking habit, which increases the risk of lung cancer and heart disease. The doctor provides Mr. Smith with information on classes and medications that can help.)

### Sample 2: Office Memo

- **Dialogue Snippet**:
  - `#Person1#`: 모든 직원들에게 주의하라... 즉시 효력을 발휘하여, 모든 사무실 동선은 이메일 통신과 공식 에모로 제한됩니다. (Attention all employees... Effective immediately, all office communication is restricted to email correspondence and official memos.)
  - `#Person1#`: 근무 시간 동안 직원들이 즉시 메시지 프로그램을 사용하는 것은 엄격히 금지됩니다. (The use of instant messaging programs by employees during work hours is strictly prohibited.)
  - `#Person2#`: 실장님, 이것은 내부 총신에만 적용되는 건가요? 아니면 외부 통신에도 제한이 되는 건가요? (Director, does this apply only to internal communications? Or is it restricted for external communications as well?)
  - `#Person1#`: 이것은 모든 통신에 적용되어야 합니다. 이 사무실 내의 직원 사이뿐만 아니라 외부 통신에도 마찬가지입니다. (This must apply to all communications. Not only between employees within this office, but also to external communications.)
- **Summary**: 더슨 씨는 Person에게 모든 사무실 통신이 어머 통신과 공식 메모 제한된다고 알려준다. #Perso는 이것이 내부와 외부 통신에 적용된다고 설명한다. (Mr. Derson informs Person that all office communication is limited to email and official memos. #Person explains that this applies to both internal and external communications.)

Certainly! Here's the continuation of the markdown document, focusing on data noise and other relevant aspects:

---
## Data Noise

## Data Noise Examples

The dataset may contain various types of noise, including:

- **Newline Characters**: Appear as `\\n` within the dialogue text.
- **HTML Tags**: Such as `<br>` used instead of newlines to separate speakers.

These noise elements can affect both model training and evaluation, so it's crucial to handle them appropriately or design models robust to such inconsistencies.

---

## Insights and Data Processing

### Text Data Exploration

1. **Understanding Text Data**: 
   - Text data, unlike numerical data, can be challenging to analyze directly. By forming hypotheses and setting conditions, patterns can be identified. This dataset involves dialogues with distinct speaker identifiers, which can be investigated for topics and lengths.

2. **Examining Training Data**:
   - The training data consists of `dialogue` and `summary` pairs, each with a unique index stored in `fname`. Dialogues involve natural spoken language, while summaries are more formal.

3. **Preprocessing**:
   - Despite being well-organized, the dataset may include informal speech patterns that need addressing, such as slang or shorthand (e.g., "ㅋㅋ", "ㅇㅇ"). Methods for replacing or handling these should be considered.

### Visualization and Analysis

4. **Word Cloud**:
   - After data cleaning, frequently occurring words can be visualized to identify main topics. Although word frequency alone is useful, visualization offers a clearer insight into prominent themes.

5. **TF-IDF Analysis**:
   - This technique helps highlight significant words within specific documents by reducing the impact of common words across multiple documents.

### Privacy and Masking

6. **Personal Data Masking**:
   - The dataset masks personal information using patterns like `#PersonN#`. Regular expressions can help identify and handle these masked values efficiently.

7. **Special Tokens**:
   - Incorporating masked values as special tokens in tokenizers can improve model processing and understanding of the dialogue structure.

---

## Discussion and Model Improvement

- The exploration of text data allows for various insights and improvements in model performance. By identifying patterns and refining data processing techniques, participants can enhance their models' capabilities.

- Utilizing regular expressions and special token handling can improve the model's ability to understand and summarize dialogues effectively.

---