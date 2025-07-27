### Competition Overview
(Summary)

[cite_start]The **Dialogue Summarization Competition** challenges participants to create models that can summarize everyday conversations on topics like school, work, healthcare, and travel[cite: 1, 3, 185, 187]. [cite_start]The competition runs from **July 25, 2025, to August 6, 2025**[cite: 7, 190].

---
### Dataset Details

[cite_start]The data is provided in `.csv` format and is structured with the following columns[cite: 202, 208]:
* [cite_start]**fname**: A unique ID for each conversation[cite: 234].
* [cite_start]**dialogue**: The text of the conversation, which involves between 2 and 7 speakers and consists of 2 to 60 turns[cite: 208, 235]. [cite_start]Speakers are identified with labels like `#Person1#`[cite: 235].
* [cite_start]**summary**: The reference summary for the dialogue[cite: 235].

[cite_start]The dataset includes various types of **noise**, such as typos, inconsistent punctuation, speaker labels, and formatting issues like `\\n` characters or `<br>` tags[cite: 250, 251, 255, 257].

The data is split into:
* [cite_start]**Test Set**: 250 dialogues[cite: 204].
* [cite_start]**Hidden Test Set**: 249 dialogues[cite: 205].

---
### Modeling and Baseline

[cite_start]A baseline solution is provided using **KoBART**, a model based on BART that is specifically tailored for the Korean language[cite: 50]. The baseline code is organized into several key files:
* [cite_start]`baseline.ipynb`: Contains the primary code for data processing, model training, and inference[cite: 24, 29].
* [cite_start]`config.yaml`: A configuration file for model parameters[cite: 25, 28].
* [cite_start]`solar_api.ipynb`: An alternative approach that demonstrates how to use the Solar Chat API for summarization, including prompt engineering techniques[cite: 27, 119, 120].

[cite_start]The training process involves preprocessing the data for the model's encoder and decoder and then using the Hugging Face `Trainer` class for training[cite: 30, 42, 43].

---
### Evaluation

[cite_start]The performance of the summarization models is evaluated using the **ROUGE score**[cite: 46, 124]. [cite_start]To ensure a fair and comprehensive assessment, each dialogue in the test set is evaluated against **three different reference summaries**[cite: 242]. [cite_start]This approach acknowledges that a single conversation can have multiple valid summaries[cite: 241, 242]. [cite_start]The baseline model achieved a **ROUGE-F1 score of 47.1244** on the public test data[cite: 176, 177].

----
```text
(Original content)
Introduction
학교 생활, 직장, 치료, 쇼핑, 여가, 여행 등 광범위한 일상 생활 중 하는 대화들에 대해 요약합니다.

Dialogue Summarization 경진대회는 주어진 데이터를 활용하여 일상 대화에 대한 요약을 효과적으로 생성하는 모델을 개발하는 대회입니다. 

일상생활에서 대화는 항상 이루어지고 있습니다. 회의나 토의는 물론이고, 사소한 일상 대화 중에도 서로 다양한 주제와 입장들을 주고 받습니다. 나누는 대화를 녹음해두더라도 대화 전체를 항상 다시 들을 수는 없기 때문에 요약이 필요하고, 이를 위한 통화 비서와 같은 서비스들도 등장하고 있습니다.

그러나 하나의 대화에서도 관점, 주제별로 정리하면 수 많은 요약을 만들 수 있습니다. 대화를 하는 도중에 이를 요약하게 되면 대화에 집중할 수 없으며, 대화 이후에 기억에 의존해 요약하게 되면 오해나 누락이 추가되어 주관이 많이 개입되게 됩니다.

이를 돕기 위해, 우리는 이번 대회에서 일상 대화를 바탕으로 요약문을 생성하는 모델을 구축합니다!

참가자들은 대회에서 제공된 데이터셋을 기반으로 모델을 학습하고, 대화의 요약문을 생성하는데 중점을 둡니다. 이를 위해 다양한 구조의 자연어 모델을 구축할 수 있습니다.

제공되는 데이터셋은 오직 "대화문과 요약문"입니다. 회의, 일상 대화 등 다양한 주제를 가진 대화문과, 이에 대한 요약문을 포함하고 있습니다.

참가자들은 이러한 비정형 텍스트 데이터를 고려하여 모델을 훈련하고, 요약문의 생성 성능을 높이기 위한 최적의 방법을 찾아야 합니다.

경진대회의 목표는 정확하고 일반화된 모델을 개발하여 요약문을 생성하는 것입니다. 나누는 많은 대화에서 핵심적인 부분만 모델이 요약해주니, 업무 효율은 물론이고 관계도 개선될 수 있습니다. 또한, 참가자들은 모델의 성능을 평가하고 대화문과 요약문의 관계를 심층적으로 이해함으로써 자연어 딥러닝 모델링 분야에서의 실전 경험을 쌓을 수 있습니다.

본 대회는 결과물 csv 확장자 파일을 제출하게 됩니다.

input : 249개의 대화문

output : 249개의 대화 요약문
```
----