Based on the documents provided, here are some samples of the dialogue data.

### Sample 1: Health Check-up

This sample includes the full dialogue, a predicted summary, and the actual ("gold") summary.

* [cite_start]**Dialogue**[cite: 133]:
    * `#Person1#`: 안녕하세요, 스미스씨, 저는 호킨스 의사입니다. [cite_start]오늘 왜 오셨나요? [cite: 134] (Hello, Mr. Smith, I'm Dr. Hawkins. Why are you here today?)
    * [cite_start]`#Person2#`: 건강검진을 받는 것이 좋을 것 같아서요. [cite: 135] (I thought it would be good to get a check-up.)
    * `#Person1#`: 그렇군요, 당신은 5년 동안 건강검진을 받지 않았습니다. [cite_start]매년 받아야 합니다. [cite: 136] (I see, you haven't had a check-up in 5 years. You should have one every year.)
    * `#Person2#`: 알고 있습니다. [cite_start]하지만 아무 문제가 없다면 왜 의사를 만나러 가야 하나요? [cite: 137] (I know. But why should I go to the doctor if there are no problems?)
    * `#Person1#`: 심각한 질병을 피하는 가장 좋은 방법은 이를 조기에 발견하는 것입니다. [cite_start]그러니 당신의 건강을 위해 최소한 매년 한 번은 오세요. [cite: 138] (The best way to avoid serious illness is to detect it early. So please come at least once a year for your health.)
    * `#Person1#`: 여기 보세요. 당신의 눈과 귀는 괜찮아 보입니다. 깊게 숨을 들이쉬세요. [cite_start]스미스씨, 담배 피우시나요? [cite: 140] (Look here. Your eyes and ears look fine. Take a deep breath. Mr. Smith, do you smoke?)
    * [cite_start]`#Person2#`: 네. [cite: 141] (Yes.)
    * `#Person1#`: 당신도 알다시피, 담배는 폐암과 심장병의 주요 원인입니다. [cite_start]정말로 끊으셔야 합니다. [cite: 142] (As you know, smoking is a major cause of lung cancer and heart disease. You really need to quit.)
    * [cite_start]`#Person2#`: 수백 번 시도했지만, 습관을 버리는 것이 어렵습니다. [cite: 143] (I've tried hundreds of times, but it's hard to break the habit.)
    * `#Person1#`: 우리는 도움이 될 수 있는 수업과 약물들을 제공하고 있습니다. [cite_start]나가기 전에 더 많은 정보를 드리겠습니다. [cite: 144] (We offer classes and medications that can help. I'll give you more information before you leave.)
* [cite_start]**Gold Summary**: 스미스씨가 건강검진을 받고 있고, 호킨스 의사는 매년 건강검진을 받는 것을 권장합니다. [cite: 149] [cite_start]호킨스 의사는 스미스씨가 담배를 끊는 데 도움이 될 수 있는 수업과 약물에 대한 정보를 제공할 것입니다. [cite: 150] (Mr. Smith is getting a check-up, and Dr. Hawkins recommends getting one every year. Dr. Hawkins will provide information on classes and medication to help Mr. Smith quit smoking.)
* [cite_start]**Predicted Summary**: 호킨스 의사는 스미스씨에게 매년 건강검진을 받는 것이 심각한 질병을 조기에 발견하여 예방하는 데 중요하다고 조언합니다. [cite: 146] [cite_start]의사는 스미스씨의 눈과 귀를 검사하고, 폐암과 심장병의 위험을 증가시키는 흡연 습관을 끊을 것을 권장합니다. [cite: 147] [cite_start]의사는 스미스씨에게 도움을 줄 수 있는 수업과 약물에 대한 정보를 제공합니다. [cite: 148] (Dr. Hawkins advises Mr. Smith that getting an annual check-up is important for preventing serious diseases by detecting them early. The doctor checks Mr. Smith's eyes and ears and recommends he quit his smoking habit, which increases the risk of lung cancer and heart disease. The doctor provides Mr. Smith with information on classes and medications that can help.)

### Sample 2: Office Memo

This sample shows a dialogue about an internal company policy change.

* **Dialogue Snippet**:
    * [cite_start]`#Person1#`: 모든 직원들에게 주의하라... 즉시 효력을 발휘하여, 모든 사무실 동선은 이메일 통신과 공식 에모로 제한됩니다. [cite: 106] (Attention all employees... Effective immediately, all office communication is restricted to email correspondence and official memos.)
    * [cite_start]`#Person1#`: 근무 시간 동안 직원들이 즉시 메시지 프로그램을 사용하는 것은 엄격히 금지됩니다. [cite: 107] (The use of instant messaging programs by employees during work hours is strictly prohibited.)
    * `#Person2#`: 실장님, 이것은 내부 총신에만 적용되는 건가요? [cite_start]아니면 외부 통신에도 제한이 되는 건가요? [cite: 108] (Director, does this apply only to internal communications? Or is it restricted for external communications as well?)
    * `#Person1#`: 이것은 모든 통신에 적용되어야 합니다. [cite_start]이 사무실 내의 직원 사이뿐만 아니라 외부 통신에도 마찬가지입니다. [cite: 109] (This must apply to all communications. Not only between employees within this office, but also to external communications.)
* [cite_start]**Summary**: 더슨 씨는 Person에게 모든 사무실 통신이 어머 통신과 공식 메모 제한된다고 알려준다. [cite: 118] [cite_start]#Perso는 이것이 내부와 외부 통신에 적용된다고 설명한다. [cite: 118] (Mr. Derson informs Person that all office communication is limited to email and official memos. #Person explains that this applies to both internal and external communications.)

### Data Noise Examples

The dataset may also contain noise, such as:
* [cite_start]Newline characters appearing as `\\n` within the dialogue text[cite: 255].
* [cite_start]HTML tags like `<br>` used instead of newlines to separate speakers[cite: 257, 258, 259].