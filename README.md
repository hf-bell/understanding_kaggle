# Understanding Kaggle 
A repo for me to document methods used to achieve the highest scores on Kaggle competitions, identifying how they were implemented for future use to myself.

## LLM 20 Questions

- **Competition**: [LLM 20 Questions](https://www.kaggle.com/competitions/llm-20-questions/overview)
- **1st place entry explanation**: [Discussion on 1st Place Entry](https://www.kaggle.com/competitions/llm-20-questions/discussion/531106)
- **Code**: [Dataset: LLM 20 Questions Final Submissions](https://www.kaggle.com/datasets/cnumber/llm-20-questions-final-submissions)

### Baseline Approach
For the baseline approach, refer to the [starter notebook](https://www.kaggle.com/code/ryanholbrook/llm-20-questions-starter-notebook).

### Models Built in PyTorch

#### Agent Formulation

##### Based on Google Gemma LLMs (from Gemini family)
- [Multimodal decoder-only transformers](https://arxiv.org/pdf/2312.11805)
- Use [sparse mixture-of-experts](https://arxiv.org/pdf/2312.17238) on feedforward layers of the decoder to improve inference speed.
- Uses the [Instruction-Tuned model](https://ai.google.dev/gemma/docs), since we are only looking at conversation.
- [GemmaCausalLM()](https://keras.io/api/keras_nlp/models/gemma/gemma_causal_lm/) — Using the `generate()` functionality to make the LLM generate from a question/answer prompt.

#### GemmaFormatter()
- A framework for dialogue between the 'questioner' and the 'answerer', drawing on [Gemma formatting docs](https://ai.google.dev/gemma/docs/formatting).
- Sets up the format for multi-turn interactions between the 'user' and 'model' as accepted by the Gemma LLM. Ensures all inputs are formatted correctly for processing and sets up an iterative cycle through turns in the dialogue.

##### Understanding `apply_turns()`
I initially wasn't sure how the `apply_turns()` method works. Here is a breakdown:

**Inputs**:
- `turns`: An iterable (e.g., list) of turns, where each turn represents an interaction (e.g., a question or answer).
- `start_agent`: A string indicating which agent starts the dialogue (`'model'` or `'user'`).

**Logic**:
- `formatters`: Determines the sequence of agents alternating turns based on the starting agent.
  - If `start_agent == 'model'`, the formatters list starts with `[self.model, self.user]`.
  - If `start_agent != 'model'`, the formatters list starts with `[self.user, self.model]`.
- `itertools.cycle(formatters)`: Creates an infinite cycle of the agents, ensuring that the sequence alternates correctly (model → user → model → user → ...).
- `zip(formatters, turns)`: Pairs each turn with the corresponding agent in the alternating sequence.
- `fmt(turn)`: Applies the formatting function (`self.model` or `self.user`) to the turn. Each agent handles the formatting of its own turn.

**Output**:
- Returns the modified instance of the `GemmaFormatter` class (`self`), allowing for method chaining.

The `apply_turns()` method is responsible for formatting the dialogue turns and generating a unified conversation string for the model.

- **Formatter State**: The formatter holds a state, which is a continuous string containing the history of dialogue turns, formatted correctly for Gemma.

#### GemmaAgent()
- `GemmaFormatter` is a parameter of each agent.
- Methods:
  - `_GemmaAgent()` — Calls the LLM using `call()` and inputs the state of the formatter (history of dialogue).
  - `_parse_response()`, `start_session()` — Placeholder methods for inheritance.
  - `_call_llm()` — Calls the `generate()` method of the `GemmaCausalLM` object, using parameters such as [temperature](https://www.iguazio.com/glossary/llm-temperature/), top_p, token probability, and token limit.
  - `_parse_keyword()` — Strips formatting from the answerer's guess and returns the 'guess keyword'.

#### GemmaQuestionerAgent (inherits from `GemmaAgent`)
- Not yet fully defined.

#### GemmaAnswererAgent (inherits from `GemmaAgent`)

---

## 1st Place Methods

### Questions
- Keyword selection
- Minimize expected entropy of questions

### Answers
- Additional explanation and methods related to answer generation and the decision-making process.
