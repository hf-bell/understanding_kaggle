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
- 

### Code Explained

**Agent Alpha** is a specialized agent designed to optimize the questioning process in the "20 Questions" game. This agent uses a set of predefined keywords with associated priority values to minimize entropy (uncertainty) and ask the most informative questions possible. Below is a breakdown of how the agent works, its strategy for minimizing entropy, and how it differs from other agents (such as Agent Beta).

## Key Concepts in `agent_alpha`

### 1. **List of Keywords with Priorities**
The agent starts with a list of potential keywords (objects) that the answerer might be thinking of. These keywords are associated with priority values, indicating the relative importance or likelihood of each keyword. The list is loaded from a file (`list_keywords_all_with_input_with_priority.pkl`), sorted alphabetically, and used throughout the game.

- **Function**: `get_list_keywords_with_priority()`
- **Purpose**: Loads and sorts the keywords list by priority.

### 2. **Sifting Through Keywords Based on Previous Questions and Answers**
The agent processes previous questions and answers to refine the list of possible keywords. Specifically, it looks for questions of the form:

"Does the keyword (in lowercase) precede [some word] in alphabetical order?"

vbnet
Copy code

- If such a question was asked, the agent narrows down the list of keywords by slicing it before or after the testword (the word being compared in the question) based on the answer ("yes" or "no").
- This **filters out irrelevant keywords**, helping the agent focus on more likely candidates.

- **Function**: `sieve_list_keywords_with_priority()`
- **Purpose**: Refines the list of keywords by processing past questions and answers.

### 3. **Choosing the Middle Keyword**
After refining the list of possible keywords, the agent selects a "midpoint" keyword based on the sum of all keyword priorities. The goal is to select a keyword that splits the remaining possibilities in half, thereby minimizing entropy and maximizing information gain.

- **Function**: `get_mid_testword()`
- **Purpose**: Selects the keyword whose cumulative priority reaches the midpoint of the remaining list. This is done to ensure the next question splits the possibilities evenly.

### 4. **Question Generation**
Depending on the current turn type, the agent either generates a question or makes a guess:
- **"Ask" turn**: The agent constructs a question asking if the target keyword precedes a selected testword in alphabetical order.
- **"Guess" turn**: The agent guesses the selected keyword.

- **Purpose**: To reduce the search space and either gather more information or make a correct guess.

### 5. **Strategy to Minimize Entropy**
The core principle of minimizing entropy is to ask questions that split the search space as evenly as possible. **Agent Alpha** does this by:
- Narrowing down the list of possible keywords after each question based on previous answers.
- Selecting the "midpoint" keyword based on cumulative priority, ensuring that the remaining search space is balanced.

This strategy reduces uncertainty and helps the questioner find the correct object more quickly, making **Agent Alpha** highly efficient in reducing the number of required questions.

## Entropy Minimization in the Code

Minimizing entropy in information theory refers to reducing uncertainty in the system. A good question should help eliminate as many possibilities as possible. **Agent Alpha** minimizes entropy by:
1. **Refining the list of possible keywords**: After each question, the agent updates the list of remaining keywords based on the answers.
2. **Selecting a midpoint keyword**: The agent picks a keyword that splits the remaining possibilities based on priority, aiming to balance the remaining search space.
3. **Maximizing information gain**: Each question is designed to provide the most information, reducing the set of remaining possible keywords efficiently.

### Key Code Methods in Action

#### `get_list_keywords_with_priority()`
- Loads the list of possible keywords and their priorities from a file.
- Sorts the list alphabetically.

#### `sieve_list_keywords_with_priority()`
- Processes past answers to refine the list of possible keywords.
- Eliminates irrelevant keywords based on previous "yes" or "no" answers.

#### `get_mid_testword()`
- Calculates the midpoint keyword based on the sum of priorities.
- Selects the keyword that splits the remaining search space evenly.

#### `agent_alpha()`
- The main function that orchestrates the questioning process.
- Uses the above methods to generate questions or guesses based on the game state.

## Overall Strategy

**Agent Alpha** employs an **information-driven** strategy for the "20 Questions" game:
- It uses a priority list of possible keywords to focus on the most likely objects.
- It minimizes entropy by narrowing down the possibilities with each question, ensuring that each question provides maximum information.

This targeted approach allows **Agent Alpha** to guess the correct object with fewer questions compared to a random or less informed strategy.

## How It Differs from Agent Beta

- **Agent Alpha** is more **specialized** and **efficient** in its questioning strategy. It uses a priority-based list of keywords and asks questions that are designed to minimize entropy.
- **Agent Beta** (likely found in `beta.py`) is a simpler, less targeted agent. It may ask more general or random questions and doesn't rely on a priority list or sophisticated entropy minimization strategy.

## Conclusion

**Agent Alpha** is an advanced agent that leverages keyword selection and entropy minimization to efficiently play the "20 Questions" game. By narrowing down the possibilities based on previous answers and selecting the most informative questions, it is able to guess the correct object in fewer questions. This makes it a highly effective agent for the game.
