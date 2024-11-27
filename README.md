# Understanding Kaggle 
A repo for me to document methods used to achieve the highest scores on Kaggle competitions, identifying how they were implemented for future use to myself

# LLM 20 Questions

Competition: https://www.kaggle.com/competitions/llm-20-questions/overview

1st place entry explanation: https://www.kaggle.com/competitions/llm-20-questions/discussion/531106
<br/> Code: https://www.kaggle.com/datasets/cnumber/llm-20-questions-final-submissions

## Baseline approach 
See _https://www.kaggle.com/code/ryanholbrook/llm-20-questions-starter-notebook_

Models built in PyTorch
### Agent formulation 

#### Based on Google Gemma LLMs (from Gemini family)
- [Multimodal decoder-only transformers](https://arxiv.org/pdf/2312.11805)
-  Use [sparse mixture-of-experts](https://arxiv.org/pdf/2312.17238) on feedforward layers of decoder to improve inference speed
-  Uses [Instruction-Tuned model](https://ai.google.dev/gemma/docs), as we are only looking at conversation
-  [GemmaCausalLM()](https://keras.io/api/keras_nlp/models/gemma/gemma_causal_lm/) -- we are going to use the _generate()_ functionality to make the LM generate from a question / answer prompt 

#### GemmaFormatter()
- Framework for dialogue between the 'questioner' and the 'answerer', drawing on [this](https://ai.google.dev/gemma/docs/formatting) 
- Sets up format of multi-turn based interactions between a 'user' and a 'model' as accepted by the Gemma LLM. Ensures all inputs to LLM are formatted correctly for processing, and also sets up iterative cycle through turns in dialogue.
- Initially I wasn't sure about how _apply_turns()_ works. Here is a breakdown:
</br> **Inputs:**

> turns: An iterable (e.g., list) of turns, where each turn represents an interaction (e.g., a question or answer).
> </br> start_agent: A string indicating which agent starts the dialogue ('model' or 'user').
</br> **Logic:**

</br> formatters:

> Determines the sequence of agents (model and user) alternating turns based on the starting agent.
> If start_agent == 'model', the formatters list starts with [self.model, self.user].
>If start_agent != 'model', the formatters list starts with [self.user, self.model].
> itertools.cycle(formatters):

Creates an infinite cycle of the agents, ensuring that the sequence alternates correctly (model → user → model → user → ...).
zip(formatters, turns):

Pairs each turn with the corresponding agent in the alternating sequence.
fmt(turn):

Applies the formatting function (either self.model or self.user) to the turn. Each agent handles the formatting of its own turn.
Output:

Returns the modified instance of the GemmaFormatter class (self), allowing for method chaining.



The question asked by the guesser.
The corresponding answer provided by the answerer.
Formatting: For each turn, the method creates a textual representation. For example:

"Guesser: What color is it?\nAnswerer: Red." This format ensures clarity in the dialogue and makes it intuitive for language models.
Concatenation: The method iterates through the list of turns, concatenating each formatted string into a single conversation.

Output: The result is a unified string that encapsulates all turns of the game session, which the model can directly consume for training or inference.


- Formatter has a state -- which is a continuous string containing the history of dialogue turns, formatted correct for Gemma
#### GemmaAgent()
- GemmaFormatter is a parameter of each Agent 
- Setup Gemma agent call & download parameters
- Methods:
-   _GemmaAgent()_ -- call() implementation: Input state of formatter (history of dialogue) into LLM and generate response
-   _parse_response_; start_session() -- empty methods for inheritance
-   _call_llm_ -- calls generate() method of GemmaCausalLLM object, using [temperature](https://www.iguazio.com/glossary/llm-temperature/), top_p (threshold cumulative probability for a set of tokens to be generated from LLM); (threshold token probability at each timestep) keywords + a limit on number of tokens to be generated. Generates the LLM output for this turn. 
-   _parse_keyword_ -- Strips formatting from answerer guess & returns 'guess keyword'
  
#### GemmaQuestionerAgent(GemmaAgent)
- d
- 
#### GemmaAnswererAgent(GemmaAgent)

## 1st place methods 

### Questions
- Keyword selection 
- Minimise expected entropy of questions
  
### Answers 

