# Understanding Kaggle 
A repo for me to document methods used to achieve the highest scores on Kaggle competitions, identifying how they were implemented for future use to myself

# LLM 20 Questions

Competition: https://www.kaggle.com/competitions/llm-20-questions/overview

1st place entry explanation: https://www.kaggle.com/competitions/llm-20-questions/discussion/531106
Code: https://www.kaggle.com/datasets/cnumber/llm-20-questions-final-submissions

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
- 'user' and 'model' - what are these? 
- 
#### GemmaAgent()
#### GemmaQuestionerAgent(GemmaAgent)
#### GemmaAnswererAgent(GemmaAgent)

## 1st place methods 

### Questions
- Keyword selection 
- Minimise expected entropy of questions
  
### Answers 

