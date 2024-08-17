## Language model
1. Text LLMs: string -> string
2. Chat models: list of messages -> single message output

## Prompt Template
Prompt templates help to translate user input and parameters into instructions for a language model. This can be used to guide a model's response, helping it understand the context and generate relevant and coherent language-based output.

There are a few different types of prompt templates:
1. String PromptTemplates
2. ChatPromptTemplates

## LangChain Expression Language (LCEL)

LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together.

### Langchain runnable:
A LangChain runnable is a protocol that allows you to create and invoke custom chains. It's designed to sequence tasks, taking the output of one call and feeding it as input to the next, making it suitable for straightforward, linear tasks where each step directly builds upon the previous one.

- for further reading refer to [langchain-expression-language](https://js.langchain.com/v0.2/docs/concepts#langchain-expression-language)

We can pipe multiple operation using pipe method for runnable sequences.

This is a standard interface, which makes it easy to define custom chains as well as invoke them in a standard way. The standard interface includes:
  - **stream**: stream back chunks of the response
  - **invoke**: call the chain on an input
  - **batch**: call the chain on an array of inputs
