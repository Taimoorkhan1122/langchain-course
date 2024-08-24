import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  ChatMessagePromptTemplate,
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import { config } from "dotenv";

config();

// const message = new HumanMessage("Tell me a joke!");
// this will create a message template.
// const promptTemplate = ChatPromptTemplate.fromTemplate("what is a good name for a {animal} pet?");

// const promptTemplate = ChatPromptTemplate.fromMessages([
//   SystemMessagePromptTemplate.fromTemplate("You are an expert in name assigning."),
//   HumanMessagePromptTemplate.fromTemplate("what is a good name for a {animal} pet?")
// ]);

const stringParsrer = new StringOutputParser();
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-1106",
});

const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "you are an expert in name assigning, suggest few names."],
  ["human", "suggest a good name for {animal} pet."],
]);

// await promptTemplate.formatMessages({
//   animal: "cat",
// })

// await promptTemplate.invoke({animal: "dog"})

// we can pipe multiple operation using pipe method for runnable sequences
const chain = promptTemplate.pipe(model).pipe(stringParsrer);

// OR we can pass a series of operaiton like this
const nameGenerationChain = RunnableSequence.from(
  [promptTemplate, model, stringParsrer],
  "test"
);

/**
 * INVOKE
 **/ 

// const response = await chain.invoke({ animal: "cat" });

/**
 * STREAM
 * we can also stream output results instead of waiting for the whole response 
 **/

// const response = await nameGenerationChain.stream({ animal: "elephant" });
// for  await (const data of response) {
//   console.log(data);
// }

const inputs = [
  {animal: "giraff"},
  {animal: "lion"},
]

const response = await nameGenerationChain.batch(inputs)

console.log(response);
