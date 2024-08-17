import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import {
  ChatMessagePromptTemplate,
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from "@langchain/core/prompts";
import { OpenAI } from "@langchain/openai";
import {config} from 'dotenv'

config()

// const message = new HumanMessage("Tell me a joke!");
// this will create a message template.
// const promptTemplate = ChatPromptTemplate.fromTemplate("what is a good name for a {animal} pet?");

// const promptTemplate = ChatPromptTemplate.fromMessages([
//   SystemMessagePromptTemplate.fromTemplate("You are an expert in name assigning."),
//   HumanMessagePromptTemplate.fromTemplate("what is a good name for a {animal} pet?")
// ]);

const model = new OpenAI({
  modelName: "gpt-3.5-turbo-1106",
});

const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "you are an expert in name assigning"],
  ["human", "suggest a good name for {animal} pet."],
]);

await promptTemplate.formatMessages({
  animal: "cat",
})


const chain = promptTemplate.pipe(model);
const response = await chain.invoke({animal: 'cat'});

console.log(response);


