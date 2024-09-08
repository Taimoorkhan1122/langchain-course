import "dotenv/config";
import { Document } from "langchain/document";
import { initializeVectorstoreWithDocuments, loadAndSplitChunk } from "./helpers";
import { RunnableSequence, RunnableMap } from "@langchain/core/runnables";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatOpenAI } from "@langchain/openai";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { RunnablePassthrough } from "@langchain/core/runnables";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";

// split document in appropriate chunk size
const splitDocs = await loadAndSplitChunk({ chunkSize: 1536, chunkOverlap: 128 });
// create embedding for splitted docs
const vectorStore = await initializeVectorstoreWithDocuments(splitDocs);
// convert the vector store as  expression language chainable
const retriever = vectorStore.asRetriever();

const convertDocsToString = (docs: Document[]): string => {
  return docs.map((d) => `<doc>\n ${d.pageContent}</doc>\n`).join("\n");
};

// this chain retrieves chunks of document separated by <doc> tags
const documentRetrievalChain = RunnableSequence.from([
  (input) => input.question,
  retriever,
  convertDocsToString,
]);

// prompt template with placeholders for variables
const promptTemplate = `You are an experienced researcher, 
expert at interpreting and answering questions based on provided sources.
Using the provided context, answer the user's question 
to the best of your ability using only the resources provided. 
Be verbose!

<context>

{context}

</context>

Now, answer this question using the above context:

{question}
`;

/**
 * Format a series of messages for a conversation.
 * Load prompt template from a template string
 * */
const answerGenerationPrompt = ChatPromptTemplate.fromTemplate(promptTemplate);

const model = new ChatOpenAI({
  model: "gpt-3.5-turbo-1106",
});

/**
 * our documentRetrievalChain returns a string, but runnable sequence requires an object.
 * so we use runnableMap to  execute multiple Runnables in parallel,
 * and to return the output of these Runnables
 **/
const runnableMap = RunnableMap.from({
  context: documentRetrievalChain,
  question: (input: any) => input.question,
});

const retrievalChain = RunnableSequence.from([
  {
    context: documentRetrievalChain,
    question: (input: any) => input.question,
  },
  answerGenerationPrompt,
  model,
  new StringOutputParser(),
]);

// const response = await retrievalChain.invoke({
//   question: "what are prerequisite for this course?",
// });

// const followUp = await retrievalChain.invoke({
//   question: "can you list them in bullet points form?",
// });

// console.log({ response, followUp });

// ==== Adding History ====

const REPHRASE_QUESTION_SYSTEM_TEMPLATE = `Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.`;

const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  ["human", "Rephrase the following question as a standalone question:\n{question}"],
]);

// rephrase the given question an generate a standalone question.
const rephraseQuestionChain = RunnableSequence.from([
  rephraseQuestionChainPrompt,
  new ChatOpenAI({ temperature: 0.1, modelName: "gpt-3.5-turbo-1106" }),
  new StringOutputParser(),
]);

const ANSWER_CHAIN_SYSTEM_TEMPLATE = `You are an experienced researcher, 
expert at interpreting and answering questions based on provided sources.
Using the below provided context and chat history, 
answer the user's question to the best of 
your ability 
using only the resources provided. Be verbose!

<context>
{context}
</context>`;

const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", ANSWER_CHAIN_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  [
    "human",
    "Now, answer this question using the previous context and chat history:\n{standalone_question}",
  ],
]);

const conversationalRetrievalChain = RunnableSequence.from([
  RunnablePassthrough.assign({
    standalone_question: rephraseQuestionChain,
  }),
  RunnablePassthrough.assign({
    context: documentRetrievalChain,
  }),
  answerGenerationChainPrompt,
  new ChatOpenAI({ modelName: "gpt-3.5-turbo" }),
  new StringOutputParser(),
]);

const messageHistory = new ChatMessageHistory();

const finalRetrievalChain = new RunnableWithMessageHistory({
  runnable: conversationalRetrievalChain,
  getMessageHistory: (_sessionId) => messageHistory,
  historyMessagesKey: "history",
  inputMessagesKey: "question",
});

const originalQuestion = "What are the prerequisites for this course?";

const originalAnswer = await finalRetrievalChain.invoke(
  {
    question: originalQuestion,
  },
  {
    configurable: { sessionId: "test" },
  }
);

const finalAnswer = await finalRetrievalChain.invoke(
  {
    question: "Can you list them in bullet point form?",
  },
  {
    configurable: { sessionId: "test" },
  }
);

console.log({originalAnswer, finalAnswer})
