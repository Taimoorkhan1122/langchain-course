import "dotenv/config";
import { Document } from "langchain/document";
import { initializeVectorstoreWithDocuments, loadAndSplitChunk } from "./helpers";
import { RunnableSequence, RunnableMap } from "@langchain/core/runnables";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatOpenAI } from "@langchain/openai";

// split document in appropriate chunk size
const splitDocs = await loadAndSplitChunk({ chunkSize: 1536, chunkOverlap: 125 });
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
  runnableMap,
  answerGenerationPrompt,
  model,
  new StringOutputParser(),
]);

const response = await retrievalChain.invoke({
  question: "Name one machine learning algorithm",
});

console.log(response);
