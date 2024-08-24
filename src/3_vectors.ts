import "dotenv/config";
import * as pdfParse from "pdf-parse";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";


  // create an embedding
  const embeddings = new OpenAIEmbeddings();
  // const query = await embeddings.embedQuery("what is cyber security?");

  // load the document

  const pdfLoader = new PDFLoader("./docs/machineLearning-lecture01.pdf");
  const doc = await pdfLoader.load();

  // initiate the splitter and split the doc to fit the context window of LLM
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 128,
    chunkOverlap: 0,
  });
  const splitDoc = await splitter.splitDocuments(doc);

  // // load in memory vector store
  const vectorStore = new MemoryVectorStore(embeddings);
  await vectorStore.addDocuments(splitDoc);

  const retrievedDocs = await vectorStore.similaritySearch("what is the topic of the lecture?", 4);

  console.log(retrievedDocs.map(d => d.pageContent));
