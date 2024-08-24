import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { Document } from "langchain/document";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

interface ISplitInput {
  chunkSize: number;
  chunkOverlap: number;
}

export const loadAndSplitChunk = async ({
  chunkSize,
  chunkOverlap,
}: ISplitInput): Promise<Document[]> => {
  const pdfLoader = new PDFLoader("./docs/machineLearning-lecture01.pdf");
  const doc = await pdfLoader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap,
  });

  const splitDoc = await splitter.splitDocuments(doc);

  return splitDoc;
};

export const initializeVectorstoreWithDocuments = async (docs: Document[]) => {
  const vectorStore = new MemoryVectorStore(new OpenAIEmbeddings());
  await vectorStore.addDocuments(docs);

  return vectorStore;
};
