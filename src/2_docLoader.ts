import { GithubRepoLoader } from "@langchain/community/document_loaders/web/github";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
// import * as parse from 'pdf-parse'
import ignore from "ignore";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// const loader = new GithubRepoLoader("https://github.com/langchain-ai/langchainjs", {
//   recursive: false,
//   ignorePaths: ["*.md", "yarn.lock"],
// });

const loader = new PDFLoader('./docs/testing.pdf');

const docs = await loader.load();
// console.log(docs.slice(0,3))

const splitter = RecursiveCharacterTextSplitter.fromLanguage("js", {
  chunkSize: 512,
  chunkOverlap: 32,
  separators: [" ", "\n"]
});

// const splitCode = await splitter.splitText(`const loader = new PDFLoader('./docs/testing.pdf');

// const docs = await loader.load();
// console.log(docs.slice(0,3))
// `);

const splitDocs = await splitter.splitDocuments(docs)
console.log(splitDocs)
