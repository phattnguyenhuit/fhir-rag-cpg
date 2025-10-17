import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any

# Updated LangChain imports (post-1.0 structure)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()


class FHIRRAGSystem:
    def __init__(self, fhir_file_path: str, persist_directory: str = "./chroma_db"):
        """Initialize the FHIR RAG system."""
        self.fhir_file_path = fhir_file_path
        self.persist_directory = persist_directory
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.vectorstore = None
        self.qa_chain = None

    # -------------------------------------------------------------------------
    # FHIR processing
    # -------------------------------------------------------------------------
    def load_fhir_bundle(self) -> Dict[str, Any]:
        """Load FHIR bundle from JSON file."""
        try:
            with open(self.fhir_file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"FHIR file not found at {self.fhir_file_path}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in FHIR file")

    def fhir_bundle_to_texts(self, bundle_json: Dict[str, Any]) -> List[str]:
        """
        Convert FHIR Bundle to structured text for embedding.
        Handles PlanDefinition, ActivityDefinition, Medication, Library, etc.
        """
        texts = []

        for entry in bundle_json.get("entry", []):
            res = entry.get("resource", {})
            resource_type = res.get("resourceType", "Unknown")
            parts = [f"=== {resource_type} ==="]

            # Common metadata
            if "id" in res:
                parts.append(f"ID: {res['id']}")
            if "title" in res:
                parts.append(f"Title: {res['title']}")
            if "name" in res:
                parts.append(f"Name: {res['name']}")
            if "status" in res:
                parts.append(f"Status: {res['status']}")
            if "description" in res:
                parts.append(f"Description: {res['description']}")

            # Resource-specific logic
            if resource_type == "PlanDefinition":
                parts.extend(self._extract_plan_definition(res))
            elif resource_type == "ActivityDefinition":
                parts.extend(self._extract_activity_definition(res))
            elif resource_type == "Medication":
                parts.extend(self._extract_medication(res))
            elif resource_type == "Library":
                parts.extend(self._extract_library(res))

            text_block = "\n".join([p for p in parts if p.strip()])
            if text_block:
                texts.append(text_block)

        return texts

    # -------------------------------------------------------------------------
    # Extraction helpers
    # -------------------------------------------------------------------------
    def _extract_plan_definition(self, res: Dict[str, Any]) -> List[str]:
        """Extract structured information from PlanDefinition."""
        parts = []
        for idx, action in enumerate(res.get("action", []), 1):
            title = action.get("title", f"Action {idx}")
            desc = action.get("description", "")
            parts.append(f"\nAction {idx}: {title}")
            if desc:
                parts.append(f"  Description: {desc}")

            for cond in action.get("condition", []):
                kind = cond.get("kind", "")
                expr = cond.get("expression", {})
                if expr:
                    parts.append(f"  Condition ({kind}): {expr.get('description', '')}")

            if "timingTiming" in action:
                parts.append(f"  Timing: {action['timingTiming']}")

            for sub in action.get("action", []):
                parts.append(f"  Sub-action: {sub.get('title', '')} - {sub.get('description', '')}")
        return parts

    def _extract_activity_definition(self, res: Dict[str, Any]) -> List[str]:
        """Extract structured info from ActivityDefinition."""
        parts = [f"Kind: {res.get('kind', 'N/A')}"]
        if "code" in res:
            code = res["code"]
            parts.append(f"Code: {code.get('text', '')} ({code.get('coding', [{}])[0].get('code', '')})")

        if "dosage" in res:
            for dosage in res["dosage"]:
                parts.append(f"Dosage: {dosage.get('text', '')}")
        return parts

    def _extract_medication(self, res: Dict[str, Any]) -> List[str]:
        """Extract medication info."""
        parts = []
        if "code" in res:
            code = res["code"]
            parts.append(f"Medication Code: {code.get('text', '')}")
        if "ingredient" in res:
            for ing in res["ingredient"]:
                parts.append(f"Ingredient: {ing}")
        return parts

    def _extract_library(self, res: Dict[str, Any]) -> List[str]:
        """Extract library content info."""
        parts = []
        if "content" in res:
            for content in res["content"]:
                parts.append(f"Content Type: {content.get('contentType', '')}")
        return parts

    # -------------------------------------------------------------------------
    # Vectorstore creation
    # -------------------------------------------------------------------------
    def build_vectorstore(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Build or load the vector store from FHIR data."""
        if os.path.exists(self.persist_directory):
            print(f"Loading existing vectorstore from {self.persist_directory}")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings
            )
            return

        print("Building new vectorstore...")
        data = self.load_fhir_bundle()
        texts = self.fhir_bundle_to_texts(data)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n=== ", "\nAction", "\n", " "]
        )

        docs = [Document(page_content=t) for t in texts]
        chunks = splitter.split_documents(docs)

        print(f"Created {len(chunks)} chunks from {len(texts)} FHIR resources")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()
        print("✅ Vectorstore built and persisted successfully.")

    # -------------------------------------------------------------------------
    # QA Chain
    # -------------------------------------------------------------------------
    def setup_qa_chain(self, model_name: str = "gpt-4o-mini", k: int = 5):
        """Set up the retrieval-augmented QA chain."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call build_vectorstore() first.")

        prompt_template = """You are a medical AI assistant specializing in FHIR-based Clinical Practice Guidelines.
Use ONLY the provided guideline context to answer the question.
If the answer is not contained in the context, clearly say "The guideline does not specify."

Context:
{context}

Question: {question}

Provide a concise, evidence-based answer (include timing, dosage, or conditions when relevant).

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        llm = ChatOpenAI(model_name=model_name, temperature=0)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k}, search_type="similarity")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    # -------------------------------------------------------------------------
    # Query Interface
    # -------------------------------------------------------------------------
    def query(self, question: str, show_sources: bool = True) -> Dict[str, Any]:
        """Query the guideline knowledge base."""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup_qa_chain() first.")

        result = self.qa_chain.invoke({"query": question})

        response = {"answer": result["result"], "sources": []}
        if show_sources:
            for i, doc in enumerate(result.get("source_documents", []), 1):
                response["sources"].append({
                    "source_id": i,
                    "content": (
                        doc.page_content[:500] + "..."
                        if len(doc.page_content) > 500 else doc.page_content
                    )
                })
        return response


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    rag_system = FHIRRAGSystem(
        fhir_file_path=r"D:\HealthCare_ChatBot\fhir-rag-cpg\data\huong-dan-dot-quy-nhe.txt"
    )

    rag_system.build_vectorstore()
    rag_system.setup_qa_chain(model_name="gpt-4o-mini", k=5)

    queries = [
       "đột quỵ là gì",
    ]

    for query in queries:
        print(f"\n{'=' * 80}")
        print(f"Question: {query}")
        print(f"{'=' * 80}")

        result = rag_system.query(query, show_sources=True)
        print(f"\nAnswer: {result['answer']}")

        if result['sources']:
            print("\nRelevant Sources:")
            for source in result['sources']:
                print(f"\n[Source {source['source_id']}]")
                print(source['content'])
