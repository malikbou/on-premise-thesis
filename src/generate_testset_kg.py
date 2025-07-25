#!/usr/bin/env python3
"""Generate a synthetic test-set for the UCL Student-Handbook RAG demo.

This script builds a Knowledge Graph from the source documents to generate a
more sophisticated test set with a higher proportion of multi-hop questions.

Example usage
-------------
python -m src.generate_testset_kg \
    --docs data/sts-student-handbook.pdf \
    --dept STS \
    --size 10

Environment
-----------
Requires an OPENAI_API_KEY in your environment. The script defaults to
`gpt-3.5-turbo-16k` for generation and `gpt-4o` for critique, but you can
override both.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import List, Optional

from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper, BaseRagasLLM
from ragas.embeddings import LangchainEmbeddingsWrapper, BaseRagasEmbeddings
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.synthesizers import QueryDistribution
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.multi_hop import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.persona import Persona


# Import graph construction components
from ragas.testset.transforms import (
    apply_transforms,
    Parallel,
    KeyphrasesExtractor,
    SummaryExtractor,
    EmbeddingExtractor,
)
from ragas.testset.transforms.extractors import NERExtractor
from ragas.testset.transforms.relationship_builders.cosine import (
    SummaryCosineSimilarityBuilder,
)


# Increased chunk size to avoid "documents too short" error
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_documents(path: str) -> List:
    """Load documents from a path (file or directory)."""
    if os.path.isdir(path):
        # Load all PDFs / MD / txt recursively
        loader = DirectoryLoader(path, glob="**/*.*")
    elif path.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path)
    return loader.load()


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Chunks documents based on semantic headings rather than fixed size.
    It assumes headings are numbered (e.g., "1. Introduction", "2.1 Key Policies")
    and uses these as boundaries for creating meaningful chunks.

    If no numbered headings are found, it falls back to the original fixed-size
    chunking method.
    """
    print("Attempting to chunk documents by semantic headings...")

    # 1. Combine all page content into a single string to handle sections
    # that span across pages.
    full_text = "\n\n".join(doc.page_content for doc in docs)

    # 2. Define a regex pattern to find numbered headings (e.g., 1. , 1.1 , 1.1.1)
    # This pattern looks for lines starting with digits, followed by a dot, a space,
    # and then the title.
    heading_pattern = re.compile(r"^\d+(\.\d+)*\s.*$", re.MULTILINE)

    # Find the start index of all headings
    heading_indices = [match.start() for match in heading_pattern.finditer(full_text)]

    # 3. If no headings are found, fall back to the original method.
    if not heading_indices:
        print("WARN: No numbered headings found. Falling back to fixed-size chunking.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        return splitter.split_documents(docs)

    # 4. Create chunks based on the text between headings.
    semantic_chunks = []
    for i in range(len(heading_indices)):
        # The start of the chunk is the start of the current heading
        start_index = heading_indices[i]
        # The end of the chunk is the start of the next heading, or the end of the document
        end_index = heading_indices[i+1] if i + 1 < len(heading_indices) else len(full_text)

        chunk_text = full_text[start_index:end_index].strip()

        if chunk_text:
            # Create a new Document for each semantic chunk.
            # We lose the specific page number but gain immense semantic value.
            # The source metadata is preserved from the first document.
            semantic_chunks.append(Document(
                page_content=chunk_text,
                metadata={"source": docs[0].metadata.get("source", "Unknown")}
            ))

    print(f"Successfully created {len(semantic_chunks)} semantic chunks.")
    return semantic_chunks


# ---------------------------------------------------------------------------
# Custom query distribution
# ---------------------------------------------------------------------------


# Define a new, stricter instruction that enforces high-quality questions.
# This instruction will be injected into the default Ragas prompts.
quality_instruction = """
Generate a question and a corresponding answer based on the provided context.

Your task is to act as a university administrator creating an official FAQ document. The questions you generate must be:
- **Grammatically flawless:** Adhere to perfect English grammar.
- **Clearly worded:** The question should be unambiguous and easy to understand.
- **Formal in tone:** Avoid slang, contractions, or informal language.
- **Directly answerable from the context:** The answer must be found verbatim in the provided text.

Do not invent any information. Do not include any misspellings or typos.
"""


def custom_query_distribution(
    llm: BaseRagasLLM,
    force_single_hop: bool = False,
    instruction: Optional[str] = None,
) -> QueryDistribution:
    """Create a custom query distribution with a mix of question types."""
    single_hop = SingleHopSpecificQuerySynthesizer(llm=llm)

    synthesizers = [single_hop]
    if not force_single_hop:
        multi_hop_specific = MultiHopSpecificQuerySynthesizer(llm=llm)
        multi_hop_abstract = MultiHopAbstractQuerySynthesizer(llm=llm)
        synthesizers.extend([multi_hop_specific, multi_hop_abstract])

    # If a custom instruction is provided, get the default prompt from each
    # synthesizer, overwrite its instruction, and set it back.
    if instruction:
        for synthesizer in synthesizers:
            try:
                prompt = synthesizer.get_prompts()["generate_query_reference_prompt"]
                prompt.instruction = instruction
                synthesizer.set_prompts({"generate_query_reference_prompt": prompt})
            except KeyError:
                print(f"WARN: Could not set custom prompt for {type(synthesizer).__name__}")

    if not force_single_hop:
        return [
            (synthesizers[0], 0.5),  # single_hop
            (synthesizers[1], 0.25), # multi_hop_specific
            (synthesizers[2], 0.25), # multi_hop_abstract
        ]
    else:
        return [(single_hop, 1.0)]


# ---------------------------------------------------------------------------
# Custom student-tone prompts disabled for now because the StringPrompt API
# in Ragas ≥0.2 has changed (no 'instruction' argument). The default prompts
# shipped with Ragas will be used instead. Uncomment and adapt using the new
# PydanticPrompt API if customised wording is required.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Ragas test-set from a handbook PDF."
    )
    parser.add_argument(
        "--docs", required=True, help="Path to PDF or directory with documents."
    )
    parser.add_argument(
        "--dept", default="STS", help="Dept slug used in output filename."
    )
    parser.add_argument(
        "--size", type=int, default=10, help="Number of questions to generate."
    )
    parser.add_argument("--generator-model", default="gpt-3.5-turbo-16k")
    parser.add_argument(
        "--force-single-hop",
        action="store_true",
        help="Force single-hop question generation only.",
    )
    args = parser.parse_args()

    print(f"Loading documents from {args.docs} …")
    raw_docs = load_documents(args.docs)
    print(f"Loaded {len(raw_docs)} source documents")

    # Optional chunking step (Ragas generator can take raw docs but chunking is safer)
    docs = chunk_documents(raw_docs)
    print(f"Chunked to {len(docs)} passages (≈{CHUNK_SIZE} chars each)")

    # Wrap OpenAI models for Ragas
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=args.generator_model, temperature=0))
    embedding_model = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # --- Testset Generation ---

    # Create UCL student personas
    persona_new_student = Persona(
        name="New Student",
        role_description="A first-year student looking for information about university policies and procedures. You write in clear, complete sentences with no slang or typos.",
    )
    persona_international = Persona(
        name="International Student",
        role_description="An international student seeking information about visa requirements, accommodation, and support services. You write in clear, complete sentences with no slang or typos.",
    )
    persona_graduate = Persona(
        name="Graduate Student",
        role_description="A graduate student interested in research opportunities, funding, and academic progression. You write in clear, complete sentences with no slang or typos.",
    )

    personas = [persona_new_student, persona_international, persona_graduate]

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=embedding_model,
        persona_list=personas,
    )

    # Define a RunConfig to control the parallelism of API calls
    run_config = RunConfig(max_workers=2) # A safer default

    if args.force_single_hop:
        print("Forcing single-hop question generation.")
        # Simplified transformations for single-hop
        transforms = [
            Parallel(
                KeyphrasesExtractor(llm=generator_llm),
                SummaryExtractor(llm=generator_llm),
                NERExtractor(llm=generator_llm),
            ),
        ]
        query_distribution = custom_query_distribution(
            generator_llm, force_single_hop=True, instruction=quality_instruction
        )
        dataset = generator.generate_with_langchain_docs(
            documents=docs,
            testset_size=args.size,
            transforms=transforms,
            query_distribution=query_distribution,
            run_config=run_config,
        )
    else:
        # Full pipeline with fallback for multi-hop
        print("Defining Knowledge Graph transformations for multi-hop attempt...")
        transforms = [
            Parallel(
                KeyphrasesExtractor(llm=generator_llm),
                SummaryExtractor(llm=generator_llm),
                NERExtractor(llm=generator_llm),
            ),
            EmbeddingExtractor(
                embedding_model=embedding_model,
                embed_property_name="summary",  # Embed summaries, not full content
                property_name="summary_embedding",  # Store embeddings in this property
            ),
            SummaryCosineSimilarityBuilder(
                threshold=0.7  # Lowered threshold to encourage relationship formation
            ),
        ]
        # Define ideal and fallback distributions
        ideal_distribution = custom_query_distribution(
            generator_llm, instruction=quality_instruction
        )
        fallback_distribution = custom_query_distribution(
            generator_llm, force_single_hop=True, instruction=quality_instruction
        )
        print("Generating test-set… this may take a few minutes.")
        try:
            print("Attempting to generate test set with multi-hop questions...")
            dataset = generator.generate_with_langchain_docs(
                documents=docs,
                testset_size=args.size,
                transforms=transforms,
                query_distribution=ideal_distribution,
                run_config=run_config,
            )
        except ValueError as e:
            if "No clusters found" in str(e):
                print(
                    "WARN: Could not generate multi-hop questions. Falling back to single-hop only."
                )
                # Re-run with simplified transforms for the fallback
                single_hop_transforms = [
                    Parallel(
                        KeyphrasesExtractor(llm=generator_llm),
                        SummaryExtractor(llm=generator_llm),
                        NERExtractor(llm=generator_llm),
                    ),
                ]
                dataset = generator.generate_with_langchain_docs(
                    documents=docs,
                    testset_size=args.size,
                    transforms=single_hop_transforms,
                    query_distribution=fallback_distribution,
                    run_config=run_config,
                )
            else:
                # Re-raise any other unexpected errors
                raise e

    out_dir = "testset"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.dept}_testset.json")

    with open(out_path, "w") as f:
        json.dump(dataset.to_list(), f, indent=2)
    print(f"Saved {len(dataset.to_list())} samples to {out_path}")


if __name__ == "__main__":
    main()
