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
from typing import List

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper, BaseRagasLLM
from ragas.embeddings import LangchainEmbeddingsWrapper, BaseRagasEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.synthesizers import QueryDistribution
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
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


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# Custom query distribution
# ---------------------------------------------------------------------------


def custom_query_distribution(
    llm: BaseRagasLLM, force_single_hop: bool = False
) -> QueryDistribution:
    """Create a custom query distribution with a mix of question types."""
    single_hop_specific = SingleHopSpecificQuerySynthesizer(llm=llm)

    if not force_single_hop:
        multi_hop_specific = MultiHopSpecificQuerySynthesizer(llm=llm)
        multi_hop_abstract = MultiHopAbstractQuerySynthesizer(llm=llm)
        return [
            (single_hop_specific, 0.5),
            (multi_hop_specific, 0.25),
            (multi_hop_abstract, 0.25),
        ]
    else:
        return [(single_hop_specific, 1.0)]


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
    parser = argparse.ArgumentParser(description="Generate a synthetic Ragas test-set from a handbook PDF.")
    parser.add_argument("--docs", required=True, help="Path to PDF or directory with documents.")
    parser.add_argument("--dept", default="STS", help="Dept slug used in output filename.")
    parser.add_argument("--size", type=int, default=10, help="Number of questions to generate.")
    parser.add_argument("--generator-model", default="gpt-3.5-turbo-16k")
    parser.add_argument("--critic-model", default="gpt-4o")
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

    # --- Define Knowledge Graph Transformations ---
    print("Defining Knowledge Graph transformations…")
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

    # --- Testset Generation ---

    # Create UCL student personas
    persona_new_student = Persona(
        name="New Student",
        role_description="A first-year student looking for information about university policies and procedures."
    )
    persona_international = Persona(
        name="International Student",
        role_description="An international student seeking information about visa requirements, accommodation, and support services."
    )
    persona_graduate = Persona(
        name="Graduate Student",
        role_description="A graduate student interested in research opportunities, funding, and academic progression."
    )

    personas = [persona_new_student, persona_international, persona_graduate]

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=embedding_model,
        persona_list=personas,
    )

    # Define ideal and fallback distributions
    ideal_distribution = custom_query_distribution(generator_llm)
    fallback_distribution = custom_query_distribution(generator_llm, force_single_hop=True)

    print("Generating test-set… this may take a few minutes.")
    try:
        print("Attempting to generate test set with multi-hop questions...")
        dataset = generator.generate_with_langchain_docs(
            documents=docs,
            testset_size=args.size,
            transforms=transforms,
            query_distribution=ideal_distribution,
        )
    except ValueError as e:
        if "No clusters found" in str(e):
            print("WARN: Could not generate multi-hop questions. Falling back to single-hop only.")
            dataset = generator.generate_with_langchain_docs(
                documents=docs,
                testset_size=args.size,
                transforms=transforms,
                query_distribution=fallback_distribution,
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
