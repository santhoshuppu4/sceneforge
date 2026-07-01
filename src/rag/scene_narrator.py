"""
SceneForge RAG Scene Narrator

LangChain + ChromaDB: embeds domain ontologies (forensic / clinical) as
retrieval context, then uses an LLM to generate plain-English scene descriptions
from raw detection output.

Falls back to a rule-based narrator when no API key is present — useful for
local dev and demo without OpenAI costs.
"""

from __future__ import annotations

import logging
import os
from typing import List

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DOMAIN ONTOLOGIES
# ─────────────────────────────────────────────
FORENSIC_ONTOLOGY = """
Forensic evidence categories:
- Biological: blood stains, hair, skin cells, fingerprints, saliva
- Documentary: papers, notebooks, envelopes, phones, laptops, USB drives
- Physical: weapons (knives, firearms, blunt objects), tools, containers, bags
- Trace: fibres, glass fragments, soil, chemical residues

Crime scene interpretation rules:
- Partially occluded objects (occlusion_score > 0.3) are high priority — may be intentionally concealed
- Objects under other objects indicate scene disturbance or deliberate hiding
- Amodal bbox (full predicted extent) helps estimate hidden portion size
- Object position relative to entry/exit points is significant
- Electronics must be documented before touching to preserve metadata
"""

CLINICAL_ONTOLOGY = """
Clinical environment object categories:
- Instruments: scalpels, forceps, scissors, retractors, clamps, needles
- Monitoring: ECG leads, pulse oximeter, blood pressure cuff, IV lines
- Medications: syringes, vials, blister packs, IV bags, pill bottles, ampoules
- Consumables: gloves, masks, gauze, bandages, catheters, swabs, drapes
- Equipment: defibrillator, ventilator, infusion pump, suction unit

Clinical urgency rules:
- Medications near patients require verification (5 rights of medication)
- Sharp instruments must be counted before and after every procedure
- Partially occluded syringes (occlusion_score > 0.3) are high-priority safety items
- IV lines and catheters: check for kinks, disconnections, air bubbles
- Equipment cables occluded by other cables increase disconnection risk
"""

GENERAL_ONTOLOGY = """
Indoor scene objects: furniture, electronics, personal items, structural elements.
Occlusion_score [0,1]: 0 = fully visible, 1 = fully occluded.
Amodal bbox predicts full object extent including hidden portions.
High confidence (> 0.7) means reliable detection.
"""


def _rule_based_narrative(detections: List[dict], domain: str) -> str:
    """Fallback when LLM is unavailable."""
    if not detections:
        return "No objects detected in the scene."

    items    = [d["class_name"] for d in detections]
    occluded = [d["class_name"] for d in detections if d["occlusion_score"] > 0.3]

    narrative = f"Detected {len(detections)} object(s): {', '.join(items)}."
    if occluded:
        narrative += f" The following appear partially occluded and warrant priority inspection: {', '.join(occluded)}."
    if domain == "forensic" and occluded:
        narrative += " In a forensic context, occluded objects may indicate deliberate concealment."
    elif domain == "clinical" and occluded:
        narrative += " In a clinical context, occluded items should be located and accounted for immediately."
    return narrative


def generate_narrative(
    detections: List[dict],
    domain: str = "general",
    model_name: str = "gpt-4o",
) -> str:
    """
    Generate a plain-English scene description using RAG over domain ontology.

    Args:
        detections: list of detection dicts from /predict
        domain:     "general" | "clinical" | "forensic"
        model_name: LLM to use (requires OPENAI_API_KEY env var)

    Returns:
        2-3 sentence natural language scene description
    """
    if not detections:
        return "No objects detected."

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        log.warning("OPENAI_API_KEY not set — using rule-based narrator")
        return _rule_based_narrative(detections, domain)

    try:
        from langchain_community.vectorstores import Chroma
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.prompts import ChatPromptTemplate
        from langchain.schema.output_parser import StrOutputParser

        # Build in-memory ChromaDB from domain ontology
        ontology = {
            "forensic": FORENSIC_ONTOLOGY,
            "clinical": CLINICAL_ONTOLOGY,
        }.get(domain, GENERAL_ONTOLOGY)

        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        docs     = splitter.create_documents([ontology])
        vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings(api_key=api_key))
        retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Format detections for the prompt
        det_text = "\n".join([
            f"- {d['class_name']}: confidence={d['confidence']:.2f}, occlusion={d['occlusion_score']:.2f}"
            f"{' [PARTIALLY OCCLUDED]' if d['occlusion_score'] > 0.3 else ''}"
            for d in sorted(detections, key=lambda x: -x["occlusion_score"])
        ])

        query = f"domain:{domain} detections:{det_text}"
        context_docs = retriever.get_relevant_documents(query)
        context_text = "\n".join(d.page_content for d in context_docs)

        prompt = ChatPromptTemplate.from_template("""
You are an expert scene analyst. Using the domain context and object detections below,
write a precise 2-3 sentence scene description. Focus on:
1. What objects are present and their spatial arrangement
2. Any partially occluded objects and what might be hidden (use amodal bbox insight)
3. Domain-specific significance of the findings

Domain context:
{context}

Detected objects:
{detections}

Scene description:""")

        llm = ChatOpenAI(model=model_name, temperature=0.3, api_key=api_key)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context_text, "detections": det_text})

    except Exception as e:
        log.warning(f"LLM narrative failed ({e}) — using rule-based fallback")
        return _rule_based_narrative(detections, domain)