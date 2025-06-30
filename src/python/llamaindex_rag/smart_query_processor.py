
"""
File: smart_query_processor.py
Dynamic Query Processing for Essay Grading - NO Static Word Lists!
"""

import re
import math
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import Counter
import numpy as np


@dataclass
class QueryAnalysis:
    """Analysis of query characteristics for grading."""
    specificity_score: float  # 0.0 = very vague, 1.0 = very specific
    content_overlap_score: float  # How much query overlaps with actual document content
    term_rarity_score: float  # Rare terms = more specific knowledge
    query_type: str  # "specific", "moderate", "vague", "minimal"
    should_expand: bool  # Whether to apply academic expansion
    similarity_boost: float  # Multiplier for final scores


class DynamicQueryProcessor:
    """Processes queries based on ACTUAL document content, not static lists."""

    def __init__(self):
        self.document_vocabulary = set()
        self.term_frequencies = Counter()
        self.rare_terms = set()
        self.common_terms = set()
        self.document_bigrams = set()

    def learn_from_documents(self, documents: List[str]) -> None:
        """Learn what's specific vs vague from the ACTUAL document content."""
        all_text = " ".join(documents).lower()

        # Extract all terms
        terms = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
        self.term_frequencies = Counter(terms)
        self.document_vocabulary = set(terms)

        # Extract bigrams (two-word phrases)
        words = all_text.split()
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if len(bigram) > 6:  # Filter short bigrams
                self.document_bigrams.add(bigram)

        # Categorize terms by frequency (rarity = specificity)
        total_terms = len(terms)
        for term, freq in self.term_frequencies.items():
            frequency_ratio = freq / total_terms
            if frequency_ratio < 0.01:  # Appears in <1% of text
                self.rare_terms.add(term)
            elif frequency_ratio > 0.05:  # Appears in >5% of text
                self.common_terms.add(term)

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query specificity based on learned document content."""
        query_lower = query.lower()
        query_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', query_lower))
        query_words = query_lower.split()

        # 1. CONTENT OVERLAP SCORE (How much query matches document content)
        matching_terms = query_terms.intersection(self.document_vocabulary)
        content_overlap_score = len(matching_terms) / max(len(query_terms), 1)

        # 2. TERM RARITY SCORE (Rare terms = specific knowledge)
        rare_term_matches = query_terms.intersection(self.rare_terms)
        term_rarity_score = len(rare_term_matches) / max(len(query_terms), 1)

        # 3. BIGRAM SPECIFICITY (Multi-word technical phrases)
        bigram_matches = 0
        for i in range(len(query_words) - 1):
            bigram = f"{query_words[i]} {query_words[i+1]}"
            if bigram in self.document_bigrams:
                bigram_matches += 1
        bigram_score = bigram_matches / max(len(query_words) - 1, 1)

        # 4. LENGTH AND DETAIL SCORE
        # Longer = more detailed
        length_score = min(len(query_words) / 20, 1.0)

        # 5. VAGUE PATTERN DETECTION (dynamic)
        vague_patterns = [
            # "I know/think/etc"
            r'\bi\s+(know|understand|think|believe|feel)',
            # Generic qualifiers
            r'\b(everything|nothing|all|good|bad)\b',
            # Non-specific references
            r'\b(this|that|it)\s+(is|was|shows)',
            r'\b(very|really|quite|pretty)\s+\w+',         # Weak intensifiers
        ]

        vague_indicators = sum(
            1 for pattern in vague_patterns if re.search(pattern, query_lower))
        vague_penalty = max(0, 1 - (vague_indicators * 0.3))

        # COMBINE SCORES
        specificity_score = (
            content_overlap_score * 0.3 +     # 30% - matches document content
            term_rarity_score * 0.35 +        # 35% - uses rare/specific terms
            bigram_score * 0.25 +             # 25% - uses technical phrases
            length_score * 0.1                # 10% - sufficient detail
        ) * vague_penalty

        # DETERMINE QUERY TYPE AND PROCESSING STRATEGY
        if specificity_score >= 0.7:
            query_type = "specific"
            should_expand = True  # Expand academic terms
            similarity_boost = 1.2  # Boost scores
        elif specificity_score >= 0.4:
            query_type = "moderate"
            should_expand = True
            similarity_boost = 1.0
        elif specificity_score >= 0.2:
            query_type = "vague"
            should_expand = False  # DON'T expand vague queries
            similarity_boost = 0.8  # Penalize scores
        else:
            query_type = "minimal"
            should_expand = False
            similarity_boost = 0.6  # Heavy penalty

        return QueryAnalysis(
            specificity_score=specificity_score,
            content_overlap_score=content_overlap_score,
            term_rarity_score=term_rarity_score,
            query_type=query_type,
            should_expand=should_expand,
            similarity_boost=similarity_boost
        )

    def process_query_for_retrieval(self, query: str) -> Tuple[str, float]:
        """Process query and return (processed_query, similarity_boost)."""
        analysis = self.analyze_query(query)

        if analysis.should_expand:
            # Only expand if query shows academic intent
            # Minimal expansion focused on the document content
            processed_query = self._smart_expand(query)
        else:
            # No expansion for vague queries
            processed_query = query

        return processed_query, analysis.similarity_boost

    def _smart_expand(self, query: str) -> str:
        """Smart expansion based on document content only."""
        # Very conservative expansion - only add terms that exist in the document
        words = query.lower().split()
        expanded_words = []

        for word in words:
            expanded_words.append(word)

            # Only add synonyms if they exist in the document vocabulary
            if word in self.document_vocabulary:
                # Simple synonym mapping based on document content
                if word == "analyze" and "examine" in self.document_vocabulary:
                    expanded_words.append("examine")
                elif word == "compare" and "contrast" in self.document_vocabulary:
                    expanded_words.append("contrast")
                # Add more only if they exist in the actual document

        return " ".join(expanded_words)

# INTEGRATION WITH RETRIEVAL ENGINE


class SmartRetrievalEngine:
    """Retrieval engine with dynamic vague query filtering."""

    def __init__(self, core, documents: List[str]):
        self.core = core
        self.query_processor = DynamicQueryProcessor()
        self.query_processor.learn_from_documents(documents)

    def retrieve_with_smart_filtering(self, query: str, retrieval_function, **kwargs) -> Dict:
        """Retrieve with smart vague query filtering."""
        # Analyze and process query
        processed_query, similarity_boost = self.query_processor.process_query_for_retrieval(
            query)
        analysis = self.query_processor.analyze_query(query)

        # Perform retrieval
        results = retrieval_function(processed_query, **kwargs)

        # Apply similarity boost/penalty
        if 'results' in results:
            for result in results['results']:
                result['score'] = result['score'] * similarity_boost
                result['original_score'] = result['score'] / similarity_boost

        # Add analysis to metadata
        results['query_analysis'] = {
            'specificity_score': analysis.specificity_score,
            'query_type': analysis.query_type,
            'similarity_boost': similarity_boost,
            'should_expand': analysis.should_expand
        }

        return results
