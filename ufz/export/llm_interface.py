"""LLM Interface: Query urban functional zones using LLM.

This module provides a unified interface for querying the knowledge graph
using different LLM backends (Google Gemini, Deepseek, etc.).

Features:
- Multi-provider support (Google, Deepseek)
- Dynamic context selection based on relevance
- Hierarchical reasoning (coarse-to-fine)
- Structured output with reasoning
"""

import logging
import json
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of an LLM query."""

    query: str
    answer: str
    reasoning: str
    recommended_clusters: List[int]
    confidence: float
    raw_response: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def query(self, prompt: str) -> str:
        """Send a prompt to the LLM and get response."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        pass


class GoogleGeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    def __init__(self, api_key: str, model: str = "gemini-pro"):
        """Initialize Google Gemini provider.

        Args:
            api_key: Google API key
            model: Model name (default: gemini-pro)
        """
        self.api_key = api_key
        self.model = model

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
            logger.info(f"✓ Google Gemini initialized ({model})")
        except ImportError:
            logger.warning("google-generativeai not installed. Install with: pip install google-generativeai")
            self.client = None

    def query(self, prompt: str) -> str:
        """Query Google Gemini."""
        if self.client is None:
            raise RuntimeError("Google Gemini client not initialized")

        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Google Gemini query failed: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough approximation: ~4 chars per token for English, ~2 for Chinese
        return len(text) // 3


class DeepseekProvider(LLMProvider):
    """Deepseek API provider."""

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        """Initialize Deepseek provider.

        Args:
            api_key: Deepseek API key
            model: Model name (default: deepseek-chat)
        """
        self.api_key = api_key
        self.model = model

        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            logger.info(f"✓ Deepseek initialized ({model})")
        except ImportError:
            logger.warning("openai not installed. Install with: pip install openai")
            self.client = None

    def query(self, prompt: str) -> str:
        """Query Deepseek."""
        if self.client is None:
            raise RuntimeError("Deepseek client not initialized")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Deepseek query failed: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 3


class UrbanFunctionalZoneQueryEngine:
    """Query engine for urban functional zone analysis."""

    def __init__(
        self,
        knowledge_graph,  # HierarchicalKnowledgeGraph
        llm_provider: LLMProvider,
        system_prompt_template: Optional[str] = None,
        max_context_tokens: int = 6000,
    ):
        """Initialize query engine.

        Args:
            knowledge_graph: HierarchicalKnowledgeGraph object
            llm_provider: LLM provider instance
            system_prompt_template: Custom system prompt template
            max_context_tokens: Maximum tokens for context (to stay under LLM limits)
        """
        self.kg = knowledge_graph
        self.llm = llm_provider
        self.max_context_tokens = max_context_tokens
        self.system_prompt_template = system_prompt_template or self._default_system_prompt()

    def query(self, user_question: str) -> QueryResult:
        """Query about urban functional zones.

        Args:
            user_question: User's question (e.g., "Where should I open a restaurant?")

        Returns:
            QueryResult with answer and reasoning
        """
        logger.info(f"Processing query: {user_question}")

        # Step 1: Determine relevant hierarchy level
        relevant_level = self._determine_relevant_level(user_question)
        logger.info(f"  Selected hierarchy level: {relevant_level}")

        # Step 2: Extract relevant clusters
        relevant_clusters = self._extract_relevant_clusters(
            user_question, relevant_level
        )
        logger.info(f"  Relevant clusters: {[c.cluster_id for c in relevant_clusters]}")

        # Step 3: Build context
        context = self._build_context(relevant_clusters, user_question)

        # Step 4: Build prompt
        prompt = self._build_prompt(context, user_question)

        # Step 5: Query LLM
        if self.llm is None:
            raise RuntimeError("LLM provider not initialized. Please provide an LLM provider instance.")

        logger.info(f"  Querying LLM ({self.llm.__class__.__name__})...")
        raw_response = self.llm.query(prompt)

        # Step 6: Parse response
        result = self._parse_response(
            raw_response, user_question, [c.cluster_id for c in relevant_clusters]
        )

        logger.info(f"  ✓ Query completed")
        return result

    def _determine_relevant_level(self, question: str) -> int:
        """Determine which hierarchy level is most relevant.

        Simple heuristic: if question is very specific, use finer level.
        """
        # Keywords for coarse-level queries
        coarse_keywords = ['city', '城市', 'overall', '总体', 'district', '区域']
        # Keywords for fine-level queries
        fine_keywords = ['street', '街道', 'specific', '具体', 'neighborhood', '社区']

        question_lower = question.lower()

        for keyword in fine_keywords:
            if keyword in question_lower:
                # Use finest level
                return self.kg.get_hierarchy_depth() - 1

        for keyword in coarse_keywords:
            if keyword in question_lower:
                # Use coarsest level
                return 0

        # Default: use first non-coarsest level
        return min(1, self.kg.get_hierarchy_depth() - 1)

    def _extract_relevant_clusters(
        self,
        question: str,
        level: int,
    ) -> List['ClusterSummary']:
        """Extract clusters relevant to the question.

        Uses semantic similarity and keyword matching.
        """
        all_clusters = self.kg.get_level_clusters(level)

        if len(all_clusters) == 0:
            logger.warning(f"No clusters at level {level}, using level 0")
            all_clusters = self.kg.get_level_clusters(0)

        # Simple relevance scoring
        relevant_clusters = []
        for cluster in all_clusters:
            score = self._compute_relevance_score(cluster, question)
            if score > 0:
                relevant_clusters.append((cluster, score))

        # Sort by relevance and return top clusters
        relevant_clusters.sort(key=lambda x: x[1], reverse=True)
        # Return top 5 or all if less than 5
        return [c for c, _ in relevant_clusters[:5]]

    @staticmethod
    def _compute_relevance_score(cluster: 'ClusterSummary', question: str) -> float:
        """Compute relevance score between a cluster and question."""
        score = 0.0

        question_lower = question.lower()

        # Keyword matching
        for poi in cluster.dominant_pois:
            if poi.lower() in question_lower:
                score += 1.0

        for business in cluster.suitable_business:
            if business.lower() in question_lower:
                score += 0.5

        # Default score if no matches
        if score == 0:
            score = 0.1

        return score

    def _build_context(self, clusters: List['ClusterSummary'], question: str) -> str:
        """Build context information from relevant clusters.

        Formats cluster information for LLM consumption.
        """
        if not clusters:
            logger.warning("No relevant clusters found")
            return "No relevant data found."

        context_parts = []
        context_parts.append("## 城市功能区信息\n")

        for i, cluster in enumerate(clusters, 1):
            context_parts.append(f"\n### 功能区 {i} (ID: {cluster.cluster_id})\n")
            context_parts.append(f"- **规模**: {cluster.node_count} 个建筑单元\n")
            context_parts.append(f"- **位置**: 经度 {cluster.center_lon:.2f}, 纬度 {cluster.center_lat:.2f}\n")

            if cluster.characteristics:
                context_parts.append(f"- **特征**: {cluster.characteristics}\n")

            if cluster.dominant_pois:
                context_parts.append(f"- **主导 POI**: {', '.join(cluster.dominant_pois)}\n")

            if cluster.poi_distribution:
                poi_str = ", ".join([
                    f"{poi}: {prob:.1%}"
                    for poi, prob in sorted(
                        cluster.poi_distribution.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                ])
                context_parts.append(f"- **POI 分布**: {poi_str}\n")

            if cluster.suitable_business:
                context_parts.append(f"- **适合行业**: {', '.join(cluster.suitable_business)}\n")

            if cluster.neighbor_clusters:
                neighbors_str = ", ".join([
                    f"区域 {cid} ({rel})"
                    for cid, rel, _ in cluster.neighbor_clusters[:3]
                ])
                context_parts.append(f"- **相邻区域**: {neighbors_str}\n")

        context = "".join(context_parts)

        # Truncate if too long
        if self.llm is not None:
            token_count = self.llm.count_tokens(context)
        else:
            token_count = len(context) // 3  # Rough estimate

        if token_count > self.max_context_tokens:
            logger.warning(f"Context too large ({token_count} tokens), truncating...")
            # Simple truncation: keep first N clusters
            context = self._build_context(clusters[:3], question)

        return context

    def _build_prompt(self, context: str, question: str) -> str:
        """Build the final prompt for LLM."""
        prompt = self.system_prompt_template.format(
            context=context,
            question=question,
        )
        return prompt

    @staticmethod
    def _default_system_prompt() -> str:
        """Get default system prompt template."""
        return """你是一个资深的城市规划顾问和商业分析专家。你深入了解城市商业生态和地产投资。

你的任务是基于以下城市功能区的详细信息，回答用户关于商业选址和区域分析的问题。

{context}

用户问题: {question}

请按以下格式回答:

## 分析结果

[简明扼要的答案，包括具体建议]

## 推荐区域

[列出最合适的功能区及其 ID]

## 理由说明

[详细解释为什么这些区域最合适，考虑以下因素:
- POI 分布和商业生态
- 区域特征和人口特性
- 相邻区域的协同效应
- 行业特定的优势]

## 风险提示

[如有必要，指出需要注意的因素]"""

    @staticmethod
    def _parse_response(
        response: str,
        question: str,
        cluster_ids: List[int],
    ) -> QueryResult:
        """Parse LLM response into QueryResult."""
        # Extract recommended clusters from response
        recommended = []
        for cid in cluster_ids:
            if f"区域 {cid}" in response or f"ID: {cid}" in response:
                recommended.append(cid)

        if not recommended:
            recommended = cluster_ids[:1]  # Default to first

        # Try to extract reasoning
        reasoning = ""
        if "理由说明" in response or "理由" in response:
            parts = response.split("理由说明")
            if len(parts) > 1:
                reasoning = parts[1].split("##")[0].strip()

        # Try to extract answer
        answer = ""
        if "分析结果" in response or "答案" in response:
            parts = response.split("分析结果")
            if len(parts) > 1:
                answer = parts[1].split("##")[0].strip()
        else:
            answer = response[:500]  # Default to first 500 chars

        return QueryResult(
            query=question,
            answer=answer,
            reasoning=reasoning,
            recommended_clusters=recommended,
            confidence=0.85,  # Simple heuristic
            raw_response=response,
        )


def create_query_engine(
    knowledge_graph,
    provider: str = "deepseek",
    api_key: Optional[str] = None,
    max_context_tokens: int = 6000,
) -> UrbanFunctionalZoneQueryEngine:
    """Factory function to create a query engine.

    Args:
        knowledge_graph: HierarchicalKnowledgeGraph object
        provider: "google" or "deepseek"
        api_key: API key for the provider
        max_context_tokens: Maximum context tokens

    Returns:
        UrbanFunctionalZoneQueryEngine instance
    """
    if api_key is None:
        raise ValueError(f"API key required for {provider}")

    if provider == "google":
        llm = GoogleGeminiProvider(api_key)
    elif provider == "deepseek":
        llm = DeepseekProvider(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return UrbanFunctionalZoneQueryEngine(
        knowledge_graph,
        llm,
        max_context_tokens=max_context_tokens,
    )
