"""
ollama_generator.py

Ollama-based image prompt generator for local LLM support.
Uses locally-hosted models via Ollama to analyze transcripts.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import requests
except ImportError:
    requests = None


class OllamaGenerator:
    """
    Image prompt generator using Ollama local LLMs.

    Analyzes transcript sections and generates detailed image prompts
    using locally-hosted models like Llama, Mistral, etc.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7
    ):
        """
        Initialize Ollama prompt generator.

        Args:
            model: Ollama model name (default: llama3)
            base_url: Ollama API base URL
            temperature: Sampling temperature (0.0-1.0)
        """
        if requests is None:
            raise ImportError(
                "Requests package not installed. Install with: pip install requests"
            )

        self.model = model or "llama3"
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature

    def generate_prompts(
        self,
        transcript_file: str,
        output_dir: str = "image_prompts"
    ) -> Dict[str, Any]:
        """
        Generate image prompts from transcript file.

        Args:
            transcript_file: Path to aligned JSON transcript
            output_dir: Directory to save prompts

        Returns:
            Dictionary with prompts_file path, sections, and metadata
        """
        # Load transcript
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)

        segments = transcript_data.get('segments', [])
        language = transcript_data.get('language', 'en')

        # Get video name from transcript file
        video_name = Path(transcript_file).stem.replace('.aligned', '')

        # Analyze transcript and generate prompts
        sections = self._analyze_and_generate(segments, language)

        # Prepare output
        os.makedirs(output_dir, exist_ok=True)
        output_file = Path(output_dir) / f"{video_name}_prompts.json"

        result = {
            "video_name": video_name,
            "transcript_file": transcript_file,
            "model": self.model,
            "generated_at": datetime.now().isoformat(),
            "language": language,
            "num_sections": len(sections),
            "sections": sections
        }

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Generated {len(sections)} image prompts")
        print(f"Saved to: {output_file}")

        return {
            "prompts_file": str(output_file),
            "sections": sections,
            "video_name": video_name
        }

    def _analyze_and_generate(
        self,
        segments: List[Dict],
        language: str
    ) -> List[Dict[str, Any]]:
        """
        Analyze transcript segments and generate image prompts using Ollama.

        Args:
            segments: List of transcript segments with timestamps
            language: Detected language of transcript

        Returns:
            List of sections with image prompts
        """
        # Combine segments into full transcript
        full_transcript = "\n".join(
            f"[{seg.get('start', 0):.1f}s - {seg.get('end', 0):.1f}s] {seg.get('text', '')}"
            for seg in segments
        )

        prompt = f"""You are an expert at analyzing video transcripts and generating detailed image prompts for visual content generation.

Your task:
1. Analyze this transcript and break it into semantic sections (scenes/topics)
2. For each section, generate a detailed image prompt that visually represents the content
3. Include sentiment, scene description, and visual style

Language: {language}

Transcript:
{full_transcript}

Output a JSON object with a "sections" array. Each section must have:
- start_time: float (seconds from transcript)
- end_time: float (seconds from transcript)
- transcript_text: string (relevant text)
- image_prompt: string (detailed visual description)
- sentiment: string (positive/neutral/negative)
- scene_description: string (brief summary)
- visual_style: string (realistic/artistic/cinematic/etc)

Respond ONLY with valid JSON, no other text."""

        try:
            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False
                },
                timeout=120  # 2 minute timeout for local generation
            )

            response.raise_for_status()
            result_data = response.json()
            content = result_data.get('response', '')

            # Parse JSON from response
            # Ollama may include extra text, try to extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)

            result = json.loads(content)

            # Extract sections
            if isinstance(result, dict) and "sections" in result:
                sections = result["sections"]
            elif isinstance(result, list):
                sections = result
            else:
                sections = [self._create_fallback_section(segments)]

            return sections

        except Exception as e:
            print(f"Warning: Ollama generation failed: {e}")
            print(f"Make sure Ollama is running and model '{self.model}' is available")
            print("Falling back to default section creation")
            return [self._create_fallback_section(segments)]

    def _create_fallback_section(self, segments: List[Dict]) -> Dict[str, Any]:
        """Create a fallback section when LLM fails."""
        if not segments:
            return {
                "start_time": 0.0,
                "end_time": 0.0,
                "transcript_text": "",
                "image_prompt": "A generic scene",
                "sentiment": "neutral",
                "scene_description": "No transcript available",
                "visual_style": "realistic"
            }

        full_text = " ".join(seg.get('text', '') for seg in segments)
        start_time = segments[0].get('start', 0.0)
        end_time = segments[-1].get('end', 0.0)

        return {
            "start_time": start_time,
            "end_time": end_time,
            "transcript_text": full_text[:500],
            "image_prompt": f"A scene depicting: {full_text[:200]}",
            "sentiment": "neutral",
            "scene_description": "Full video content",
            "visual_style": "realistic"
        }
