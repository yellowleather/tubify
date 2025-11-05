"""
anthropic_generator.py

Anthropic Claude-based image prompt generator.
Uses Claude models to analyze transcripts and generate image prompts.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import anthropic
except ImportError:
    anthropic = None


class AnthropicGenerator:
    """
    Image prompt generator using Anthropic's Claude models.

    Analyzes transcript sections and generates detailed image prompts
    suitable for image generation models.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000
    ):
        """
        Initialize Anthropic prompt generator.

        Args:
            model: Anthropic model name (default: claude-3-5-sonnet-20241022)
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        if anthropic is None:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )

        self.model = model or "claude-3-5-sonnet-20241022"
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Set API key from parameter or environment
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            # Uses ANTHROPIC_API_KEY environment variable
            self.client = anthropic.Anthropic()

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
        Analyze transcript segments and generate image prompts using Claude.

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

        system_prompt = """You are an expert at analyzing video transcripts and generating detailed image prompts for visual content generation.

Your task is to:
1. Analyze the transcript and break it into semantic sections (scenes/topics)
2. For each section, generate a detailed image prompt that visually represents the content
3. Include sentiment, scene description, and visual style suggestions

Generate prompts that are:
- Detailed and descriptive (for image generation models)
- Visually focused (describe what should be seen, not just the topic)
- Appropriate for the content and tone
- Suitable for models like DALL-E, Stable Diffusion, or Midjourney

Output format: JSON with a "sections" array, each section containing:
- start_time: float (seconds)
- end_time: float (seconds)
- transcript_text: string (the relevant transcript portion)
- image_prompt: string (detailed prompt for image generation)
- sentiment: string (positive/neutral/negative)
- scene_description: string (brief scene summary)
- visual_style: string (suggested style: realistic, artistic, cinematic, etc.)
"""

        user_prompt = f"""Analyze this transcript and generate image prompts:

Language: {language}

Transcript:
{full_transcript}

Generate a JSON object with a "sections" array. Each section should represent a coherent scene or topic shift."""

        try:
            # Call Anthropic API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            # Parse response
            content = message.content[0].text

            # Extract JSON from response (Claude may wrap it in markdown)
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()

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
            print(f"Warning: Claude generation failed: {e}")
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
