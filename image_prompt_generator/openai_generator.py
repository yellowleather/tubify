"""
openai_generator.py

OpenAI-based image prompt generator.
Uses GPT models to analyze transcripts and generate image prompts.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import openai
except ImportError:
    openai = None


class OpenAIGenerator:
    """
    Image prompt generator using OpenAI's GPT models.

    Analyzes transcript sections and generates detailed image prompts
    suitable for image generation models like DALL-E, Stable Diffusion, etc.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000
    ):
        """
        Initialize OpenAI prompt generator.

        Args:
            model: OpenAI model name (default: gpt-4o-mini)
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
        """
        if openai is None:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        self.model = model or "gpt-4o-mini"
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Set API key from parameter or environment
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            # Uses OPENAI_API_KEY environment variable
            self.client = openai.OpenAI()

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
        Analyze transcript segments and generate image prompts.

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

        # Create prompt for LLM
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

Output format: JSON array of sections, each with:
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

Generate a JSON array of sections with image prompts. Each section should represent a coherent scene or topic shift."""

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )

            # Parse response
            content = response.choices[0].message.content
            print(f"[DEBUG] LLM response length: {len(content)} chars")

            result = json.loads(content)
            print(f"[DEBUG] Parsed JSON type: {type(result)}")

            # Extract sections (handle different response structures)
            if isinstance(result, dict) and "sections" in result:
                sections = result["sections"]
                print(f"[DEBUG] Found {len(sections)} sections in 'sections' key")
            elif isinstance(result, list):
                sections = result
                print(f"[DEBUG] Found {len(sections)} sections as list")
            else:
                # Fallback: create single section
                print(f"[WARNING] Unexpected response format. Keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
                print(f"[WARNING] Using fallback section")
                sections = [self._create_fallback_section(segments)]

            return sections

        except Exception as e:
            print(f"Warning: LLM generation failed: {e}")
            print("Falling back to default section creation")
            return [self._create_fallback_section(segments)]

    def _create_fallback_section(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        Create a fallback section when LLM fails.

        Args:
            segments: Transcript segments

        Returns:
            Single section covering entire transcript
        """
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
            "transcript_text": full_text[:500],  # Truncate if too long
            "image_prompt": f"A scene depicting: {full_text[:200]}",
            "sentiment": "neutral",
            "scene_description": "Full video content",
            "visual_style": "realistic"
        }
