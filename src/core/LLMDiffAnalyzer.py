"""Core code analysis logic."""
import json
from typing import List, Dict, Any, Tuple
import re
import logging

from src.services.LLMCallService import LLMCallService
from src.core.models.Violation import Violation, ViolationList

logger = logging.getLogger(__name__)



class LLMDiffAnalyzer:
    """Analyzes code for policy violations using LLMs."""
    
    def __init__(self):
        self.llm_service = LLMCallService()
        self.LLM_SYSTEM_PROMPT = {
            "role": "system", 
            "content": [
                {
                "type": "text",
                "text": "You are a code review assistant that checks if code lines violate specific policies. The policies mention what they look for and what violation needs to be flagged."
                },
                {
                    "type": "text",
                    "text": "For each violation, you respond with the policy number, line number, \
                            (IMPORTANT: If multiple violations of the same policy occur in consecutive lines or within the same code block/file, \
                            only report one violation (the FIRST one). \
                            Include the A or R prefix),\
                            \
                            is_violation (its 'true' ONLY if the line clearly is a violation pattern, else false). If your explanation indicates the line is compliant or contains phrases like 'not a violation' or 'compliant'\
                            [is_violation] MUST be 'false',\
                            file path (with the correct letter casing, preserving any capitalization), and a brief explanation (less than ~35 words) in this format:\n\
                            'Policy [policy_number]: File Path: [file_path], Line [line_number] - [explanation] - [is_violation]\n'\
                            You report ONLY actual violations. If a line isn't a violation, you don't respond with anything.\
                            You review the added and removed lines (comes from the diff) and identify which ones violate any of the policies, you do NOT report the ones that don't violate a policy.",
                },
            ]
        }


    async def analyze_diff_file(self, diff_file_path: str, policies: List[str]) -> List[Violation]:
        """
        Args:
            diff_file_path: The path to the diff file to analyze
            policies: List of policy descriptions to check against
        Returns:
            List of violations found
        """
        with open(diff_file_path, 'r') as file:
            diff_content = file.read()
        return await self.analyze_diff_content_with_LLM(diff_content, policies)
    


    async def analyze_diff_content_with_LLM(self, diff_content: str, policies: List[str]) -> List[Violation]:
        """Analyze a diff against multiple policies.
        Args:
            diff_content: The diff content to analyze
            policies: List of policy descriptions to check against
        Returns:
            List of violations found
        """
        if not diff_content or not policies:
            raise ValueError("Missing diff content or policies")
        
        # Parse the diff to extract added lines with their original line numbers
        added_lines = self._extract_diff_added_lines(diff_content)
        removed_lines = self._extract_diff_removed_lines(diff_content)
        
        if not added_lines and not removed_lines:
            return []
            
        violation_list = await self.analyze_content(added_lines, removed_lines, policies)
        return violation_list


    
    async def analyze_content(self, added_lines, removed_lines, policies: List[str], extra_context: str = None) -> List[Violation]:
        """Analyze a diff against multiple policies.
        Args:
            added_lines: List of added lines with line numbers and file paths
            removed_lines: List of removed lines with line numbers and file paths
            policies: List of policy descriptions to check against
            extra_context: Optional additional context to include
        Returns:
            List of violations found
        """
        if not (added_lines or removed_lines) or not policies:
            raise ValueError("Missing content or policies")
        
        logger.warning('Sending request to LLM...')

        messages = self._build_prompt(added_lines, removed_lines, policies, extra_context)
            
        response = await self.llm_service.generate_completion(messages, response_format=ViolationList)
        result = response.choices[0].message.content
        logger.info(f"\n\n\nLLM response:\n%s\n\ {result}")

        violation_list = self._convert_to_schema(result)
        for v in violation_list.violations:
            v.is_violation = "VIOLATION: YES" in v.explanation
            sentence = v.explanation.replace("VIOLATION: YES", "").replace("VIOLATION: NO", "").strip()
            v.explanation = re.sub(r'^[. ]+', '', sentence)                 # Remove leading dots and spaces
        return [v for v in violation_list.violations if v.is_violation]


    def _extract_diff_added_lines(self, diff_content: str) -> List[Tuple[int, str, str]]:
        added_lines = []
        current_file = None
        line_number = 0
        for line in diff_content.split('\n'):
            if line.startswith('+++'):
                current_file = line[6:].strip()
            elif line.startswith('@@'):
                # Parse the hunk header to get the starting line number
                # Example: @@ -34,6 +34,8 @@ class Example:
                parts = line.split(' ')                                 # ["@@", "-34,6", "+34,8", "class", "Example:"]
                if len(parts) >= 3:
                    line_info = parts[2].strip()                        # "+34,8"
                    line_number = int(line_info.split(',')[0][1:]) - 1  # -1 because we increment the counter before using
            elif line.startswith('+') and not line.startswith('+++'):
                line_number += 1
                added_lines.append((line_number, line[1:], current_file))
            elif not line.startswith('-'):
                line_number += 1
        return added_lines


    def _extract_diff_removed_lines(self, diff_content: str) -> List[Tuple[int, str, str]]:
        removed_lines = []
        current_file = None
        line_number = 0
        for line in diff_content.split('\n'):
            if line.startswith('---'):
                current_file = line[6:].strip()
            elif line.startswith('@@'):
                # Parse the hunk header to get the starting line number
                # Example: @@ -34,6 +34,8 @@ class Example:
                parts = line.split(' ')
                if len(parts) >= 2:
                    line_info = parts[1].strip()  # "-34,6"
                    line_number = int(line_info.split(',')[0][1:]) - 1  # -1 because we increment before using
            elif line.startswith('-') and not line.startswith('---'):
                line_number += 1
                removed_lines.append((line_number, line[1:], current_file))
            elif not line.startswith('+'):
                line_number += 1
        return removed_lines
    

    def _convert_to_schema(self, content: str) -> ViolationList:
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                # Handle case where response isn't valid JSON
                logger.error("Failed to parse LLM response as JSON")
                return ViolationList(violations=[])
        return ViolationList(**content)
    

    def _build_prompt(self, added_lines, removed_lines, policies: List[str], extra_context: str = None) -> List[Dict]:
        """Build the prompt for LLM analysis.
        Args:
            added_lines: List of added lines with line numbers and file paths
            removed_lines: List of removed lines with line numbers and file paths
            policies: List of policy descriptions to check against
            extra_context: Optional additional context to include
        Returns:
            List of message dictionaries for LLM completion
        """
        added_lines_text = "\n".join([f"Added Line A{num} ({file}): {content}\n" for num, content, file in added_lines])
        removed_lines_text = "\n".join([f"Removed Line R{num} ({file}): {content}\n" for num, content, file in removed_lines])
        policies_text = "\n".join([f"{i+1}. {policy}" for i, policy in enumerate(policies)])

        messages = [
            {
                **self.LLM_SYSTEM_PROMPT
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": f"\
                                Added lines:\n{added_lines_text}\n\
                                Removed lines:\n{removed_lines_text}\n",
                    },
                    {
                        "type": "text",
                        "text": f"\
                                Here is the policy you need to check for:\
                                \n{policies_text}\n\n",
                        "cache_control": {"type": "ephemeral"},
                    },
                ]
            }
        ]
        if extra_context:
            messages[1]["content"].append({
                "type": "text",
                "text": f"{extra_context}",
            })
        messages[1]["content"].append({
                "type": "text",
                "text": "\
                        Is this a real violation of the policy? Respond in the explanation with:\n\
                        1. \"VIOLATION: YES\" or \"VIOLATION: NO\" on the first line\n\
                        2. A brief explanation (less than 35 words)",
                "cache_control": {"type": "ephemeral"},
        })
        
        return messages
