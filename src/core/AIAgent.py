"""AI Agent that intelligently combines regex pattern matching and LLM analysis."""
from collections import defaultdict
from textwrap import dedent
from typing import List, Dict, Any, Optional, Pattern, Tuple, Set
import re
import logging
from dataclasses import dataclass
import json
import asyncio
import itertools



from src.core.LLMDiffAnalyzer import LLMDiffAnalyzer, Violation
from src.services.LLMCallService import LLMCallService

logger = logging.getLogger(__name__)

@dataclass
class RegexMatch:
    """Represents a regex match in code with context."""
    file_path: str
    line_number: int
    content: str
    policy: str
    context_added_lines: List[Tuple[int, str, str]]  # Lines before
    context_removed_lines: List[Tuple[int, str, str]]   # Lines after
    match_details: Dict[str, Any] = None


class AIAgent:
    """Agent that intelligently combines regex pattern matching and LLM analysis."""

    def __init__(self, llm_analyzer: LLMDiffAnalyzer = None, llm_service: LLMCallService = None):
        """Initialize the agent with analyzers."""
        self.llm_analyzer = llm_analyzer or LLMDiffAnalyzer()
        self.llm_service = llm_service or LLMCallService()
        self.num_context_lines = 25  # Number of lines before and after for context

    async def run(self, diff_content: str, policies: List[str]) -> List[Violation]:
        """Analyze diff content using the most appropriate method for each policy.

        Args:
            diff_content: The diff content to analyze
            policies: List of policy descriptions to check against

        Returns:
            List of violations found
        """
        if not diff_content or not policies:
            raise ValueError("Missing diff content or policies")

        tasks = []
        for i, policy in enumerate(policies):
            logger.info(f"Processing policy: {i} {policy[:50]}")
            task = asyncio.create_task(
                self._analyze_policy(diff_content, policy, i)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        # Turn list of lists into a single list
        all_violations = list(itertools.chain.from_iterable(results))
        return all_violations
    

    
    async def _analyze_policy(self, diff_content: str, policy: str, policy_idx: int) -> List[Violation]:
        all_violations = []
        # Step 1: Determine if regex can be beneficial for this policy
        regex_strategy = await self._generate_regex_strategy(policy) 

        if regex_strategy and regex_strategy.get("use_regex", False):
            # Step 2: Use regex to find potential matches
            logger.info(f"Using regex strategy for policy: {policy_idx}")
            matches = self._apply_regex_strategy(diff_content, policy, regex_strategy)

            if matches:
                logger.info(f"Found potential matches for policy{policy_idx}: {len(matches)}")
                # Step 3: Send matches with context to LLM for verification
                policy_violations = await self._verify_matches_with_llm(matches, regex_strategy)
                all_violations.extend(policy_violations)
            else:
                logger.info(f"No regex matches found for policy: {policy_idx}")
        else:
            # Use LLM directly for this policy
            logger.info(f"Using LLM directly for policy: {policy_idx}")
            policy_violations = await self.llm_analyzer.analyze_diff_content_with_LLM(diff_content, [policy])
            all_violations.extend(policy_violations)
        return all_violations


    async def _extract_key_entities(self, policy: str) -> Dict[str, Any]:
        """Extract key entities from a policy description.

        Args:
            policy: The policy description

        Returns:
            Dict with extracted entities
        """
        prompt = f"""
        I need to extract key entities from the following policy:
        "{policy}"

        Please identify the possible coding Lanugage/s and then identify:
        1. Model names or class names that are the focus of this policy
        2. Method names that are relevant, including variations of the operations mentioned in the policy (based on the language) 
        3. Parameters or properties that need to be checked
        4. Conditions that need to be verified

        Respond in JSON format:
        {{
            "model_names": ["ModelName1", "ModelName2"],
            "method_names": ["methodName1", "methodName2"],
            "parameters": ["param1", "param2"],
            "conditions": ["condition1", "condition2"],
            "verification_criteria": [
                "criterion 1",
                "criterion 2"
            ]
        }}

        Be precise and extract ONLY the exact names mentioned in the policy.
        """

        try:
            response = await self.llm_service.generate_completion([
                {
                    "role": "system",
                    "content": "You are an expert in code analysis and policy interpretation. Extract key entities precisely as they appear in the policy.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ])

            content = response.choices[0].message.content
            # Extract JSON from the response (in case there's additional text)
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

            entities = json.loads(content)
            self.__log_key_entities(entities)
            return entities
        except Exception as e:
            logger.error(f"Error extracting key entities: {e}")
            return {
                "model_names": [],
                "method_names": [],
                "parameters": [],
                "conditions": [],
                "verification_criteria": []
            }

    def __log_key_entities(self, key_entities: Dict[str, Any]) -> None:
        log_msg = []
        log_msg.append(f"\n--- Key Entities Extracted ---")
        log_msg.append(f"  -- Model Names: {key_entities.get('model_names', [])}")
        log_msg.append(f"  -- Method Names: {key_entities.get('method_names', [])}")
        log_msg.append(f"  -- Parameters: {key_entities.get('parameters', [])}")
        log_msg.append(f"  -- Conditions: {key_entities.get('conditions', [])}")
        log_msg.append(f"  -- Verification Criteria: {key_entities.get('verification_criteria', [])}")
        logger.info(" |\n ".join(log_msg))



    async def _generate_regex_strategy(self, policy: str) -> Dict[str, Any]:
        """Generate a regex-based strategy for analyzing a policy.

        Args:
            policy: The policy description

        Returns:
            Dict with regex strategy information
        """
        # First, extract key entities from the policy
        key_entities = await self._extract_key_entities(policy)

        # Use the extracted entities to enhance the prompt for the LLM
        model_names = key_entities.get("model_names", [])
        method_names = key_entities.get("method_names", [])
        parameters = key_entities.get("parameters", [])
        conditions = key_entities.get("conditions", [])
        verification_criteria = key_entities.get("verification_criteria", [])

        if not(model_names or method_names or parameters):
            return {
                "use_regex": False
            }

        # Build a more specific prompt based on extracted entities
        entity_context = ""
        if model_names: entity_context += f"\nKey model names: {', '.join(model_names)}"
        if method_names: entity_context += f"\nKey method names: {', '.join(method_names)}"
        if parameters: entity_context += f"\nKey parameters: {', '.join(parameters)}"
        # if conditions: entity_context += f"\nKey conditions: {', '.join(conditions)}"
        # if verification_criteria: entity_context += f"\nKey verification criteria: {', '.join(verification_criteria)}"

        # Create a prompt that guides the LLM to generate appropriate regex patterns
        prompt = dedent(f"""
        I need to analyze code for policy violations.
        The policy is: {policy}
        
        Here are the key-entities extracted from the policy:
        {entity_context}

        I want to use regex pattern matching as a first step to identify potential matches.

        Please provide a regex strategy with these components:

        1. Should I use regex for initial filtering? (yes/no)
        2. If we should use regex, provide regex patterns for each of the key-entities to identify potential violations.
        3. Explain what each pattern is looking for.
        4. Describe what context I need to capture around matches (e.g., function definitions, variable declarations)
        5. What specific conditions should the LLM check in the matched code?

        IMPORTANT CONSIDERATIONS:
        - The code may be in JavaScript, TypeScript, or other languages
        - Don't match entire statements, just the relevant key-entities extracted.
        - Focus on simple patterns that identify candidate code sections
        - The regex should cast a wide net to find potential matches.
        - Prioritize patterns that match the most specific entities first
        - Use very simple patterns that are more likely to match.

        Respond in JSON format:
        {{
            "use_regex": true/false,
            "patterns": [
                {{
                    "pattern": "regex_pattern_here",
                    "description": "what this pattern matches",
                    "priority": 1-5 (1 is highest)
                }}
            ],
            "context_needed": "description of needed context",
            "llm_verification_criteria": [
                "criterion 1",
                "criterion 2"
            ]
        }}
        """)

        response = await self.llm_service.generate_completion([
            {
                "role": "system",
                "content": "You are an expert in code analysis, regex pattern matching, and static analysis. You also decide whether it's needed or not.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ], response_format = None)

        # Parse the response
        try:
            content = response.choices[0].message.content
            # Extract JSON from the response (in case there's additional text)
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

            strategy = json.loads(content)

            # Validate the regex patterns
            if strategy.get("use_regex", False) and "patterns" in strategy:
                valid_patterns = []
                for pattern_info in strategy["patterns"]:
                    try:
                        # Test compile the regex to ensure it's valid
                        re.compile(pattern_info["pattern"])
                        valid_patterns.append(pattern_info)
                    except re.error:
                        logger.warning(f"Invalid regex pattern: {pattern_info['pattern']}")

                if not valid_patterns:
                    strategy["use_regex"] = False
                else:
                    strategy["patterns"] = valid_patterns

            self.__log_regex_strategy(strategy)
            return strategy
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Error parsing regex strategy response: {e}")
            return {"use_regex": False}


    def __log_regex_strategy(self, regex_strategy: Dict[str, Any]) -> None:
        """Log the regex strategy for debugging."""
        log_msg = []
        if regex_strategy.get('patterns'):
            log_msg.append("\nGenerated Valid Regex Strategy:")
            for i, pattern in enumerate(regex_strategy.get('patterns', [])):
                log_msg.append(f"Pattern {i+1}: {pattern.get('pattern')}")
                log_msg.append(f"Description: {pattern.get('description')}")
                log_msg.append(f"Priority: {pattern.get('priority')}")
                log_msg.append("\n\n")
        else:
            log_msg.append("No Regex Strategy.")
        log_msg.append(f"Use regex: {regex_strategy.get('use_regex', False)}")
        logger.info(" |\n ".join(log_msg))

    

    def _parse_diff_content(self, diff_content: str) -> Dict[str, Dict[str, Any]]:
        """Parse diff content into a structured format with line numbers.

        Args:
            diff_content: The diff content

        Returns:
            Dict mapping file paths to file information
        """
        result = {}
        line_number = 0

        # Split diff into file chunks
        file_pattern = re.compile(r'diff --git .+ .+')
        file_chunks = file_pattern.split(diff_content)

        # Skip empty first chunk if needed
        if file_chunks and file_chunks[0].strip() == '':
            file_chunks.pop(0)

        # Process each file chunk
        for i, chunk in enumerate(file_chunks):
            # Extract file path
            file_path = None
            for line in chunk.split('\n'):
                if line.startswith('+++') and '/dev/null' not in line:
                    file_path = line[6:].strip()
                    break

            if not file_path:
                continue

            # Initialize file info
            result[file_path] = {
                "added_lines": self.llm_analyzer._extract_diff_added_lines(chunk),  # line_num -> content
                "removed_lines": self.llm_analyzer._extract_diff_removed_lines(chunk),  # line_num -> content
            }

        return result


    def _apply_regex_strategy(self, diff_content: str, policy: str, strategy: Dict[str, Any]) -> List[RegexMatch]:
        """Apply regex strategy to find potential matches in the diff content.

        Args:
            diff_content: The diff content
            policy: The policy description
            strategy: The regex strategy

        Returns:
            List of regex matches with context
        """
        matches = []

        # Parse the diff to extract file structure with line numbers
        parsed_diff = self._parse_diff_content(diff_content)


        parsed_diff = self._parse_diff_content(diff_content)
        # Compile all regex patterns
        compiled_patterns = []
        for pattern_info in strategy.get("patterns", []):
            try:
                compiled_patterns.append({
                    "pattern": re.compile(pattern_info["pattern"]),
                    "description": pattern_info.get("description", ""),
                    "priority": pattern_info.get("priority", 3)
                })
            except re.error:
                continue

        # Debug: Print all added lines to help with debugging
        logger.info(f"Searching for regex matches in {len(parsed_diff)} files")

        # Track potential matches before adding context
        potential_matches = []

        # First pass: Find all potential matches without context
        for file_path, file_info in parsed_diff.items():
            added_lines = file_info["added_lines"]
            removed_lines = file_info["removed_lines"]

            # logger.info(f"File: {file_path} has {len(added_lines)} added lines")
            try:
                # Find matches in each file's added lines
                for line_num, content, file_path in added_lines:
                    for pattern_info in compiled_patterns:
                        pattern = pattern_info["pattern"]
                        match_obj = pattern.search(content)
                        if match_obj:
                            potential_matches.append({
                                "file_path": file_path,
                                "line_number": line_num,
                                "content": content,
                                "pattern_info": pattern_info,
                                "match_obj": match_obj,
                                "all_lines": {
                                    "added_lines": added_lines,
                                    "removed_lines": removed_lines,
                                }  
                            })

                # Find matches in each file's removed lines
                for line_num, content, file_path in removed_lines:
                    for pattern_info in compiled_patterns:
                        pattern = pattern_info["pattern"]
                        match_obj = pattern.search(content)
                        if match_obj:
                            potential_matches.append({
                                "file_path": file_path,
                                "line_number": line_num,
                                "content": content,
                                "pattern_info": pattern_info,
                                "match_obj": match_obj,
                                "all_lines": {
                                    "added_lines": added_lines,
                                    "removed_lines": removed_lines,
                                }  
                            })
            except Exception as e:
                logger.error(f"Error applying regex pattern {pattern.pattern}: {e}")

        # Sort potential matches by file path and line number
        potential_matches.sort(key=lambda m: (m["file_path"], m["line_number"]))

        # Second pass: Add context to matches while avoiding overlaps
        for file_path, file_matches in itertools.groupby(potential_matches, key=lambda m: m["file_path"]):
            file_matches = list(file_matches)

            for i, match_info in enumerate(file_matches):
                line_num = match_info["line_number"]
                content = match_info["content"]
                all_lines = match_info["all_lines"]
                pattern_info = match_info["pattern_info"]
                match_obj = match_info["match_obj"]

                # Initialize context lists
                added_lines = all_lines["added_lines"]
                removed_lines = all_lines["removed_lines"]
                context_added_lines, context_removed_lines = [], []
                # Get context before & after
                for j in range(line_num - self.num_context_lines,  line_num + self.num_context_lines + 1):
                    for idx, (curr_line_num, _, _) in enumerate(added_lines):
                        if curr_line_num == j: 
                            context_added_lines.append(added_lines[idx])

                    for idx, (curr_line_num, _, _) in enumerate(removed_lines):
                        if curr_line_num == j: 
                            context_removed_lines.append(removed_lines[idx])

                matches.append(RegexMatch(
                    file_path=file_path,
                    line_number=line_num,
                    content=content,
                    policy=policy,
                    context_added_lines=context_added_lines,
                    context_removed_lines=context_removed_lines,
                    match_details={
                        "pattern": pattern_info["pattern"].pattern,
                        "match": match_obj.group(0),
                        "description": pattern_info["description"],
                        "priority": pattern_info["priority"],
                        "groups": {i: match_obj.group(i) for i in range(1, len(match_obj.groups()) + 1)} if match_obj.groups() else {}
                    }
                ))

        # Sort matches by priority (lower number = higher priority)
        matches.sort(key=lambda m: m.match_details["priority"])
        self.__log_regex_matches(matches)
        return matches

    def __log_regex_matches(self, matches: List[RegexMatch]) -> None:
        """Log the regex matches for debugging."""
        log_msg = []
        log_msg.append("\nFound Regex Matches:")
        for i, match in enumerate(matches):
            log_msg.append(f"Match {i+1}:")
            log_msg.append(f"  File: {match.file_path}")
            log_msg.append(f"  Line: {match.line_number}")
            log_msg.append(f"  Content: {match.content}")
            log_msg.append(f"  Match: {match.match_details['match']}")
            log_msg.append(f"  Priority: {match.match_details['priority']}")
            log_msg.append(f"  Pattern: {match.match_details['pattern']}")
            log_msg.append(f"  Description: {match.match_details['description']}")
            log_msg.append(f"  Groups: {match.match_details['groups']}")
        logger.info(" |\n ".join(log_msg))


    async def _verify_matches_with_llm(self, matches: List[RegexMatch], strategy: Dict[str, Any]) -> List[Violation]:
        """Verify regex matches using LLM to determine if they are actual violations.
        Args:
            matches: List of regex matches with context
            strategy: The regex strategy used
        Returns:
            List of violations
        """
        
        result_violations = []
        # Deduplicate matches based on file path and line number
        unique_matches = {}
        for match in matches:
            key = f"{match.file_path}:{match.line_number}"
            # Keep the match with the highest priority (lowest number)
            if key not in unique_matches or match.match_details["priority"] < unique_matches[key].match_details["priority"]:
                unique_matches[key] = match

        # Use the deduplicated matches
        matches = list(unique_matches.values())
        logger.info(f"Deduplicated {len(matches)} matches from original set")

        # Get verification criteria from strategy
        verification_criteria = strategy.get("llm_verification_criteria", [])
        criteria_text = "\n".join([f"- {criterion}" for criterion in verification_criteria])
        context_verification = f"""
            Policy violation verification criteria:
            {criteria_text}
        """

        # Group matches by file path
        matches_by_file = defaultdict(list)
        for match in matches:
            matches_by_file[match.file_path].append(match)

        # Process each file's matches
        tasks_async = []
        for file_path, file_matches in matches_by_file.items():
            # Prepare added_lines and removed_lines in the format expected by analyze_content
            all_added_lines = []
            all_removed_lines = []
            
            context_regex_pattern = ""
            for match in file_matches:
                # Add context lines if available
                if hasattr(match, 'context_added_lines') and match.context_added_lines:
                    all_added_lines.extend(match.context_added_lines)
                
                if hasattr(match, 'context_removed_lines') and match.context_removed_lines:
                            all_removed_lines.extend(match.context_removed_lines)
            
            # Remove duplicates while preserving order
            added_lines = list(dict.fromkeys(all_added_lines))
            removed_lines = list(dict.fromkeys(all_removed_lines))
            
            # Create task for this file
            task = self.llm_analyzer.analyze_content(added_lines, removed_lines, [file_matches[0].policy], context_verification )
            tasks_async.append(task)

        results = await asyncio.gather(*tasks_async)
        for result in results:
            result_violations.extend(result)

        self.__log_violations(result_violations)
        return result_violations
    

    def __log_violations(self, violations: List[Violation]):
        log_msg = []
        log_msg.append(f"\n--- Violations found  ---")
        for idx, violation in enumerate(violations):
            log_msg.append(f"  -- Violation {idx+1}")
            log_msg.append(f"  -- File:        {violation.file_path}")
            log_msg.append(f"  -- Line:        {violation.line_number_one_based}")
            log_msg.append(f"  -- Content:     {violation.content}")
            log_msg.append(f"  -- Policy:      {violation.prompt}")
            log_msg.append(f"  -- Explanation: {violation.explanation}\n\n")
        logger.info(" |\n ".join(log_msg))