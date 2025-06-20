
import asyncio
import logging
from src.core.AIAgent import AIAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def read_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            if not content:
                logger.error(f"File is empty: {file_path}")
                return None
            return content
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return None



async def analyze_diff_file(diff_file_path, policy_file_path):    
    policy = read_from_file(policy_file_path)
    diff_content = read_from_file(diff_file_path)
    
    # Initialize AI agent
    agent = AIAgent()
    logger.info(f"Analyzing diff file: {diff_file_path}")
    violations = await agent.run(diff_content, [policy])  # Pass as a list with one item
    return violations


def main():
    # Path to files
    diff_file_path = "tests/pr_diff.txt"
    policy_file_path = "tests/violation_prompt.txt"
    
    # Run the analysis
    all_violations = asyncio.run(analyze_diff_file(diff_file_path, policy_file_path))
    
    # Print results
    if all_violations:
        logger.info(f"Found {len(all_violations)} violations:")
        for violation in all_violations:
            logger.info(f"- {violation}")
    else:
        logger.info("No violations found.")



if __name__ == "__main__":
    main()