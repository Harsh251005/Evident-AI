import sys
import os
import time
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()


def check_quality_gate(project_name):

    if not os.getenv("LANGCHAIN_API_KEY"):
        print("❌ Error: LANGCHAIN_API_KEY is missing from environment/env file.")
        sys.exit(1)

    client = Client()
    print(f"🧐 Fetching results for: {project_name}...")

    # Give LangSmith a moment to process (5 seconds)
    time.sleep(5)

    try:
        # 1. Get all runs from this experiment
        runs = list(client.list_runs(project_name=project_name, is_root=True))

        scores = []
        for run in runs:
            # 2. Get the feedback (scores) for each run
            feedbacks = list(client.list_feedback(run_ids=[run.id]))
            for f in feedbacks:
                if f.key == "citation_coverage":
                    scores.append(f.score)

        if not scores:
            print(f"⚠️ No feedback data found yet. LangSmith is likely still processing.")
            print("Please wait 20 seconds and run this command again.")
            sys.exit(1)

        # 3. Calculate the average ourselves
        avg_citation = sum(scores) / len(scores)

        print(f"\n--- 🛡️ Production Quality Gate: {project_name} ---")
        print(f"📈 Analyzed {len(scores)} feedback entries.")
        print(f"📊 Average Citation Coverage: {avg_citation * 100:.1f}%")

        # 4. The Deployment Threshold
        THRESHOLD = 0.80
        if avg_citation < THRESHOLD:
            print(f"❌ DEPLOYMENT BLOCKED: Quality is below {THRESHOLD * 100}%.")
            sys.exit(1)

        print("✅ DEPLOYMENT PASSED: Metrics meet production standards.")
        sys.exit(0)

    except Exception as e:
        print(f"❌ Error accessing LangSmith: {e}")
        sys.exit(1)


if __name__ == "__main__":
    target_project = os.getenv("LANGSMITH_PROJECT_NAME")
    if not target_project:
        print("❌ Error: Set LANGSMITH_PROJECT_NAME env var first.")
        sys.exit(1)

    check_quality_gate(target_project)