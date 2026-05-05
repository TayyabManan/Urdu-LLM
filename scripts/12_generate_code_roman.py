"""
Generate code explanation examples in Roman Urdu using GPT-4o-mini.

Produces Q&A pairs where a Pakistani dev asks about programming in Roman Urdu
and gets an explanation in Roman Urdu with code blocks in English.
Natural code-switching — how Pakistani devs actually talk on WhatsApp/Discord.

Same token-saving strategies as 05_transliterate_to_roman.py:
    1. Batch 3 examples per API call
    2. Compact system prompt
    3. Saves progress — resume if interrupted
    4. Tracks cost in real-time

Run with:
    python scripts/12_generate_code_roman.py --n 2000 --budget 1.50

Resume:
    python scripts/12_generate_code_roman.py --n 2000 --budget 1.50 --resume
"""

import argparse
import json
import os
import time
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BATCH_SIZE = 2            # Reduced from 3 — code responses are long, 3 often truncates
MODEL = "gpt-4o-mini"
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60
SLEEP_BETWEEN = 0.5
MAX_RETRIES = 2           # Retry failed batches before skipping
OUTPUT_DIR = Path("data/code_roman")
OUTPUT_FILE = OUTPUT_DIR / "code_roman.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

SYSTEM_PROMPT = (
    "You are a Pakistani software developer who explains programming in Roman Urdu. "
    "Generate instruction-output pairs. Instruction: a question a Pakistani student would ask "
    "in Roman Urdu (natural WhatsApp style). Output: explanation in Roman Urdu with code examples "
    "in English. Keep English technical terms in English (function, loop, API, database, etc). "
    "Return valid JSON array."
)

# Topic pools — each call picks a random topic
TOPICS = [
    # Python basics
    "Python variables aur data types",
    "Python mein if-else conditions",
    "Python mein for loop aur while loop",
    "Python functions kaise banate hain",
    "Python mein list ka use",
    "Python dictionary kaise kaam karti hai",
    "Python mein tuple aur set ka farq",
    "Python string methods",
    "Python file reading aur writing",
    "Python mein error handling (try-except)",
    "Python classes aur objects (OOP)",
    "Python mein inheritance",
    "Python decorators",
    "Python list comprehension",
    "Python mein lambda functions",
    "Python generators aur yield",
    "Python mein modules aur imports",
    "Python virtual environment kyun zaroori hai",
    "Python mein args aur kwargs",
    "Python mein map, filter, reduce",

    # Web development
    "HTML basics — tags, elements, attributes",
    "CSS styling — colors, fonts, layout",
    "CSS Flexbox kaise use karte hain",
    "CSS Grid layout",
    "JavaScript variables (let, const, var)",
    "JavaScript mein DOM manipulation",
    "JavaScript async/await",
    "JavaScript mein fetch API",
    "React mein components kaise banate hain",
    "React useState hook",
    "React useEffect hook",
    "React mein props kya hain",
    "Node.js mein server banana",
    "Express.js routing",
    "REST API design principles",
    "JSON kya hai aur kaise use hota hai",
    "HTTP methods (GET, POST, PUT, DELETE)",
    "CORS error kya hota hai aur fix kaise karein",
    "Authentication — JWT tokens",
    "API rate limiting",

    # Data structures & algorithms
    "Array vs Linked List ka farq",
    "Stack data structure",
    "Queue data structure",
    "Binary search algorithm",
    "Sorting algorithms — bubble sort, merge sort",
    "Hash table / hash map",
    "Tree data structure basics",
    "Graph data structure",
    "Big O notation kya hai",
    "Recursion kaise kaam karti hai",

    # Database
    "SQL SELECT query basics",
    "SQL JOIN types (INNER, LEFT, RIGHT)",
    "SQL mein WHERE aur HAVING ka farq",
    "SQL mein GROUP BY",
    "Database normalization kya hai",
    "SQL vs NoSQL ka farq",
    "MongoDB basics",
    "Database indexing kyun zaroori hai",
    "SQL injection kya hai aur kaise roukein",
    "ORM kya hota hai (SQLAlchemy, Prisma)",

    # DevOps & tools
    "Git basics — init, add, commit, push",
    "Git branching aur merging",
    "Git mein merge conflict kaise solve karein",
    "Docker containers kya hain",
    "Docker Compose",
    "CI/CD pipeline kya hoti hai",
    "Linux terminal basic commands",
    "SSH kya hai aur kaise connect karein",
    "Environment variables kaise set karein",
    ".env file kaise use karein",

    # General CS concepts
    "API kya hoti hai — restaurant example",
    "Frontend vs Backend ka farq",
    "Client-server architecture",
    "Microservices vs Monolith",
    "Caching kya hai aur kyun zaroori hai",
    "DNS kaise kaam karta hai",
    "HTTPS aur SSL/TLS",
    "WebSocket vs HTTP",
    "Design patterns — Singleton, Factory",
    "MVC pattern kya hai",

    # AI/ML basics
    "Machine Learning kya hai — simple explanation",
    "Supervised vs Unsupervised learning",
    "Neural network kaise kaam karta hai",
    "Overfitting kya hota hai aur kaise rokein",
    "Train/test split kyun zaroori hai",
    "Pandas DataFrame basics",
    "NumPy arrays",
    "Matplotlib se graph banana",
    "Jupyter Notebook kaise use karein",
    "Hugging Face Transformers library basics",

    # Debugging & best practices
    "Code mein bugs kaise dhundhein — debugging tips",
    "Print debugging vs debugger tools",
    "Code review kaise karein",
    "Clean code principles",
    "DRY principle (Don't Repeat Yourself)",
    "Comments aur documentation likhne ka tareeqa",
    "Unit testing kya hai aur kyun zaroori hai",
    "Code refactoring kaise karein",
    "Version control best practices",
    "README file kaise likhein",

    # Freelancing & career
    "Freelancing shuru karne ke liye kya skills chahiye",
    "Portfolio website kaise banayein",
    "Upwork/Fiverr pe profile kaise optimize karein",
    "Technical interview ki tayyari kaise karein",
    "Data Structures interview questions",
    "System design interview basics",
    "Open source contribute kaise karein",
    "GitHub profile kaise strong banayein",
    "Resume mein projects kaise highlight karein",
    "Remote job kaise dhundhein",

    # Python intermediate/advanced
    "Python mein context managers (with statement)",
    "Python mein dataclasses kya hain",
    "Python typing module — type hints kaise likhen",
    "Python mein async/await kaise kaam karta hai",
    "Python mein pickle module se data save karna",
    "Python mein regular expressions (regex)",
    "Python mein collections module — Counter, defaultdict",
    "Python mein itertools kaise use karein",
    "Python mein pathlib vs os.path",
    "Python mein logging module kaise set karein",
    "Python mein multiprocessing vs threading",
    "Python mein pip aur package management",
    "Python mein f-strings aur string formatting",
    "Python mein walrus operator (:=) kya hai",
    "Python mein enumerate aur zip functions",

    # Web frameworks
    "FastAPI se REST API banana",
    "Django vs Flask — kaunsa choose karein",
    "Next.js kya hai aur React se kaise alag hai",
    "Tailwind CSS kaise use karte hain",
    "Bootstrap vs Tailwind ka farq",
    "Vue.js basics — components aur reactivity",
    "TypeScript kya hai aur kyun use karein",
    "Svelte framework ka introduction",
    "Angular vs React vs Vue — comparison",
    "Vite build tool kya hai",

    # Cloud & deployment
    "AWS basics — EC2, S3, Lambda",
    "Vercel pe Next.js app deploy karna",
    "Heroku pe Python app deploy karna",
    "Nginx kya hai aur reverse proxy kaise set karein",
    "Domain name kaise connect karein website se",
    "SSL certificate kaise lagayein website pe",
    "Firebase kya hai — authentication aur database",
    "Supabase vs Firebase ka comparison",
    "Cloudflare Workers kya hain",
    "GitHub Actions se CI/CD setup karna",

    # Mobile development
    "React Native se mobile app banana",
    "Flutter vs React Native ka farq",
    "Expo kya hai React Native mein",
    "Mobile app ka APK kaise banayein",
    "App store pe app publish kaise karein",

    # Data science & analytics
    "Pandas mein CSV file load aur analyze karna",
    "Data visualization — matplotlib vs seaborn",
    "Web scraping Python mein — BeautifulSoup",
    "APIs se data fetch karke analyze karna",
    "Excel vs Python data analysis ke liye",
    "Kaggle competitions mein participate kaise karein",
    "Feature engineering kya hoti hai ML mein",
    "Cross-validation kya hai aur kyun zaroori hai",
    "Confusion matrix kaise samjhein",
    "Random Forest vs Decision Tree",

    # Security
    "Password hashing kya hai — bcrypt example",
    "XSS attack kya hota hai aur kaise rokein",
    "CSRF attack se kaise bachein",
    "OAuth 2.0 kaise kaam karta hai",
    "Two-factor authentication implement karna",
    "Environment variables mein secrets store karna",
    "HTTPS kyun zaroori hai",
    "Input validation kaise karein backend mein",
    "Rate limiting kyun lagani chahiye API pe",
    "Helmet.js kya karta hai Express mein",

    # Testing
    "Pytest se Python tests likhna",
    "Jest se JavaScript tests likhna",
    "Integration testing vs unit testing ka farq",
    "Test-driven development (TDD) kya hai",
    "Mock objects kya hotay hain testing mein",
    "Code coverage kya hoti hai",
    "End-to-end testing — Cypress, Playwright",
    "API testing — Postman kaise use karein",
    "Continuous testing CI/CD pipeline mein",
    "Snapshot testing React mein",

    # Architecture & patterns
    "Clean architecture kya hoti hai",
    "Repository pattern kya hai",
    "Event-driven architecture",
    "Message queues — RabbitMQ, Redis",
    "GraphQL vs REST ka comparison",
    "Monorepo vs polyrepo ka farq",
    "SOLID principles kya hain",
    "Dependency injection samjhao",
    "Observer pattern kya hai",
    "State management — Redux vs Context API",

    # Practical projects
    "Todo app kaise banayein React mein",
    "Blog website kaise banayein Django se",
    "Chat application WebSocket se kaise banayein",
    "E-commerce site ka database design",
    "URL shortener kaise banayein",
    "File upload feature kaise implement karein",
    "Search functionality kaise add karein website mein",
    "Pagination implement karna API mein",
    "Image optimization web applications mein",
    "PDF generate karna Python se",
]


def build_user_message(topics):
    topic_list = "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics))
    return (
        f"Generate {len(topics)} instruction-output pairs, one for each topic:\n"
        f"{topic_list}\n\n"
        f'Return JSON object: {{"examples": [{{"instruction": "...", "output": "..."}}]}}\n'
        f"Instruction = question in Roman Urdu. Output = answer in Roman Urdu with code if relevant."
    )


def parse_response(response_text):
    cleaned = response_text.strip()

    # Try direct parse — JSON object with "examples" key (structured output)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "examples" in parsed:
            return parsed["examples"]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Strip markdown code blocks
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "examples" in parsed:
                return parsed["examples"]
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Find JSON array or object anywhere in text
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = cleaned.find(start_char)
        end = cleaned.rfind(end_char)
        if start != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end + 1])
                if isinstance(parsed, dict) and "examples" in parsed:
                    return parsed["examples"]
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

    return None


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"done": 0, "total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0.0}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000, help="Number of examples to generate")
    parser.add_argument("--budget", type=float, default=1.50, help="Max spend in dollars")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    random.seed(args.seed)

    if args.resume:
        progress = load_progress()
        start_from = progress["done"]
        print(f"Resuming from example {start_from} (${progress['total_cost']:.4f} spent)")
    else:
        progress = {"done": 0, "total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0.0}
        start_from = 0
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()

    # Generate topic batches — cycle through topics randomly
    remaining = args.n - start_from
    batches_needed = (remaining + BATCH_SIZE - 1) // BATCH_SIZE

    # Shuffle topics for variety
    all_topics = TOPICS * ((args.n // len(TOPICS)) + 2)
    random.shuffle(all_topics)

    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")

    print(f"\nPlan:")
    print(f"  Examples to generate: {remaining:,}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  API calls needed: {batches_needed:,}")
    print(f"  Budget: ${args.budget:.2f}")
    print(f"  Topics pool: {len(TOPICS)}")
    print(f"\nStarting...\n")

    failed_batches = 0

    for batch_idx in range(batches_needed):
        topic_start = (start_from + batch_idx * BATCH_SIZE) % len(all_topics)
        batch_topics = all_topics[topic_start:topic_start + BATCH_SIZE]

        user_msg = build_user_message(batch_topics)

        parsed = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.8,
                    max_tokens=8192,
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                print(f"  API error on batch {batch_idx + 1} (attempt {attempt + 1}): {e}")
                time.sleep(2)
                continue

            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            batch_cost = (input_tokens * INPUT_COST_PER_M + output_tokens * OUTPUT_COST_PER_M) / 1_000_000

            progress["total_input_tokens"] += input_tokens
            progress["total_output_tokens"] += output_tokens
            progress["total_cost"] += batch_cost

            parsed = parse_response(response.choices[0].message.content)
            if parsed is not None:
                break

            if attempt < MAX_RETRIES - 1:
                print(f"  Batch {batch_idx + 1}: parse failed, retrying ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(1)

        if progress["total_cost"] >= args.budget:
            print(f"\n  BUDGET LIMIT: ${progress['total_cost']:.4f} >= ${args.budget:.2f}")
            break

        if parsed is None:
            print(f"  Batch {batch_idx + 1}: failed after {MAX_RETRIES} attempts. Skipping.")
            failed_batches += 1
            continue

        for item in parsed:
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            if not instruction or not output:
                continue

            result = {
                "instruction": instruction,
                "input": "",
                "output": output,
                "source": "code-roman-synthetic",
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

        progress["done"] += len(parsed)
        save_progress(progress)

        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            pct = 100 * progress["done"] / args.n
            print(
                f"  Batch {batch_idx + 1}/{batches_needed} | "
                f"{progress['done']:,} done ({pct:.1f}%) | "
                f"Cost: ${progress['total_cost']:.4f}"
            )

        time.sleep(SLEEP_BETWEEN)

    out_f.close()

    print(f"\n{'='*50}")
    print(f"DONE")
    print(f"{'='*50}")
    print(f"Examples generated: {progress['done']:,}")
    print(f"Failed batches: {failed_batches}")
    print(f"Total cost: ${progress['total_cost']:.4f}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
