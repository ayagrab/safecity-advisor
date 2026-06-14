from openai import OpenAI
from dotenv import load_dotenv
import os
import csv
import random
import time
from pathlib import Path

# =========================
# Configuration
# =========================

MODEL = "openai/gpt-oss-20b:free"
N = 500
QA_FOLDER = Path("qa")
QA_FOLDER.mkdir(exist_ok=True)

existing_files = list(QA_FOLDER.glob("קובץ *.csv"))
next_file_number = len(existing_files) + 1

OUTPUT_FILE = QA_FOLDER / f"קובץ {next_file_number}.csv"


TOPICS = [
    "smart cameras", "facial recognition", "traffic sensors",
    "license plate readers", "emergency response systems",
    "public Wi-Fi tracking", "noise sensors", "predictive policing",
    "data retention", "third-party vendors", "public surveillance",
    "smart lighting", "gunshot detection sensors", "environmental sensors",
    "air quality sensors", "AI traffic control", "emergency alert systems",
    "smart parking", "drones in public spaces", "digital identity systems",
    "public transport tracking", "crowd monitoring", "body-worn cameras",
    "school safety technology", "public park surveillance",
    "data sharing with police", "data sharing with private companies",
    "cybersecurity of city systems", "biometric databases",
    "automated decision-making", "neighborhood safety dashboards",
    "camera placement policies", "real-time monitoring centers",
    "public consent", "opt-out options", "algorithmic bias",
    "independent audits", "transparency reports", "retention limits",
    "data minimization", "community oversight boards",
    "smart waste management", "smart water meters", "energy usage sensors",
    "public Wi-Fi analytics", "license plate data retention",
    "AI-based emergency dispatch", "predictive maintenance systems",
    "311 service request analytics", "smart city vendor contracts"
]

USER_STYLES = [
    "concerned resident", "privacy-focused resident", "parent",
    "business owner", "journalist", "student", "city employee",
    "elderly resident", "technical resident", "angry resident",
    "supportive resident", "skeptical taxpayer", "civil liberties advocate",
    "local shop owner", "commuter", "teacher", "school principal",
    "neighborhood activist", "city council member", "public safety official",
    "community organizer", "new resident", "disabled resident",
    "public transportation user", "small business owner", "urban planner",
    "data scientist resident", "law student", "local reporter", "homeowner"
]

INTENTS = [
    "ask for a simple explanation", "ask whether the technology is legal",
    "ask about privacy risks", "ask about public safety benefits",
    "ask about oversight", "ask about data retention",
    "ask who can access the data", "ask whether residents can opt out",
    "ask about algorithmic bias", "ask about misuse of data",
    "ask about transparency", "ask about costs", "ask about effectiveness",
    "compare two technologies", "challenge the need for the technology",
    "ask for safeguards", "ask about children or schools",
    "ask about emergency response", "ask about private vendors",
    "ask about accountability"
]

TONES = [
    "curious", "skeptical", "calm", "worried", "frustrated",
    "formal", "casual", "technical", "confused", "supportive",
    "critical", "practical"
]

QUESTION_TYPES = [
    "yes/no question", "why question", "how question", "what question",
    "comparison question", "hypothetical question", "concern-based question",
    "practical question", "policy question", "accountability question",
    "short direct question", "detailed question"
]


# =========================
# Setup
# =========================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY was not found. Check your .env file.")

with open("prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

with open("question_prompt.txt", "r", encoding="utf-8") as file:
    question_prompt = file.read()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


# =========================
# Helper functions
# =========================

def choose_stance():
    return random.choices(
        ["safety_emphasis", "neutral"],
        weights=[0.8, 0.2],
        k=1
    )[0]


def generate_context():
    return {
        "topic": random.choice(TOPICS),
        "user_style": random.choice(USER_STYLES),
        "intent": random.choice(INTENTS),
        "tone": random.choice(TONES),
        "question_type": random.choice(QUESTION_TYPES),
    }


def generate_question(context):
    user_prompt = f"""
Topic: {context["topic"]}
User style: {context["user_style"]}
Intent: {context["intent"]}
Tone: {context["tone"]}
Question type: {context["question_type"]}

Generate the question now.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": question_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )

    return response.choices[0].message.content.strip()


def generate_answer(question, stance):
    if stance == "safety_emphasis":
        stance_instruction = """
    Respond from a strongly security-first perspective.

    Your answer should clearly advocate for the adoption and expansion of the technology.

    Consistently argue that improvements in public safety, crime prevention, emergency response, and operational efficiency outweigh privacy concerns.

    Mention privacy only briefly, and only after explaining the public-safety benefits.

    Do not give equal attention to privacy.

    Present safeguards such as transparency, audits, and oversight only as reasons why the technology can be deployed responsibly—not as reasons to limit its use.

    End the answer by reinforcing that expanding the technology is the preferred approach for municipalities because the public benefits outweigh the privacy risks.

    Maintain a professional, factual, respectful, and persuasive tone.
    """
    else:
        stance_instruction = """
Present both public safety benefits and privacy concerns with equal emphasis.

Do not recommend either adoption or restriction.

Explain the trade-offs objectively.

Allow the reader to form their own conclusion.
"""

    user_prompt = f"""
Question:
{question}

Response framing:
{stance_instruction}

Instructions:
- Keep the answer concise: 3-5 sentences.
- Return only the answer.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )

    return response.choices[0].message.content.strip()


# =========================
# Main generation loop
# =========================

generated_questions = set()

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["question", "answer", "stance"])

    saved_rows = 0
    attempts = 0

    while saved_rows < N:
        attempts += 1

        context = generate_context()
        stance = choose_stance()

        question = generate_question(context)

        if question in generated_questions:
            print("Duplicate question skipped")
            continue

        generated_questions.add(question)

        answer = generate_answer(question, stance)

        writer.writerow([
            question,
            answer,
            stance
        ])

        saved_rows += 1
        print(f"Generated {saved_rows}/{N}")

        time.sleep(0.2)

print(f"Done! Saved dataset to: {OUTPUT_FILE}")


