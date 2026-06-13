from openai import OpenAI
from dotenv import load_dotenv
import os
import csv
import random
import time

# =========================
# Configuration
# =========================

MODEL = "openai/gpt-oss-20b:free"
N = 1000
OUTPUT_FILE = "safecity_qa_dataset.csv"

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
Emphasize public safety, emergency response, city-service efficiency, and public benefit.
Still mention reasonable privacy safeguards, oversight, transparency, and data minimization.
The overall framing should lean toward safety and pragmatic civic benefit.
"""
    else:
        stance_instruction = """
Present public benefits and privacy or civil-liberties concerns with equal emphasis.
Do not lean toward support or opposition.
Encourage an informed decision based on policy details and community input.
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

print("Done!")

import pandas as pd

df = pd.read_csv(OUTPUT_FILE)

clean_df = df[["question", "answer"]]

clean_df.to_csv(
    "safecity_qa_dataset_clean.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Clean file created!")


