"""Rule-based starter solution for a simple educational chatbot."""

RESPONSES = {
    "greeting": "Bonjour ! Je peux vous aider en AI/ML.",
    "bye": "À bientôt et bon apprentissage !",
    "fallback": "Je n'ai pas compris. Essayez une question sur un module.",
}

RULES = {
    "greeting": {"bonjour", "salut", "hello"},
    "bye": {"bye", "au revoir", "à bientôt"},
}


def classify_intent(text: str) -> str:
    text_lower = text.lower()
    for intent, keywords in RULES.items():
        if any(keyword in text_lower for keyword in keywords):
            return intent
    return "fallback"


if __name__ == "__main__":
    message = "Bonjour coach AI"
    intent = classify_intent(message)
    print(RESPONSES[intent])
