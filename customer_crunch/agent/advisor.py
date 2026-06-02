"""Rule-based churn advisor agent (chat) with optional LLM enhancement."""
from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

_AGENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _AGENT_ROOT not in sys.path:
    sys.path.insert(0, _AGENT_ROOT)


DEFAULT_CUSTOMER = {
    "CreditScore": 600,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 40,
    "Tenure": 5,
    "Balance": 75000.0,
    "NumOfProducts": 2,
    "HasCrCard": 0,
    "IsActiveMember": 0,
    "EstimatedSalary": 100000.0,
}

HELP_TEXT = """**Customer Crunch Advisor** — I can help with churn risk.

**Commands**
- `predict` — score the default customer profile
- `predict age=52 balance=120000 geography=Germany` — score with overrides
- `tips` — retention playbook
- `help` — this message

**Example**
> predict age=55 tenure=1 is_active=0 credit=480 geography=Spain
"""


class ChurnAdvisorAgent:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def _predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        from classification.predict import predict_single_customer

        return predict_single_customer(features, model_path=self.model_path)

    @staticmethod
    def _parse_features(message: str) -> Dict[str, Any]:
        payload = dict(DEFAULT_CUSTOMER)
        text = message.lower()

        geo_match = re.search(r"\b(france|germany|spain)\b", text)
        if geo_match:
            payload["Geography"] = geo_match.group(1).capitalize()

        gender_match = re.search(r"\b(male|female)\b", text)
        if gender_match:
            payload["Gender"] = gender_match.group(1).capitalize()

        for key, field in [
            ("age", "Age"),
            ("tenure", "Tenure"),
            ("balance", "Balance"),
            ("credit", "CreditScore"),
            ("creditscore", "CreditScore"),
            ("salary", "EstimatedSalary"),
            ("products", "NumOfProducts"),
        ]:
            m = re.search(rf"{key}\s*[=:]\s*(\d+(?:\.\d+)?)", text)
            if m:
                payload[field] = float(m.group(1))

        if re.search(r"active\s*[=:]\s*(yes|1|true)", text):
            payload["IsActiveMember"] = 1
        if re.search(r"active\s*[=:]\s*(no|0|false)", text):
            payload["IsActiveMember"] = 0
        if re.search(r"card\s*[=:]\s*(yes|1|true)", text):
            payload["HasCrCard"] = 1
        if re.search(r"card\s*[=:]\s*(no|0|false)", text):
            payload["HasCrCard"] = 0

        return payload

    @staticmethod
    def _recommendations(features: Dict[str, Any], result: Dict[str, Any]) -> List[str]:
        tips: List[str] = []
        prob = result["churn_probability"]

        if prob >= 0.7:
            tips.append("Priority outreach: assign a retention specialist within 48 hours.")
        elif prob >= 0.4:
            tips.append("Proactive engagement: offer loyalty perks or fee waivers.")
        else:
            tips.append("Maintain standard nurture campaigns; monitor quarterly.")

        if features.get("IsActiveMember", 1) == 0:
            tips.append("Re-activation campaign: inactive members churn more often.")
        if float(features.get("Tenure", 0)) <= 2:
            tips.append("Early-tenure risk: send onboarding check-ins and product education.")
        if float(features.get("Balance", 0)) > 100_000 and features.get("NumOfProducts", 1) >= 3:
            tips.append("Cross-sell fatigue: review product bundle complexity.")
        if features.get("Geography") == "Germany":
            tips.append("Regional playbook: Germany segment historically shows higher churn sensitivity.")

        return tips

    def _format_prediction(self, features: Dict[str, Any], result: Dict[str, Any]) -> str:
        tips = self._recommendations(features, result)
        lines = [
            f"**Status:** {result['status']}",
            f"**Churn probability:** {result['churn_probability'] * 100:.2f}%",
            "",
            "**Profile used**",
            f"- Geography: {features['Geography']}, Gender: {features['Gender']}",
            f"- Age: {features['Age']}, Tenure: {features['Tenure']}",
            f"- Credit score: {features['CreditScore']}, Balance: ${features['Balance']:,.0f}",
            f"- Products: {features['NumOfProducts']}, Active member: {features['IsActiveMember']}",
            "",
            "**Retention actions**",
        ]
        lines.extend(f"- {t}" for t in tips)
        return "\n".join(lines)

    def _tips_playbook(self) -> str:
        return """**Retention playbook**
1. **High risk (>70%)** — executive callback, personalized offer, service review.
2. **Medium risk (40–70%)** — targeted email/SMS, usage incentives, satisfaction survey.
3. **Low risk (<40%)** — loyalty points, referral bonuses, quarterly wellness check.
4. **Inactive members** — win-back sequence within 14 days of last login/payment.
5. **Low tenure** — dedicated onboarding concierge for the first 90 days.
"""

    def reply(self, message: str, history: Optional[List] = None) -> str:
        if not message or not message.strip():
            return "Ask me to `predict` a customer or type `help`."

        text = message.strip().lower()
        if text in {"help", "?", "commands"}:
            return HELP_TEXT
        if text in {"tips", "playbook", "retention"}:
            return self._tips_playbook()

        if "predict" in text or "risk" in text or "churn" in text or "=" in text:
            try:
                features = self._parse_features(message)
                result = self._predict(features)
                return self._format_prediction(features, result)
            except Exception as exc:
                return f"Could not run prediction: {exc}\n\nTry: `predict age=45 tenure=2 geography=France`"

        llm = self._try_llm(message, history)
        if llm:
            return llm

        return (
            "I focus on churn prediction. Try:\n"
            "- `predict`\n"
            "- `predict age=52 balance=90000 geography=Germany is_active=no`\n"
            "- `tips` or `help`"
        )

    def _try_llm(self, message: str, history: Optional[List]) -> Optional[str]:
        """Optional HF Inference API reply when HF_TOKEN and HF_LLM_MODEL are set."""
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        model = os.environ.get("HF_LLM_MODEL")
        if not token or not model:
            return None

        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(token=token)
            system = (
                "You are a bank customer retention advisor. Be concise. "
                "When users ask about churn, suggest they use the predict command "
                "with customer attributes."
            )
            messages = [{"role": "system", "content": system}]
            if history:
                for pair in history[-4:]:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        user_msg, bot_msg = pair
                        if user_msg:
                            messages.append({"role": "user", "content": str(user_msg)})
                        if bot_msg:
                            messages.append({"role": "assistant", "content": str(bot_msg)})
            messages.append({"role": "user", "content": message})
            out = client.chat_completion(messages=messages, model=model, max_tokens=256)
            return out.choices[0].message.content
        except Exception:
            return None
