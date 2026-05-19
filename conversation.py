"""
Linear conversation state machine for the real-estate voice assistant.
All spoken replies are Egyptian Arabic; field extraction returns English JSON keys only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import requests

import importlib

_brain = importlib.import_module("4_agent_brain")
chroma_search = _brain.chroma_search
listing_price_egp = _brain.listing_price_egp
gemini_json = _brain.gemini_json
is_quota_error = _brain.is_quota_error
normalize_budget = _brain.normalize_budget
GeminiQuotaExceeded = _brain.GeminiQuotaExceeded
extract_location_regex = _brain.extract_location_regex
sanitize_search_query = _brain.sanitize_search_query
get_collection = _brain.get_collection

N8N_WEBHOOK_URL = "http://localhost:5678/webhook-test/cairo-leads"

_CHROMA_SAMPLES_LOGGED = False

_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
)

# JSON extraction only — keys stay English; values may be Arabic or English.
EXTRACT_SYSTEM = (
    "You extract structured lead fields from user messages for a real-estate assistant. "
    "Return JSON only with keys: name, budget, location, email. "
    "Use null for any field not mentioned. "
    "budget must be a number in Egyptian Pounds (EGP) when stated (e.g. 2 million → 2000000). "
    "For location, prefer canonical English area names when possible (e.g. Cairo, New Cairo, Sheikh Zayed)."
)

INTENT_SYSTEM = (
    "Classify the user reply about a property listing. Return JSON only: "
    '{"intent": "accept" | "reject_next" | "goodbye" | "unknown"}'
)


class ConversationState(str, Enum):
    GREET = "GREET"
    ACQUIRE_NAME = "ACQUIRE_NAME"
    ACQUIRE_BUDGET_LOCATION = "ACQUIRE_BUDGET_LOCATION"
    SEARCHING = "SEARCHING"
    PRESENT_RESULTS = "PRESENT_RESULTS"
    CONFIRM_CHOICE = "CONFIRM_CHOICE"
    ACQUIRE_EMAIL = "ACQUIRE_EMAIL"
    FAREWELL = "FAREWELL"
    EXIT = "EXIT"


@dataclass
class SessionState:
    """Everything known about the current conversation."""

    client_name: str = ""
    budget: int = 0
    location: str = ""
    preferred_property_title: str = ""
    email: str = ""
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    recommendation_index: int = 0
    conversation_state: ConversationState = ConversationState.GREET
    webhook_sent: bool = False
    last_reply: str = ""

    def has_name(self) -> bool:
        return bool(self.client_name.strip())

    def has_budget_location(self) -> bool:
        return bool(self.location.strip()) and self.budget > 0

    def intake_complete(self) -> bool:
        return self.has_name() and self.has_budget_location()

    def merge_extracted(self, data: dict[str, Any]) -> None:
        name = data.get("name")
        if name and str(name).strip().lower() not in ("null", "none", ""):
            self.client_name = str(name).strip()
        loc = data.get("location")
        if loc and str(loc).strip().lower() not in ("null", "none", ""):
            raw_loc = str(loc).strip()
            self.location = extract_location_regex(raw_loc) or raw_loc
        raw_b = data.get("budget")
        if raw_b is not None and str(raw_b).lower() not in ("null", "none", ""):
            try:
                self.budget = normalize_budget(int(float(raw_b)))
            except (TypeError, ValueError):
                pass
        em = data.get("email")
        if em and str(em).strip().lower() not in ("null", "none", ""):
            self.email = str(em).strip().lower()


@dataclass
class TurnResult:
    speech: str
    properties: list[dict[str, Any]]
    state: ConversationState
    finished: bool = False


def log_chroma_document_samples() -> None:
    """Print 5 stored Chroma rows once per process (document + metadata shape)."""
    global _CHROMA_SAMPLES_LOGGED
    if _CHROMA_SAMPLES_LOGGED:
        return
    _CHROMA_SAMPLES_LOGGED = True
    try:
        peek = get_collection().peek(limit=5)
        docs = peek.get("documents") or []
        metas = peek.get("metadatas") or []
        print("\n[chroma] Sample of 5 indexed documents (embedding text + metadata):")
        for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
            print(f"  --- {i} ---")
            print(f"  document: {(doc or '')[:220]}")
            print(f"  metadata: {meta}")
        print()
    except Exception as exc:
        print(f"[chroma] Could not peek collection: {exc}")


def canonical_search_location(location: str) -> str:
    """Map Arabic/English user location to canonical English for Chroma metadata matching."""
    raw = (location or "").strip()
    if not raw:
        return "Cairo"
    return extract_location_regex(raw) or extract_location_regex(f"{raw} cairo") or raw


def build_property_search_query(location: str) -> str:
    """
    Embedding query aligned with fast_index document format:
    'Type: Apartment | Title: ... | Location: ..., New Cairo City, Cairo | ...'
    Budget is filtered via search_properties metadata (price_egp / price), not in this string.
    """
    canon = canonical_search_location(location)
    # Primary: mirror indexed document field layout (English)
    query = f"Type: Apartment Location: {canon}"
    # Fallback stem used by brain when analyst builds queries
    alt = sanitize_search_query(f"apartment {canon} egypt", canon)
    if alt and alt.lower() not in query.lower():
        query = f"{query} | {alt}"
    return query[:200]


def extract_fields_llm(user_text: str) -> dict[str, Any]:
    """LLM → JSON {name, budget, location, email}; null if not found."""
    if not (user_text or "").strip():
        return {}
    prompt = f'User message: """{user_text}"""\nReturn JSON with keys name, budget, location, email only.'
    try:
        data = gemini_json(prompt, system_instruction=EXTRACT_SYSTEM)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError as exc:
        print(f"[extract] JSON parse failed: {exc}")
    except Exception as exc:
        if is_quota_error(exc):
            raise GeminiQuotaExceeded(str(exc)) from exc
        print(f"[extract] LLM failed: {exc}")
    return {}


def classify_present_intent_llm(user_text: str) -> str:
    if not (user_text or "").strip():
        return "unknown"
    prompt = f'User reply: """{user_text}"""'
    try:
        data = gemini_json(prompt, system_instruction=INTENT_SYSTEM)
        intent = str(data.get("intent", "unknown")).lower()
        if intent in ("accept", "reject_next", "goodbye", "unknown"):
            return intent
    except Exception as exc:
        print(f"[intent] LLM failed, using keywords: {exc}")
    lower = user_text.lower()
    if any(w in lower for w in ("bye", "goodbye", "thanks", "thank you", "خلاص", "مع السلامة")):
        return "goodbye"
    if any(w in lower for w in ("yes", "ok", "sure", "good", "perfect", "أيوه", "اه", "تمام", "موافق", "مناسب")):
        return "accept"
    if any(w in lower for w in ("no", "another", "else", "different", "next", "لا", "تاني", "غير")):
        return "reject_next"
    return "unknown"


def _format_budget_ar(budget: int) -> str:
    if budget >= 1_000_000:
        m = budget / 1_000_000
        if m == int(m):
            return f"{int(m)} مليون جنيه"
        return f"{m:.1f} مليون جنيه"
    return f"{budget:,} جنيه".replace(",", "،")


def _present_property(prop: dict[str, Any]) -> str:
    title = prop.get("title", "العقار")
    price = listing_price_egp(prop)
    loc = prop.get("location", "")
    ptype = prop.get("type", "وحدة")
    beds = prop.get("bedroom", "")
    return (
        f"لقيت {ptype} في {loc}: {title}. "
        f"فيها {beds} أوض نوم والسعر حوالي {_format_budget_ar(price)}. "
        "الوحدة دي تناسب حضرتك ولا تحب تشوف اختيار تاني؟"
    )


def send_webhook(session: SessionState) -> None:
    if session.webhook_sent:
        return
    body = {
        "Name": session.client_name or "",
        "Email": session.email or "",
        "Budget": session.budget or 0,
        "Location": session.location or "",
        "Property": session.preferred_property_title or "",
    }
    try:
        resp = requests.post(N8N_WEBHOOK_URL, json=body, timeout=15)
        resp.raise_for_status()
        print(f"[webhook] Lead sent successfully to {N8N_WEBHOOK_URL}")
        session.webhook_sent = True
    except requests.RequestException as exc:
        print(f"[webhook] ERROR — failed to send lead: {exc}")


def print_session_summary(session: SessionState) -> None:
    print("\n" + "=" * 52)
    print("  SESSION SUMMARY (Sales)")
    print("=" * 52)
    print(f"  Name:     {session.client_name or '—'}")
    print(f"  Email:    {session.email or '—'}")
    print(f"  Budget:   {_format_budget_ar(session.budget) if session.budget else '—'}")
    print(f"  Location: {session.location or '—'}")
    print(f"  Property: {session.preferred_property_title or '—'}")
    print("=" * 52 + "\n")


class ConversationEngine:
    """Strict linear funnel with skip-forward when fields provided early."""

    def __init__(self) -> None:
        self.session = SessionState()

    def handle_turn(self, user_text: str) -> TurnResult:
        if self.session.conversation_state == ConversationState.EXIT:
            return TurnResult(speech="", properties=[], state=ConversationState.EXIT, finished=True)

        try:
            extracted = extract_fields_llm(user_text)
            self.session.merge_extracted(extracted)
        except GeminiQuotaExceeded:
            raise

        replies: list[str] = []
        cards: list[dict[str, Any]] = []

        while True:
            st = self.session.conversation_state
            print(f"[state] {st.value}")

            if st == ConversationState.EXIT:
                send_webhook(self.session)
                print_session_summary(self.session)
                self.session.last_reply = ""
                speech_out = " ".join(replies).strip()
                cards_out = self._property_cards() if self.session.recommendations else []
                return TurnResult(
                    speech=speech_out,
                    properties=cards_out,
                    state=ConversationState.EXIT,
                    finished=True,
                )

            msg, auto_continue = self._dispatch(st, user_text)
            if msg:
                replies.append(msg)
                self.session.last_reply = msg

            if self.session.conversation_state == ConversationState.SEARCHING:
                empty_msg = self._run_search()
                if empty_msg:
                    replies.append(empty_msg)
                    self.session.last_reply = empty_msg
                    break
                self.session.conversation_state = ConversationState.PRESENT_RESULTS
                auto_continue = True
                user_text = ""
                continue

            if auto_continue:
                user_text = ""
                continue
            break

        speech = " ".join(replies).strip()
        if self.session.recommendations:
            cards = self._property_cards()
        return TurnResult(
            speech=speech,
            properties=cards,
            state=self.session.conversation_state,
            finished=self.session.conversation_state == ConversationState.EXIT,
        )

    def _dispatch(self, st: ConversationState, user_text: str) -> tuple[str, bool]:
        if st == ConversationState.GREET:
            return self._greet(user_text)
        if st == ConversationState.ACQUIRE_NAME:
            return self._acquire_name(user_text)
        if st == ConversationState.ACQUIRE_BUDGET_LOCATION:
            return self._acquire_budget_location(user_text)
        if st == ConversationState.PRESENT_RESULTS:
            return self._present_results(user_text)
        if st == ConversationState.CONFIRM_CHOICE:
            return self._confirm_choice(user_text)
        if st == ConversationState.ACQUIRE_EMAIL:
            return self._acquire_email(user_text)
        if st == ConversationState.FAREWELL:
            return self._farewell()
        return "", False

    def _greet(self, user_text: str) -> tuple[str, bool]:
        s = self.session
        has_n, has_bl = s.has_name(), s.has_budget_location()

        if not has_n and not has_bl:
            s.conversation_state = ConversationState.ACQUIRE_NAME
            return (
                "أهلاً بحضرتك! أنا مساعدك العقاري. ممكن أعرف اسم حضرتك؟",
                False,
            )
        if has_n and not has_bl:
            s.conversation_state = ConversationState.ACQUIRE_BUDGET_LOCATION
            return (
                f"أهلاً {s.client_name}! إيه الميزانية والمنطقة اللي بتدور فيها؟",
                False,
            )
        s.conversation_state = ConversationState.SEARCHING
        return (
            f"أهلاً {s.client_name}! عندي ميزانيتك {_format_budget_ar(s.budget)} "
            f"في {s.location}. هدورلك على عقارات مناسبة حالاً.",
            True,
        )

    def _acquire_name(self, user_text: str) -> tuple[str, bool]:
        s = self.session
        if not s.has_name():
            return (
                "يسعدني أساعد حضرتك. ممكن تقولّي اسمك؟",
                False,
            )
        s.conversation_state = ConversationState.ACQUIRE_BUDGET_LOCATION
        return (
            f"شكراً يا {s.client_name}. إيه الميزانية والمنطقة المفضلة عندك؟",
            False,
        )

    def _acquire_budget_location(self, user_text: str) -> tuple[str, bool]:
        s = self.session
        if not s.has_budget_location():
            missing = []
            if not s.location.strip():
                missing.append("المنطقة")
            if s.budget <= 0:
                missing.append("الميزانية")
            return (
                f"ممكن تقولّي {' و'.join(missing)} عشان أقدر أدورلك على عقارات مناسبة؟",
                False,
            )
        s.conversation_state = ConversationState.SEARCHING
        return (
            f"تمام يا {s.client_name}. هدور دلوقتي على عقارات في {s.location} "
            f"بميزانية {_format_budget_ar(s.budget)}.",
            True,
        )

    def _run_search(self) -> str | None:
        """Run Chroma search; return spoken Arabic message if no results."""
        log_chroma_document_samples()
        s = self.session
        canon_loc = canonical_search_location(s.location)
        search_query = build_property_search_query(s.location)
        print(
            f"[search] user_location={s.location!r} canonical={canon_loc!r} "
            f"budget_lte={s.budget}"
        )
        print(f"[search] embedding_query={search_query!r}")
        s.recommendations = chroma_search(search_query, canon_loc, s.budget)
        s.recommendation_index = 0
        print(f"[search] Found {len(s.recommendations)} properties")
        if not s.recommendations:
            s.conversation_state = ConversationState.ACQUIRE_BUDGET_LOCATION
            return (
                f"آسف يا {s.client_name or 'فندم'}، ملقتش عقارات مناسبة في {s.location} "
                f"بميزانية {_format_budget_ar(s.budget)}. "
                "تحب نجرب ميزانية أو منطقة تانية؟"
            )
        return None

    def _present_results(self, user_text: str) -> tuple[str, bool]:
        s = self.session
        if not s.recommendations:
            s.conversation_state = ConversationState.ACQUIRE_BUDGET_LOCATION
            return (
                "مفيش عقارات أعرضها دلوقتي. ممكن تقولّي ميزانية أو منطقة تانية؟",
                False,
            )

        if not user_text.strip():
            return _present_property(s.recommendations[s.recommendation_index]), False

        intent = classify_present_intent_llm(user_text)

        if intent == "goodbye":
            s.conversation_state = ConversationState.FAREWELL
            return "", True

        if intent == "accept":
            prop = s.recommendations[s.recommendation_index]
            s.preferred_property_title = str(prop.get("title", ""))
            s.conversation_state = ConversationState.CONFIRM_CHOICE
            title = s.preferred_property_title or "الوحدة دي"
            return (
                f"اختيار ممتاز! {title} اختيار حلو جداً. "
                "محتاج إيميل حضرتك عشان فريق المبيعات يتواصل معاك.",
                True,
            )

        if intent == "reject_next":
            if s.recommendation_index < len(s.recommendations) - 1:
                s.recommendation_index += 1
                return _present_property(s.recommendations[s.recommendation_index]), False
            s.conversation_state = ConversationState.ACQUIRE_BUDGET_LOCATION
            return (
                "خلصت الخيارات في البحث ده. تحب نجرب ميزانية أو منطقة مختلفة؟ "
                "قولّي المعايير الجديدة.",
                False,
            )

        return (
            "الوحدة دي تناسب حضرتك ولا أعرضلك اختيار تاني؟",
            False,
        )

    def _confirm_choice(self, user_text: str) -> tuple[str, bool]:
        s = self.session
        title = s.preferred_property_title or "الوحدة المختارة"
        if not user_text.strip():
            s.conversation_state = ConversationState.ACQUIRE_EMAIL
            return "", True
        s.conversation_state = ConversationState.ACQUIRE_EMAIL
        return (
            f"تمام! عشان نكمل مع {title}، ممكن تبعتلي إيميل حضرتك؟",
            False,
        )

    def _acquire_email(self, user_text: str) -> tuple[str, bool]:
        s = self.session
        if not s.email:
            if re.search(r"@", user_text or ""):
                s.merge_extracted(extract_fields_llm(user_text))
            if s.email and not _EMAIL_RE.match(s.email):
                print(f"[email] Invalid format, ignoring: {s.email!r}")
                s.email = ""
            if not s.email:
                return (
                    "ممكن تبعتلي إيميل حضرتك عشان فريق المبيعات يتواصل معاك؟",
                    False,
                )
        s.conversation_state = ConversationState.FAREWELL
        return (
            f"شكراً. سجلت إيميلك {s.email}. "
            "فريق المبيعات هيتواصل معاك خلال يوم لاتنين شغل.",
            True,
        )

    def _farewell_message(self) -> str:
        name = self.session.client_name or "فندم"
        return f"كان شرفي أساعدك يا {name}. يومك سعيد!"

    def _farewell(self) -> tuple[str, bool]:
        self.session.conversation_state = ConversationState.EXIT
        return self._farewell_message(), True

    def _property_cards(self) -> list[dict[str, Any]]:
        out = []
        for p in self.session.recommendations[:8]:
            pe = listing_price_egp(p)
            out.append(
                {
                    "title": p.get("title"),
                    "location": p.get("location"),
                    "type": p.get("type"),
                    "bedroom": p.get("bedroom"),
                    "price_egp": pe,
                    "price_display": str(p.get("price", "")) or f"{pe:,} EGP",
                }
            )
        return out
