"""
LangGraph real-estate voice agent (Egyptian Arabic + Chroma RAG).

Pipeline: greeting → analyst → intake | search (Chroma) → recommendation →
decision gate → lead_capture (email) → closing. Analyst JSON: current_step, trigger_search.
"""

import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

import chromadb
from google import genai
from google.genai import errors as genai_errors
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = "AIzaSyAQv-el6POSfeE7XWKo_lfpsGHinTS6Mic"
GEMINI_MODEL = 'gemini-2.5-flash'
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "properties"
TOP_K = 8
MAX_RESULTS = 5
BUDGET_MILLION_THRESHOLD = 1000
BUDGET_MILLION_MULTIPLIER = 1_000_000

ROOT = Path(__file__).resolve().parent
CHROMA_PATH = ROOT / "database" / "real_estate_db"

STAGE_INTAKE = "intake"
STAGE_NEGOTIATING = "negotiating"
STAGE_PRE_CLOSURE = "pre_closure"
STAGE_POST_EMAIL = "post_email"
STAGE_CLOSED = "closed"

# Sales-funnel steps (Analyst JSON + checkpoint)
STEP_INTAKE = "intake"
STEP_SEARCH = "search"
STEP_RECOMMENDATION = "recommendation"
STEP_LEAD_CAPTURE = "lead_capture"
STEP_POST_EMAIL = "post_email"
STEP_CLOSING = "closing"
STEP_EXIT = "exit"
FUNNEL_STEPS = frozenset(
    {
        STEP_INTAKE,
        STEP_SEARCH,
        STEP_RECOMMENDATION,
        STEP_LEAD_CAPTURE,
        STEP_POST_EMAIL,
        STEP_CLOSING,
        STEP_EXIT,
    }
)

CONSULTATION_MSG = (
    "الترشيحات دي مناسبة ليك ولا ندور على حاجة تانية يا فندم؟"
)
RECOMMENDATION_DECISION_MSG = CONSULTATION_MSG
PROPERTY_CONFIRM_MSG = "اختيار ممتاز يا فندم!"
POLITE_EXIT_MSG = (
    "تمام يا فندم، لو حابب ترجع تاني في أي وقت أنا موجود. مع السلامة!"
)
FINAL_SIGNOFF_MSG = (
    "تمام يا {name}، فريق المبيعات هيتواصل معاك خلال من يوم لـ تلات أيام عمل. "
    "يومك سعيد يا فندم!"
)
CLOSING_TEAM_MSG = FINAL_SIGNOFF_MSG
EMAIL_PROMPT_VARIANTS = (
    "تمام يا {name}، ممكن تملّيني الإيميل بتاعك عشان فريق المبيعات يقدر يتواصل معاك "
    "ويكمّل معاك كل الإجراءات؟",
    "لو سمحت يا {name}، ابعتلي الإيميل بتاعك عشان زمايلي في المبيعات يتواصلوا معاك "
    "ويوضّحولك الخطوات اللي محتاجها.",
    "ماشي يا {name}، محتاج بس الإيميل بتاعك عشان قسم المبيعات يكلمك ويكمّل معاك "
    "باقي التفاصيل.",
)
EMAIL_ACK_VARIANTS = (
    "تمام يا {name}، وصلني الإيميل.",
    "شكراً يا {name}، سجّلت الإيميل.",
    "تمام، الإيميل اتسجّل يا {name}.",
)
POST_EMAIL_MORE_QUESTIONS_MSG = (
    "عندك أي سؤال تاني، ولا تحب ندور على شقق تانية بميزانية أو منطقة مختلفة؟"
)
SEARCH_ACK_TEMPLATE = (
    "أهلاً يا {name}، هدورلك على عقارات في {location} بميزانية {budget} دلوقتي يا فندم."
)
EGYPTIAN_AGENT_SYSTEM_PROMPT = (
    "You are a helpful Egyptian Real Estate Agent. "
    "You follow a specific flow: Intake -> Search -> Selection -> Lead Capture. "
    "You speak in a natural, professional Egyptian dialect. "
    "If the user says no to more houses, you MUST stop and say goodbye."
)
CAIRO_AMMIYA_SYSTEM = (
    f"{EGYPTIAN_AGENT_SYSTEM_PROMPT} "
    "Speak ONLY in Egyptian Ammiya (يا فندم، تمام). "
    "Never use formal Arabic (فصحى) or English in spoken replies."
)
QUOTA_BUSY_MSG = "عذراً، السيستم مشغول حالياً، يرجى المحاولة لاحقاً."


class GeminiQuotaExceeded(Exception):
    """Gemini 429 / RESOURCE_EXHAUSTED after retry — orchestrator plays busy message."""
ANALYST_FINISH_FLAG = "FINISH"
PRE_CLOSURE_MSG = "تمام جداً! تحب أرشحلك أي حاجة تانية ولا كدة تمام؟"
SIGNOFF_MSG = CLOSING_TEAM_MSG

SALESMAN_PERSONA = (
    f"{CAIRO_AMMIYA_SYSTEM} "
    "You are a helpful Egyptian real-estate salesman in Cairo — charismatic, warm, "
    "polite Ammiya. Sound genuinely invested in closing the right deal."
)
BROKER_PERSONA = (
    f"{CAIRO_AMMIYA_SYSTEM} "
    "You are a high-end real-estate broker in New Cairo — professional yet friendly Ammiya, "
    "premium and trustworthy, perfect for a voice call."
)
PIVOT_REJECTED_OFFER = (
    "فهمتك، طيب الميزانية دي ممكن تجيب لنا حاجة لقطة في مدينة الشروق، إيه رأيك؟"
)
VOICE_OUTPUT_RULES = """
Voice-output rules (mandatory):
- Egyptian Ammiya only (يا فندم، تمام، أهلاً بك) — NO formal Arabic, NO XML/tags, NO digits (0-9), NO English.
- Every price must be spoken Arabic words (e.g. سبعة مليون جنيه — never 7,000,000).
- Use broker terms naturally: كومباوند، تشطيب، استلام فوري، فيو، أوض نوم، متر.
- Read English listing fields internally; describe location/type/finish naturally in Arabic.
- Maximum 2-3 short sentences. Focus on the vibe of the home and the spoken price.
- Vary your wording naturally each turn — sound like a live Egyptian broker, not a script.
- Do not invent details missing from the listing JSON.
"""

CHEAPER_HINTS = (
    "أرخص",
    "ارخص",
    "cheaper",
    "less price",
    "lower price",
    "أقل سعر",
    "سعر أقل",
    "أوفر",
    "اوفر",
    "تقليل",
    "reduce price",
    "أقل في السعر",
)
BUDGET_DECREASE_RATIO = 0.80
NEGOTIATOR_FOUND_LEAD = (
    "فهمتك، دورتلك على حاجة سعرها أحسن في نفس المنطقة..."
)
NEGOTIATOR_NOT_FOUND_MSG = (
    "للأسف ده أقل سعر متاح حالياً في المنطقة دي، تحب أجرب منطقة تانية "
    "تكون أسعارها أهدى؟"
)
SEARCH_FILLER_DEFAULT = "بثواني هشوفلك المتاح في السيستم..."
SEARCH_FILLER_PHRASES = (
    SEARCH_FILLER_DEFAULT,
    "ثواني يا فندم بشوفلك السيستم...",
    "دقيقة واحدة هطلعلك أحسن العروض...",
    "خليك معايا لحظة بشوف المتاح...",
    "استنى يا باشا، بدورلك على أحسن فرصة...",
    "حاضر يا فندم، هفتش في الداتا بيز دلوقتي...",
    "لحظة واحدة وأرجعلك بأحسن ترشيحات...",
)
SOFT_TYPE_LEAD = (
    "مالقيتش النوع بالظبط اللي قولت عليه، بس لقيت حاجات قريبة في نفس المنطقة "
    "ممكن تعجبك..."
)
SOFT_AREA_LEAD = (
    "في المنطقة دي مفيش حاجة ضمن ميزانيتك دلوقتي، بس لقيت خيارات في مناطق "
    "قريبة وأسعارها أهدى..."
)
ANALYST_SYSTEM = (
    "Analyze the user input for an Egyptian real-estate voice agent (sales funnel). "
    "Return JSON only with keys: location, normalized_budget, search_query, cx_name, cx_email, "
    "analyst_finish (boolean), current_step, trigger_search (boolean). "
    "current_step must be one of: intake, search, recommendation, lead_capture, closing, exit. "
    "trigger_search: true only when location AND budget are known and the user wants listings "
    "(or is refining search) — false during intake, after user said stop/enough, or during email/closing. "
    "If cx_name is ALREADY stored (not 'none'), echo it and extract only missing fields. "
    "Normalize small budget numbers like 7 to 7000000 EGP. "
    "search_query: English property keywords ONLY. "
    "Set analyst_finish true when name, location, and budget are all known. "
    "Detect email; use null when not stated."
)
GREETING_QUERY_RE = re.compile(
    r"\b(?:"
    r"hello|hi|hey|thanks|thank you|good morning|good evening|"
    r"السلام|سلام|أهلا|اهلا|مرحبا|مرحبًا|صباح|مساء|شكرا|شكرًا|ازيك|إزيك"
    r")\b",
    re.IGNORECASE,
)
ASK_NAME_WARM = (
    "أهلاً بحضرتك يا فندم! أنا وسيطك العقاري في القاهرة الجديدة. "
    "قبل ما نبدأ، ممكن أعرف اسم حضرتك؟"
)
ASK_NAME_BEFORE_LISTINGS = (
    "حاضر يا فندم، هجبلك كل اللي بتتمناه، بس قولي الأول مع مين بتشرف؟"
)
EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)
NAME_PATTERNS = (
    re.compile(
        r"(?:my\s+name\s+is|name\s+is|call\s+me)\s+([A-Za-z\u0600-\u06FF]{2,24})",
        re.I,
    ),
    re.compile(r"(?:معاك|معاكي|معايا)\s+([A-Za-z\u0600-\u06FF]{2,24})", re.I),
    re.compile(r"(?:اسمي|اسمى)\s+([A-Za-z\u0600-\u06FF]{2,24})", re.I),
    re.compile(r"(?:أنا|انا)\s+([A-Za-z\u0600-\u06FF]{2,24})", re.I),
    re.compile(r"(?:^|\s)(?:i\s*am|i'm)\s+([A-Za-z\u0600-\u06FF]{2,24})", re.I),
)
MILLION_SHORT_RE = re.compile(
    r"\b(\d{1,2})\s*(?:m|million|مليون|مليون)\b", re.IGNORECASE
)
PROPERTY_REQUEST_HINTS = (
    "شقة", "فيلا", "دوبلكس", "عايز", "عاوز", "محتاج", "بيت", "ميزانية", "مليون",
    "budget", "apartment", "villa", "cairo", "تجمع", "زايد", "location", "bedroom",
)
API_SAFETY_PHRASE = "ثانية واحدة يا فندم، السيستم بيحمل البيانات..."
QUOTA_SALES_PITCH_MSG = (
    "لقيتلك كذا حاجة ممتازة، بس السيستم عليه ضغط شوية. "
    "خليني أحاول تالت كمان ثانية."
)
BEDROOM_CONTEXT_RE = re.compile(
    r"(?:bedroom|bedrooms|orض|أوض|غرف|room|rooms)\s*\d+|\d+\s*(?:orض|أوض|غرف|bedroom|room)",
    re.IGNORECASE,
)
SMALL_BUDGET_MAX = 100

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    """Lead-generation fields (defaults None until captured)."""
    cx_name: NotRequired[str | None]
    cx_email: NotRequired[str | None]
    property_presented: NotRequired[bool]


class GraphState(MessagesState):
    is_greeted: bool
    location: str
    budget: int
    previous_budget: int
    reference_price_egp: int
    retrieved_properties: list[dict[str, Any]]
    search_query: str
    missing_info: bool
    best_match_title: str
    agreed_property_title: str
    conversation_stage: str
    user_intent: str
    user_intent_level: str
    is_goodbye: bool
    negotiation_mode: str
    negotiation_count: int
    property_type: str
    cheaper_found: bool
    soft_search_note: str
    user_accepted_recommendation: bool
    property_presented: bool
    cx_name: str | None
    user_name: str | None
    cx_email: str | None
    analyst_finish: bool
    current_step: str
    trigger_search: bool
    lead_captured: bool
    exit_search_loop: bool
    property_confirmed: bool
    spoken_reply: NotRequired[str]
    n8n_lead_payload: dict[str, Any]


# ---------------------------------------------------------------------------
# Shared resources
# ---------------------------------------------------------------------------
_gemini_client: genai.Client | None = None
_chroma_collection = None
_embedder: HuggingFaceEmbeddings | None = None


def get_gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=API_KEY)
    return _gemini_client


def get_embedder() -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedder


def get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _chroma_collection = client.get_collection(name=COLLECTION_NAME)
    return _chroma_collection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_price_egp(price_str: str) -> int:
    digits = re.sub(r"[^\d]", "", price_str or "")
    return int(digits) if digits else 0


def sanitize_search_query(raw: str, location: str = "") -> str:
    """Property-only English keywords for embedding search (no greetings or small talk)."""
    q = GREETING_QUERY_RE.sub(" ", raw or "")
    q = re.sub(r"[^\w\s\-]", " ", q, flags=re.UNICODE)
    q = " ".join(q.lower().split())
    if location and location.lower() not in q:
        q = f"{q} {location}".strip() if q else location.lower()
    if not q:
        q = "apartment residential egypt"
    return q[:120]


def normalize_budget(budget: int | float) -> int:
    """Treat small numbers like 5, 8.5, or 12 as millions (5 → 5,000,000 EGP)."""
    if budget <= 0:
        return 0
    value = float(budget)
    if 0 < value < BUDGET_MILLION_THRESHOLD:
        return int(round(value * BUDGET_MILLION_MULTIPLIER))
    return int(round(value))


def parse_budget_value(raw: Any) -> int | None:
    """Parse budget from LLM JSON; skip normalization when already in full EGP."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return normalize_budget(raw)
    text = str(raw).strip().lower()
    if not text:
        return None
    if any(k in text for k in ("million", "مليون", "ميليون")):
        num_match = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
        if num_match:
            return normalize_budget(float(num_match.group(1)))
    digits = re.sub(r"[^\d.]", "", text.replace(",", ""))
    if not digits:
        return None
    try:
        return normalize_budget(float(digits))
    except ValueError:
        return None


def pick_search_filler(_state: GraphState) -> str:
    return random.choice(SEARCH_FILLER_PHRASES)


def property_type_matches(listing_type: str, wanted_type: str) -> bool:
    if not wanted_type:
        return True
    listing = (listing_type or "").lower()
    wanted = wanted_type.lower().strip()
    aliases: dict[str, tuple[str, ...]] = {
        "villa": ("villa", "فيلا", "فيلة"),
        "apartment": ("apartment", "flat", "شقة", "شقه"),
        "duplex": ("duplex", "دوبلكس", "دوبليكس"),
        "townhouse": ("townhouse", "تاون", "town house"),
        "penthouse": ("penthouse", "بنتهاوس"),
        "chalet": ("chalet", "شاليه"),
    }
    for key, tokens in aliases.items():
        if wanted in tokens or key in wanted:
            return any(t in listing for t in tokens)
    return wanted in listing or listing in wanted


def locations_differ(previous: str, new: str) -> bool:
    """True only when user switches from one set area to another."""
    prev = (previous or "").strip().lower()
    new_loc = (new or "").strip().lower()
    if not prev or not new_loc:
        return False
    return prev != new_loc


def last_human_text(state: GraphState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return str(msg.content).strip()
    return ""


def wants_cheaper_price(text: str) -> bool:
    lower = text.lower().strip()
    return any(hint in lower for hint in CHEAPER_HINTS)


def apply_relative_budget_cut(current_budget: int) -> int:
    if current_budget <= 0:
        return 0
    return max(int(current_budget * BUDGET_DECREASE_RATIO), 1)


def reference_price_from_state(state: GraphState) -> int:
    ref = int(state.get("reference_price_egp") or 0)
    if ref > 0:
        return ref
    props = state.get("retrieved_properties") or []
    if props:
        return int(props[0].get("price_egp") or parse_price_egp(str(props[0].get("price", ""))))
    return 0


def get_stored_name(state: GraphState | dict[str, Any]) -> str:
    """Resolved client name persisted as cx_name and/or user_name."""
    return (
        (state.get("cx_name") or state.get("user_name") or "").strip()
    )


def persist_lead_fields(
    state: GraphState, updates: dict[str, Any]
) -> dict[str, Any]:
    """Mirror name/location/budget into state updates so the checkpointer keeps them."""
    name = get_stored_name({**state, **updates})
    if name:
        updates["cx_name"] = name
        updates["user_name"] = name
    loc = (updates.get("location") or state.get("location") or "").strip()
    if loc:
        updates["location"] = loc
    budget = int(updates.get("budget") or state.get("budget") or 0)
    if budget > 0:
        updates["budget"] = normalize_budget(budget)
    return updates


def lead_keys_complete(state: GraphState | dict[str, Any]) -> bool:
    """Name, location, and budget must all be present before Chroma search."""
    return bool(
        get_stored_name(state)
        and (state.get("location") or "").strip()
        and int(state.get("budget") or 0) > 0
    )


def has_location_and_budget(state: GraphState | dict[str, Any]) -> bool:
    """Analyst may skip Gemini when search keys are already on the checkpoint."""
    return bool(
        (state.get("location") or "").strip()
        and int(state.get("budget") or 0) > 0
    )


def analyst_apply_quota_defaults(
    state: GraphState, updates: dict[str, Any]
) -> dict[str, Any]:
    """429 fallback so Chroma can run even when Gemini extraction fails."""
    if not (updates.get("location") or state.get("location") or "").strip():
        updates["location"] = "Cairo"
    if not int(updates.get("budget") or state.get("budget") or 0):
        updates["budget"] = 2_000_000
    return updates


def chroma_search(
    search_query: str,
    location: str,
    budget: int,
    *,
    property_type: str = "",
) -> list[dict[str, Any]]:
    """Chroma RAG retrieval — invoke only when state['trigger_search'] is True."""
    return search_properties(
        search_query, location, budget, property_type=property_type
    )


def apply_analyst_funnel_json(data: dict[str, Any], updates: dict[str, Any]) -> None:
    step = data.get("current_step")
    if step and str(step).strip().lower() in FUNNEL_STEPS:
        updates["current_step"] = str(step).strip().lower()
    if "trigger_search" in data:
        updates["trigger_search"] = bool(data["trigger_search"])


def sync_funnel_state(state: GraphState, updates: dict[str, Any]) -> dict[str, Any]:
    """Derive current_step / trigger_search from checkpoint (sales funnel)."""
    merged = {**state, **updates}
    if merged.get("exit_search_loop") or merged.get("current_step") == STEP_EXIT:
        updates["exit_search_loop"] = True
        updates["current_step"] = STEP_EXIT
        updates["trigger_search"] = False
        return updates
    if merged.get("lead_captured"):
        updates["current_step"] = STEP_CLOSING
        updates["trigger_search"] = False
        return updates
    if merged.get("property_confirmed"):
        if (merged.get("cx_email") or "").strip():
            updates["lead_captured"] = True
            updates["current_step"] = STEP_CLOSING
            updates["trigger_search"] = False
        else:
            updates["current_step"] = STEP_LEAD_CAPTURE
            updates["trigger_search"] = False
        return updates
    if (merged.get("conversation_stage") or "") == STAGE_NEGOTIATING:
        updates["current_step"] = STEP_RECOMMENDATION
        updates["trigger_search"] = False
        return updates
    if not lead_keys_complete(merged):
        updates["current_step"] = STEP_INTAKE
        updates["trigger_search"] = False
        return updates
    if has_location_and_budget(merged):
        updates["current_step"] = STEP_SEARCH
        updates["trigger_search"] = True
        updates["analyst_finish"] = True
    return updates


def detect_exit_search_intent(text: str) -> bool:
    """User wants to stop — no more search (لا / no / enough)."""
    raw = (text or "").strip()
    lower = raw.lower()
    if not lower:
        return False
    # Standalone refusal (common voice replies)
    standalone = {
        "لا", "لأ", "no", "nope", "stop", "enough", "خلاص", "كفاية", "بس",
        "كدة تمام", "مش عايز", "مش محتاج", "مع السلامة", "bye", "goodbye",
    }
    normalized = re.sub(r"\s+", " ", lower)
    if normalized in standalone:
        return True
    if re.search(r"(?:^|\s)(?:no|stop|enough|خلاص|كفاية)(?:\s|$|[.!،])", normalized):
        if not re.search(r"\b(?:أيوه|اه|yes|ok|تمام|موافق|مناسب)\b", normalized):
            return True
    exit_phrases = (
        "مش عايز", "مش محتاج", "no more", "that's all", "done searching",
        "مش مناسب", "مش عاجبني", "كفاية كدة", "خلاص كدة",
    )
    if any(p in lower for p in exit_phrases):
        return True
    # "لا" at start without affirmative follow-up
    if re.match(r"^لا\b", normalized) and not any(
        w in normalized for w in ("أيوه", "اه", "تمام", "موافق", "مناسب", "عاجبني")
    ):
        return True
    return False


def detect_search_again_intent(text: str) -> bool:
    lower = (text or "").lower()
    return any(
        w in lower
        for w in (
            "تاني", "حاجة تانية", "غير", "دور تاني", "search again", "another",
            "something else", "لا مش", "أرخص", "أغلى", "منطقة تانية", "دور تاني",
        )
    )


def detect_property_selection_intent(text: str) -> bool:
    lower = (text or "").lower()
    return any(
        w in lower
        for w in (
            "تمام", "موافق", "مناسب", "عاجبني", "اختار", "ده", "دي", "حلو",
            "ماشي", "كويس", "yes", "good", "perfect", "this one", "هاخد",
        )
    ) and not detect_search_again_intent(text) and not detect_exit_search_intent(text)


def build_intake_message(state: GraphState) -> str:
    """Greet / collect name, budget, location — skip fields already known."""
    name = get_stored_name(state)
    location = (state.get("location") or "").strip()
    budget = int(state.get("budget") or 0)
    ack_parts: list[str] = []

    if name:
        ack_parts.append(f"أهلاً يا {name}")
    if location and budget > 0:
        ack_parts.append(
            f"تمام، هشتغل على {localize_area_phrase(location)} "
            f"بميزانية {price_to_spoken_arabic(budget)}"
        )
    elif location:
        ack_parts.append(f"تمام، المنطقة {localize_area_phrase(location)}")
    elif budget > 0:
        ack_parts.append(f"تمام، الميزانية {price_to_spoken_arabic(budget)}")

    if ack_parts and lead_keys_complete(state):
        return " ".join(ack_parts) + " يا فندم."

    if not name and not location and budget <= 0:
        return ASK_NAME_WARM
    if not name:
        return ASK_NAME_BEFORE_LISTINGS
    if not location:
        return (
            f"{ack_parts[0] + '، ' if ack_parts else ''}"
            f"تحب السكن فين؟ (التجمع، الشيخ زايد، المعادي...)"
        )
    return (
        f"{ack_parts[0] + '، ' if ack_parts else ''}"
        f"الميزانية معاك كام بالجنيه في {localize_area_phrase(location)}؟"
    )


def extract_standalone_name(user_text: str) -> str | None:
    """Short reply that is likely only a name (e.g. user answers 'عمر')."""
    text = (user_text or "").strip()
    if not text or len(text) > 40:
        return None
    if preprocess_budget_regex(text) or extract_email_regex(text):
        return None
    lower = text.lower()
    if any(h in lower for h in ("شقة", "فيلا", "ميزانية", "مليون", "budget", "apartment")):
        return None
    if re.fullmatch(r"[A-Za-z\u0600-\u06FF]+(?:\s+[A-Za-z\u0600-\u06FF]+){0,2}", text):
        return text.split()[0] if len(text.split()) == 1 else " ".join(text.split()[:2])
    return None


def capture_lead_name(
    state: GraphState, user_text: str, updates: dict[str, Any]
) -> None:
    if get_stored_name({**state, **updates}):
        return
    name = extract_name_regex(user_text) or extract_standalone_name(user_text)
    if name:
        updates["cx_name"] = name
        updates["user_name"] = name


LOCATION_REGEX_MAP: tuple[tuple[str, str], ...] = (
    ("cairo", "Cairo"),
    ("القاهرة", "Cairo"),
    ("قاهرة", "Cairo"),
    ("in cairo", "Cairo"),
    ("التجمع", "New Cairo"),
    ("تجمع", "New Cairo"),
    ("new cairo", "New Cairo"),
    ("الشيخ زايد", "Sheikh Zayed"),
    ("شيخ زايد", "Sheikh Zayed"),
    ("زايد", "Sheikh Zayed"),
    ("المعادي", "Maadi"),
    ("معادي", "Maadi"),
    ("أكتوبر", "6th October"),
    ("اكتوبر", "6th October"),
    ("الشروق", "Shorouk City"),
    ("شروق", "Shorouk City"),
    ("مدينة نصر", "Nasr City"),
)


def extract_location_regex(user_text: str) -> str | None:
    lower = (user_text or "").lower()
    for hint, canonical in LOCATION_REGEX_MAP:
        if hint in lower:
            return canonical
    return None


def analyst_apply_regex_fallback(
    state: GraphState, user_text: str, updates: dict[str, Any]
) -> dict[str, Any]:
    """429 / Gemini unavailable: regex + hard defaults so RAG can still run."""
    capture_lead_name(state, user_text, updates)
    if not (updates.get("location") or state.get("location")):
        loc = extract_location_regex(user_text)
        if loc:
            updates["location"] = loc
    if not int(updates.get("budget") or state.get("budget") or 0):
        b = preprocess_budget_regex(user_text)
        if b:
            updates["budget"] = b
    updates = analyst_apply_quota_defaults(state, updates)
    updates = persist_lead_fields(state, updates)
    merged = {**state, **updates}
    loc = (merged.get("location") or "").strip()
    if loc and not (merged.get("search_query") or "").strip():
        updates["search_query"] = sanitize_search_query(f"apartment {loc} egypt", loc)
    if has_location_and_budget({**state, **updates}):
        updates["analyst_finish"] = True
        print(f"[Analyst] {ANALYST_FINISH_FLAG} — 429 fallback (Cairo/2M if needed)")
    updates = sync_funnel_state(state, updates)
    return updates


def format_budget_display_english(budget: int) -> str:
    b = normalize_budget(budget)
    if b >= 1_000_000:
        millions = b / 1_000_000
        if millions == int(millions):
            return f"{int(millions):,} million EGP"
        return f"{millions:,.1f} million EGP"
    return f"{b:,} EGP"


def is_quota_error(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.ClientError):
        code = getattr(exc, "code", None)
        if code == 429:
            return True
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text or "quota" in text


def extract_email_regex(text: str) -> str | None:
    match = EMAIL_RE.search(text or "")
    return match.group(0).lower() if match else None


def extract_email_spoken(text: str) -> str | None:
    """Voice-style email: amr at gmail dot com"""
    lower = (text or "").lower()
    m = re.search(
        r"([a-z0-9._%+-]{2,32})\s*(?:at|@|ات)\s*([a-z0-9-]{2,32})\s*(?:dot|\.|دوت)\s*([a-z]{2,12})",
        lower,
    )
    if m:
        return f"{m.group(1)}@{m.group(2)}.{m.group(3)}".lower()
    m2 = re.search(
        r"([a-z0-9._%+-]+)\s*@\s*([a-z0-9.-]+)\s*(?:dot|\.)\s*([a-z]{2,12})",
        lower,
    )
    if m2:
        return f"{m2.group(1)}@{m2.group(2)}.{m2.group(3)}".lower()
    return None


def extract_email_from_text(text: str) -> str | None:
    return extract_email_regex(text) or extract_email_spoken(text)


def pick_email_prompt(state: GraphState) -> str:
    name = get_stored_name(state) or "فندم"
    return random.choice(EMAIL_PROMPT_VARIANTS).format(name=name)


def pick_email_ack(state: GraphState) -> str:
    name = get_stored_name(state) or "فندم"
    return random.choice(EMAIL_ACK_VARIANTS).format(name=name)


def extract_name_regex(text: str) -> str | None:
    skip = {"في", "عايز", "عاوز", "look", "house", "for", "and", "the", "in", "a"}
    for pattern in NAME_PATTERNS:
        match = pattern.search(text or "")
        if match:
            name = match.group(1).strip()
            if len(name) >= 2 and name.lower() not in skip:
                return name.capitalize() if name.isascii() else name
    return None


def user_message_looks_like_property_request(text: str, state: GraphState) -> bool:
    lower = (text or "").lower()
    if preprocess_budget_regex(text):
        return True
    if any(h in lower for h in PROPERTY_REQUEST_HINTS):
        return True
    if (state.get("location") or "").strip():
        return True
    if int(state.get("budget") or 0) > 0:
        return True
    return False


def preprocess_budget_regex(text: str) -> int | None:
    """Regex pre-processor: re.findall(r'\\d+'); values < 100 → × 1,000,000 EGP."""
    if not text:
        return None
    m_short = MILLION_SHORT_RE.search(text)
    if m_short:
        return int(m_short.group(1)) * BUDGET_MILLION_MULTIPLIER
    numbers = [int(d) for d in re.findall(r"\d+", text)]
    if not numbers:
        return None

    for match in re.finditer(r"\d+", text):
        n = int(match.group(0))
        window = text[max(0, match.start() - 14) : match.end() + 16]
        if BEDROOM_CONTEXT_RE.search(window):
            continue
        if 0 < n < SMALL_BUDGET_MAX:
            return n * BUDGET_MILLION_MULTIPLIER
        if n >= BUDGET_MILLION_THRESHOLD:
            return n

    for n in reversed(numbers):
        if 0 < n < SMALL_BUDGET_MAX:
            return n * BUDGET_MILLION_MULTIPLIER
    return None


def emit_safety_phrase() -> None:
    print(f"\nAgent: {API_SAFETY_PHRASE}\n")


def call_with_quota_retry(fn, /, *args, **kwargs):
    """On 429: sleep 2s, retry once; then raise GeminiQuotaExceeded."""
    last_exc: genai_errors.APIError | None = None
    for attempt in range(2):
        try:
            return fn(*args, **kwargs)
        except genai_errors.APIError as exc:
            if is_quota_error(exc):
                last_exc = exc
                if attempt == 0:
                    time.sleep(2)
                    continue
                raise GeminiQuotaExceeded(str(exc)) from exc
            raise
    if last_exc is not None:
        raise GeminiQuotaExceeded(str(last_exc)) from last_exc
    raise RuntimeError("call_with_quota_retry: unreachable")


def looks_like_goodbye(text: str) -> bool:
    lower = text.lower().strip()
    hints = (
        "bye", "goodbye", "مع السلامة", "سلام", "شكرا", "شكراً", "يسلمو", "باي",
    )
    return any(h in lower for h in hints)


def _gemini_json_once(
    prompt: str,
    *,
    system_instruction: str | None = None,
) -> dict[str, Any]:
    client = get_gemini()
    config: dict[str, Any] = {"response_mime_type": "application/json"}
    if system_instruction:
        config["system_instruction"] = system_instruction
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )
    raw = (response.text or "{}").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def gemini_json(
    prompt: str,
    *,
    system_instruction: str | None = None,
) -> dict[str, Any]:
    return call_with_quota_retry(
        _gemini_json_once, prompt, system_instruction=system_instruction
    )


def _gemini_text_once(prompt: str) -> str:
    client = get_gemini()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"system_instruction": CAIRO_AMMIYA_SYSTEM},
    )
    return (response.text or "").strip()


def gemini_text(prompt: str) -> str:
    return call_with_quota_retry(_gemini_text_once, prompt)


def print_demo_analyst(location: str, budget: int, search_query: str) -> None:
    print("\n" + "=" * 60)
    print("  [DEMO] ANALYST OUTPUT")
    print(f"  Location:         {location or '(pending)'}")
    budget_line = f"{budget:,} EGP" if budget > 0 else "(pending)"
    print(f"  Normalized budget: {budget_line}")
    print(f"  Search query:     {search_query}")
    print("=" * 60)


def print_demo_properties(
    properties: list[dict[str, Any]],
    *,
    location: str = "",
    budget: int = 0,
    label: str = "MATCHED PROPERTIES",
) -> None:
    print("\n" + "=" * 60)
    print(f"  [DEMO] {label}")
    if location or budget > 0:
        budget_line = f"{budget:,} EGP" if budget > 0 else "—"
        print(f"  Filters → location: {location or '—'} | budget: {budget_line}")
    print("-" * 60)
    if not properties:
        print("  (no listings — budget/location may not match inventory)")
    else:
        for idx, prop in enumerate(properties, start=1):
            price_egp = int(
                prop.get("price_egp") or parse_price_egp(str(prop.get("price", "")))
            )
            listed = prop.get("price", "N/A")
            print(f"  #{idx}  {prop.get('title', 'N/A')}")
            print(f"       Type:     {prop.get('type', 'N/A')}")
            print(f"       Area:     {prop.get('location', 'N/A')}")
            print(f"       Price:    {price_egp:,} EGP  (catalog: {listed})")
            print(f"       Beds:     {prop.get('bedroom', '—')} | "
                  f"Size: {prop.get('size_sqm', '—')} sqm")
            if budget > 0:
                within = "YES" if price_egp <= budget else "OVER BUDGET"
                print(f"       vs budget: {within}")
    print("=" * 60 + "\n")


def ensure_price_tags(text: str) -> str:
    return re.sub(
        r"(?<!<price>)(\d{1,3}(?:,\d{3})+|\d+)(?!</price>)",
        lambda m: f"<price>{m.group(1).replace(',', '')}</price>",
        text,
    )


_AR_ONES = (
    "", "واحد", "اتنين", "تلاتة", "أربعة", "خمسة", "ستة", "سبعة", "تمانية", "تسعة",
    "عشرة", "حداشر", "اتناشر", "تلتاشر", "أربعتاشر", "خمستاشر", "ستاشر",
    "سبعتاشر", "تمنتاشر", "تسعتاشر",
)
_AR_TENS = (
    "", "", "عشرين", "تلاتين", "أربعين", "خمسين", "ستين", "سبعين", "تمانين", "تسعين",
)


def _cardinal_arabic(n: int) -> str:
    if n <= 0:
        return ""
    if n < len(_AR_ONES):
        return _AR_ONES[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        if unit == 0:
            return _AR_TENS[tens]
        return f"{_AR_ONES[unit]} و{_AR_TENS[tens]}"
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        head = "مية" if hundreds == 1 else f"{_AR_ONES[hundreds]} مية"
        if rem == 0:
            return head
        return f"{head} و{_cardinal_arabic(rem)}"
    return str(n)


def price_to_spoken_arabic(egp: int) -> str:
    """Convert EGP amount to spoken Egyptian Arabic (no digits)."""
    if egp <= 0:
        return "مش محدد"
    millions = egp // 1_000_000
    remainder = egp % 1_000_000
    thousands = remainder // 1000
    parts: list[str] = []
    if millions:
        parts.append(f"{_cardinal_arabic(millions)} مليون")
    if thousands:
        parts.append(f"{_cardinal_arabic(thousands)} ألف")
    elif remainder:
        parts.append(_cardinal_arabic(remainder))
    return " و".join(parts) + " جنيه"


def strip_voice_tags(text: str) -> str:
    """Remove XML/markup so TTS can read plain Arabic."""
    cleaned = re.sub(r"<[^>]+>", "", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def localize_property_type(type_en: str) -> str:
    mapping = {
        "duplex": "دوبلكس",
        "villa": "فيلا",
        "apartment": "شقة",
        "townhouse": "تاون هاوس",
        "penthouse": "بنتهاوس",
        "chalet": "شاليه",
        "twin house": "توين هاوس",
        "studio": "استوديو",
    }
    key = (type_en or "").strip().lower()
    return mapping.get(key, type_en or "وحدة سكنية")


def localize_finish_hint(title_en: str) -> str:
    title = (title_en or "").lower()
    if "fully finished" in title or "super lux" in title:
        return "تشطيب سوبر لوكس"
    if "finished" in title or "furnished" in title:
        return "تشطيب كامل"
    if "semi finished" in title:
        return "نصف تشطيب"
    if "core" in title or "shell" in title:
        return "مفرغ"
    if "ready" in title or "immediate" in title:
        return "استلام فوري"
    return ""


def localize_area_phrase(location_en: str) -> str:
    loc = (location_en or "").lower()
    if "shorouk" in loc or "shorok" in loc:
        return "مدينة الشروق"
    if "new cairo" in loc or "tagammu" in loc or "تجمع" in loc:
        return "التجمع الخامس"
    if "zayed" in loc or "زايد" in loc:
        return "الشيخ زايد"
    if "october" in loc or "أكتوبر" in loc:
        return "6 أكتوبر"
    if "maadi" in loc or "معادي" in loc:
        return "المعادي"
    if "north coast" in loc:
        return "الساحل الشمالي"
    if "compound" in loc or "park view" in loc or "mountain view" in loc:
        return "كومباوند مميز"
    parts = [p.strip() for p in re.split(r"[,|]", location_en or "") if p.strip()]
    return parts[0] if parts else "منطقة مميزة"


def listing_price_egp(prop: dict[str, Any]) -> int:
    return int(prop.get("price_egp") or parse_price_egp(str(prop.get("price", ""))))


def listing_voice_catalog(properties: list[dict[str, Any]], limit: int = 2) -> str:
    rows: list[dict[str, Any]] = []
    for prop in properties[:limit]:
        title = str(prop.get("title", ""))
        rows.append(
            {
                "type_en": prop.get("type"),
                "type_ar": localize_property_type(str(prop.get("type", ""))),
                "location_en": prop.get("location"),
                "area_ar": localize_area_phrase(str(prop.get("location", ""))),
                "title_en": title,
                "finish_ar": localize_finish_hint(title),
                "bedrooms": prop.get("bedroom"),
                "size_sqm": prop.get("size_sqm"),
                "price_spoken_ar": price_to_spoken_arabic(listing_price_egp(prop)),
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2)


def pivot_lead_for_state(
    state: GraphState, properties: list[dict[str, Any]]
) -> str:
    """Acknowledge rejection / pivot when re-pitching after negotiation or area change."""
    soft = state.get("soft_search_note") or ""
    negotiation_count = int(state.get("negotiation_count") or 0)
    if negotiation_count > 0 or soft in ("alternate_area", "alternate_type"):
        if properties:
            area = localize_area_phrase(str(properties[0].get("location", "")))
            return (
                f"فهمتك، طيب الميزانية دي ممكن تجيب لنا حاجة لقطة في {area}، إيه رأيك؟"
            )
        return PIVOT_REJECTED_OFFER
    return ""


def build_broker_voice_prompt(
    *,
    catalog: str,
    location: str,
    budget: int,
    cx_name: str | None = None,
    pivot_lead: str = "",
    soft_lead: str = "",
    extra_instructions: str = "",
) -> str:
    budget_spoken = price_to_spoken_arabic(budget) if budget > 0 else "مش محددة"
    lead_block = ""
    if pivot_lead:
        lead_block += f'- Open with this pivot (adapt area name if needed): "{pivot_lead}"\n'
    elif soft_lead:
        lead_block += f'- Open with: "{soft_lead}"\n'
    name_rule = ""
    if cx_name:
        name_rule = f"- Address the client warmly as {cx_name} once (يا {cx_name}).\n"
    return f"""{BROKER_PERSONA}

{VOICE_OUTPUT_RULES}

Client name: {cx_name or "unknown — do not invent"}
Client target area: {location or "غير محددة"}
Client budget (spoken): {budget_spoken}

Listings (English source — describe in natural Ammiya using price_spoken_ar verbatim):
{catalog}

{name_rule}{lead_block}{extra_instructions}
Write the voice reply now (2-3 sentences only).
"""


def finalize_voice_reply(reply: str) -> str:
    return strip_voice_tags(reply)


def location_matches(property_location: str, user_location: str) -> bool:
    if not user_location:
        return True
    prop = property_location.lower()
    user = user_location.lower()
    tokens = [t.strip() for t in re.split(r"[,|\s]+", user) if len(t.strip()) > 2]
    if not tokens:
        return user in prop
    return any(token in prop for token in tokens)


def search_properties(
    search_query: str,
    location: str,
    budget: int,
    *,
    strict_location: bool = True,
    strict_budget: bool = True,
    property_type: str = "",
) -> list[dict[str, Any]]:
    collection = get_collection()
    vector = get_embedder().embed_query(search_query)
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [vector],
        "n_results": TOP_K * 2,
        "include": ["metadatas", "documents", "distances"],
    }
    result: dict[str, Any]
    if budget > 0:
        try:
            result = collection.query(
                **query_kwargs,
                where={"price_egp": {"$lte": budget}},
            )
            if not (result.get("metadatas") or [[]])[0]:
                result = collection.query(**query_kwargs)
        except Exception as exc:
            print(f"[warn] Chroma $lte budget filter failed, scanning without: {exc}")
            result = collection.query(**query_kwargs)
    else:
        result = collection.query(**query_kwargs)

    metadatas = (result.get("metadatas") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    matches: list[dict[str, Any]] = []
    for meta, doc, dist in zip(metadatas, documents, distances):
        if not meta:
            continue
        price = parse_price_egp(str(meta.get("price", "")))
        if strict_budget and budget > 0 and price > budget:
            continue
        if strict_location and location and not location_matches(
            str(meta.get("location", "")), location
        ):
            continue
        if property_type and not property_type_matches(str(meta.get("type", "")), property_type):
            continue
        matches.append({**meta, "price_egp": price, "document": doc, "distance": dist})

    if not matches and metadatas and strict_budget:
        for meta, doc, dist in zip(metadatas, documents, distances):
            if not meta:
                continue
            price = parse_price_egp(str(meta.get("price", "")))
            if strict_budget and budget > 0 and price > budget:
                continue
            if strict_location and location and not location_matches(
                str(meta.get("location", "")), location
            ):
                continue
            if property_type and not property_type_matches(str(meta.get("type", "")), property_type):
                continue
            matches.append({**meta, "price_egp": price, "document": doc, "distance": dist})

    if strict_budget:
        matches.sort(key=lambda x: x.get("distance", 999))
    else:
        matches.sort(key=lambda x: (x.get("price_egp", 0), x.get("distance", 999)))

    return matches[:MAX_RESULTS]


def evaluate_cheaper_found(
    properties: list[dict[str, Any]],
    negotiation_mode: str,
    reference_price: int,
) -> bool:
    if negotiation_mode != "cheaper" or not properties:
        return False
    if reference_price > 0:
        return any(int(p.get("price_egp") or 0) < reference_price for p in properties)
    return True


def sort_properties_for_mode(
    properties: list[dict[str, Any]], negotiation_mode: str
) -> list[dict[str, Any]]:
    if negotiation_mode == "cheaper":
        return sorted(properties, key=lambda x: x.get("price_egp", 0))
    return properties


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def greeting_node(state: GraphState) -> dict[str, Any]:
    if state.get("is_greeted"):
        return {}
    payload: dict[str, Any] = {
        "is_greeted": True,
        "conversation_stage": STAGE_INTAKE,
        "current_step": STEP_INTAKE,
        "trigger_search": False,
    }
    msgs = state.get("messages") or []
    user_already_spoke = any(isinstance(m, HumanMessage) for m in msgs)
    if user_already_spoke:
        return payload
    payload["messages"] = [AIMessage(content=ASK_NAME_WARM)]
    return payload


def analyst_node(state: GraphState) -> dict[str, Any]:
    """Extract lead fields; skip Gemini when name+location+budget are complete (FINISH)."""
    user_text = last_human_text(state)
    if not user_text:
        return {}

    current_location = (state.get("location") or "").strip()
    current_budget = int(state.get("budget") or 0)
    current_name = get_stored_name(state)
    is_goodbye = looks_like_goodbye(user_text)
    wants_cheaper = wants_cheaper_price(user_text)

    if state.get("exit_search_loop"):
        return {
            "current_step": STEP_EXIT,
            "trigger_search": False,
            "spoken_reply": POLITE_EXIT_MSG,
            "messages": [AIMessage(content=POLITE_EXIT_MSG)],
        }

    updates: dict[str, Any] = {
        "is_goodbye": is_goodbye,
        "negotiation_mode": "",
        "cheaper_found": False,
        "analyst_finish": False,
    }
    if detect_exit_search_intent(user_text):
        updates["exit_search_loop"] = True
        updates["trigger_search"] = False
        updates["current_step"] = STEP_EXIT
        updates["spoken_reply"] = POLITE_EXIT_MSG
        updates["messages"] = [AIMessage(content=POLITE_EXIT_MSG)]
        return updates

    regex_budget = preprocess_budget_regex(user_text)
    if regex_budget:
        updates["budget"] = regex_budget

    current_email = state.get("cx_email")
    capture_lead_name(state, user_text, updates)
    if not (updates.get("location") or current_location):
        loc_rx = extract_location_regex(user_text)
        if loc_rx:
            updates["location"] = loc_rx
    regex_email = extract_email_from_text(user_text)
    if regex_email:
        updates["cx_email"] = regex_email
        if state.get("property_confirmed"):
            updates["conversation_stage"] = STAGE_POST_EMAIL
            updates["current_step"] = STEP_POST_EMAIL

    updates = persist_lead_fields(state, updates)
    merged = {**state, **updates}

    if (
        has_location_and_budget(merged)
        and not wants_cheaper
        and not merged.get("exit_search_loop")
    ):
        loc = (merged.get("location") or "").strip()
        if not (merged.get("search_query") or "").strip():
            updates["search_query"] = sanitize_search_query(
                f"apartment {loc} egypt", loc
            )
        updates["analyst_finish"] = True
        updates = persist_lead_fields(state, updates)
        updates = sync_funnel_state(state, updates)
        print(
            f"[Analyst] {ANALYST_FINISH_FLAG} — skip Gemini "
            f"(location={loc!r}, budget={int(merged.get('budget') or 0):,}, "
            f"step={updates.get('current_step')}, trigger_search={updates.get('trigger_search')})"
        )
        return updates

    user_prompt = f"""Stored user_name / cx_name: "{get_stored_name(merged) or "none"}"
Stored location: "{current_location or "none"}"
Stored budget (EGP): {current_budget or "none"}
Stored cx_email: "{current_email or "none"}"
User message: \"\"\"{user_text}\"\"\"
If user_name is already stored, do NOT ask for it again — only extract missing fields."""

    data: dict[str, Any] = {}
    try:
        data = gemini_json(user_prompt, system_instruction=ANALYST_SYSTEM)
    except genai_errors.APIError as exc:
        if is_quota_error(exc):
            print("[warn] Analyst 429/quota — regex + Cairo/2M defaults")
            return analyst_apply_regex_fallback(state, user_text, updates)
        print(f"[warn] Analyst API failed: {exc}")
        data = {}
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"[warn] Analyst parse failed: {exc}")
        data = {}

    api_name = data.get("cx_name")
    if (
        api_name
        and str(api_name).strip().lower() not in ("null", "none", "")
        and not get_stored_name(merged)
    ):
        updates["cx_name"] = str(api_name).strip()
        updates["user_name"] = updates["cx_name"]
    elif get_stored_name(merged):
        updates["cx_name"] = get_stored_name(merged)
        updates["user_name"] = get_stored_name(merged)
    if data.get("analyst_finish") is True:
        updates["analyst_finish"] = True
    apply_analyst_funnel_json(data, updates)
    api_email = data.get("cx_email")
    if api_email and str(api_email).strip().lower() not in ("null", "none", ""):
        updates["cx_email"] = str(api_email).strip().lower()
        if state.get("property_confirmed") or updates.get("property_confirmed"):
            updates["lead_captured"] = True

    resolved_location = current_location
    api_location = data.get("location")
    if api_location and str(api_location).strip().lower() not in ("null", "none", ""):
        candidate = str(api_location).strip()
        if not current_location:
            resolved_location = candidate
            updates["location"] = candidate
        elif not wants_cheaper:
            resolved_location = candidate
            updates["location"] = candidate
        elif locations_differ(current_location, candidate):
            resolved_location = candidate
            updates["location"] = candidate
    elif wants_cheaper and current_location:
        updates["location"] = current_location

    raw_query = str(
        data.get("search_query") or f"apartment {resolved_location or 'cairo'} egypt"
    ).strip()
    updates["search_query"] = sanitize_search_query(raw_query, resolved_location)

    if locations_differ(current_location, resolved_location):
        updates["retrieved_properties"] = []
        updates["negotiation_count"] = 0

    raw_budget = None
    if data:
        raw_budget = data.get("normalized_budget", data.get("budget"))
    api_budget = parse_budget_value(raw_budget) if raw_budget is not None else None
    new_budget: int | None = api_budget if api_budget else regex_budget

    if wants_cheaper and current_budget > 0 and not api_budget and not regex_budget:
        updates["previous_budget"] = current_budget
        new_budget = apply_relative_budget_cut(current_budget)
    elif wants_cheaper and current_budget > 0 and new_budget and new_budget < current_budget:
        updates["previous_budget"] = current_budget

    if new_budget is not None and new_budget > 0:
        updates["budget"] = normalize_budget(new_budget)
    elif current_budget > 0:
        normalized = normalize_budget(current_budget)
        if normalized != current_budget:
            updates["budget"] = normalized

    if wants_cheaper:
        ref = reference_price_from_state(state)
        if ref > 0:
            updates["reference_price_egp"] = ref
        updates["negotiation_mode"] = "cheaper"
        loc = updates.get("location") or current_location
        cap = updates.get("budget", current_budget)
        updates["search_query"] = sanitize_search_query(
            f"affordable apartment property {loc} under {cap} egypt", loc
        )

    updates = persist_lead_fields(state, updates)
    merged_after = {**state, **updates}
    if lead_keys_complete(merged_after) and not wants_cheaper:
        updates["analyst_finish"] = True
        print(f"[Analyst] {ANALYST_FINISH_FLAG} — lead keys complete after extraction")

    updates = sync_funnel_state(state, updates)
    print(
        f"[Analyst] step={updates.get('current_step')!r} "
        f"trigger_search={updates.get('trigger_search')}"
    )
    return updates


def intake_node(state: GraphState) -> dict[str, Any]:
    """GREETING & INTAKE — collect name, budget, location; acknowledge known fields."""
    name = get_stored_name(state)
    location = (state.get("location") or "").strip()
    budget = int(state.get("budget") or 0)
    content = build_intake_message(state)
    payload: dict[str, Any] = {
        "messages": [AIMessage(content=content)],
        "missing_info": not lead_keys_complete(state),
        "current_step": STEP_INTAKE,
        "trigger_search": False,
    }
    if name:
        payload["cx_name"] = name
        payload["user_name"] = name
    if location:
        payload["location"] = location
    if budget > 0:
        payload["budget"] = budget
    return payload


def ask_for_name_node(state: GraphState) -> dict[str, Any]:
    """Legacy alias — intake_node handles name collection."""
    return intake_node(state)


def ask_for_email_node(state: GraphState) -> dict[str, Any]:
    """LEAD CAPTURE — natural Egyptian prompt; accept spoken or written email."""
    user_text = last_human_text(state)
    found = extract_email_from_text(user_text) or (state.get("cx_email") or "").strip()
    if found:
        ack = pick_email_ack({**state, "cx_name": get_stored_name(state)})
        follow = POST_EMAIL_MORE_QUESTIONS_MSG
        reply = f"{ack} {follow}"
        return {
            "cx_email": found.lower() if "@" in found else found,
            "current_step": STEP_POST_EMAIL,
            "conversation_stage": STAGE_POST_EMAIL,
            "trigger_search": False,
            "spoken_reply": reply,
            "messages": [AIMessage(content=reply)],
        }
    content = pick_email_prompt(state)
    return {
        "current_step": STEP_LEAD_CAPTURE,
        "trigger_search": False,
        "spoken_reply": content,
        "messages": [AIMessage(content=content)],
    }


def post_email_intent_node(state: GraphState) -> dict[str, Any]:
    """After email: offer another search or close with 1–3 business days."""
    user_text = last_human_text(state)
    lower = (user_text or "").lower()
    wants_more = any(
        w in lower
        for w in (
            "أيوه", "اه", "ايوه", "yes", "more", "تاني", "كمان", "غير", "ميزانية",
            "منطقة", "another", "different", "search",
        )
    )
    done = any(
        w in lower
        for w in ("لا", "لأ", "no", "خلاص", "كفاية", "مش", "thanks", "thank", "كدة تمام")
    )
    if wants_more and not done:
        return {
            "property_confirmed": False,
            "retrieved_properties": [],
            "conversation_stage": STAGE_INTAKE,
            "current_step": STEP_INTAKE,
            "trigger_search": False,
            "exit_search_loop": False,
            "user_intent": "search",
            "messages": [
                AIMessage(
                    content="تمام يا فندم، قولّي المنطقة والميزانية الجديدة اللي تحب ندور عليها."
                )
            ],
            "spoken_reply": "تمام يا فندم، قولّي المنطقة والميزانية الجديدة اللي تحب ندور عليها.",
        }
    name = get_stored_name(state) or "فندم"
    msg = FINAL_SIGNOFF_MSG.format(name=name)
    return {
        "conversation_stage": STAGE_CLOSED,
        "current_step": STEP_CLOSING,
        "lead_captured": True,
        "trigger_search": False,
        "spoken_reply": msg,
        "messages": [AIMessage(content=msg)],
    }


def intent_node(state: GraphState) -> dict[str, Any]:
    """DECISION GATE after recommendations — search again / confirm / exit."""
    stage = state.get("conversation_stage") or STAGE_INTAKE
    user_text = last_human_text(state)

    if stage == STAGE_CLOSED or state.get("exit_search_loop"):
        return {"user_intent": "noop", "current_step": STEP_EXIT}

    if state.get("is_goodbye") or detect_exit_search_intent(user_text):
        return {
            "user_intent": "done",
            "exit_search_loop": True,
            "trigger_search": False,
            "current_step": STEP_EXIT,
            "spoken_reply": POLITE_EXIT_MSG,
            "messages": [AIMessage(content=POLITE_EXIT_MSG)],
        }

    if stage == STAGE_INTAKE:
        if not lead_keys_complete(state):
            return {"user_intent": "collect", "current_step": STEP_INTAKE}
        return {"user_intent": "search", "trigger_search": True, "current_step": STEP_SEARCH}

    if stage == STAGE_NEGOTIATING:
        return _classify_negotiation_intent(state, user_text)

    if stage == STAGE_PRE_CLOSURE:
        return _classify_pre_closure_intent(user_text)

    return {"user_intent": "search"}


def _classify_negotiation_intent(state: GraphState, user_text: str) -> dict[str, Any]:
    """THE DECISION GATE: search again, confirm property, or polite exit."""
    if state.get("exit_search_loop"):
        return {"user_intent": "done", "current_step": STEP_EXIT}

    if detect_exit_search_intent(user_text):
        return {
            "user_intent": "done",
            "exit_search_loop": True,
            "trigger_search": False,
            "current_step": STEP_EXIT,
            "spoken_reply": POLITE_EXIT_MSG,
            "messages": [AIMessage(content=POLITE_EXIT_MSG)],
        }

    if detect_search_again_intent(user_text):
        return {
            "user_intent": "search",
            "trigger_search": True,
            "current_step": STEP_SEARCH,
            "exit_search_loop": False,
        }

    if detect_property_selection_intent(user_text):
        title = state.get("best_match_title") or "N/A"
        confirm = PROPERTY_CONFIRM_MSG
        return {
            "user_intent": "positive",
            "property_confirmed": True,
            "agreed_property_title": title,
            "user_accepted_recommendation": True,
            "current_step": STEP_LEAD_CAPTURE,
            "trigger_search": False,
            "conversation_stage": STAGE_PRE_CLOSURE,
            "spoken_reply": confirm,
            "messages": [AIMessage(content=confirm)],
        }

    prompt = f"""{SALESMAN_PERSONA}

The agent just showed property recommendations and asked:
"{RECOMMENDATION_DECISION_MSG}"

User reply: \"\"\"{user_text}\"\"\"

Classify intent. Return JSON:
{{
  "is_satisfied": true if user says yes / matches what I want / تمام / مناسب / موافق,
  "wants_changes": true if user wants different price, size, area, or another option,
  "reason": "brief English note"
}}

If user wants to adjust budget, size, or location, wants_changes must be true.
If clearly happy with the listing, is_satisfied must be true.
"""

    try:
        data = gemini_json(prompt)
    except (genai_errors.APIError, json.JSONDecodeError, ValueError) as exc:
        print(f"[warn] Suitability check failed: {exc}")
        if detect_exit_search_intent(user_text):
            return {
                "user_intent": "done",
                "exit_search_loop": True,
                "trigger_search": False,
                "current_step": STEP_EXIT,
                "spoken_reply": POLITE_EXIT_MSG,
                "messages": [AIMessage(content=POLITE_EXIT_MSG)],
            }
        lower = user_text.lower()
        positive = any(
            w in lower
            for w in (
                "ok", "fine", "good", "great", "yes", "perfect", "sounds good",
                "أيوه", "اه", "تمام", "مناسب", "موافق", "كويس", "حلو", "ماشي",
            )
        )
        changes = any(
            w in lower
            for w in ("أكبر", "أصغر", "أرخص", "أغلى", "تاني", "غير", "عدّل", "عدل", "حاجة تانية")
        )
        if positive and not changes:
            title = state.get("best_match_title") or "N/A"
            confirm = PROPERTY_CONFIRM_MSG
            return {
                "user_intent": "positive",
                "property_confirmed": True,
                "agreed_property_title": title,
                "current_step": STEP_LEAD_CAPTURE,
                "trigger_search": False,
                "conversation_stage": STAGE_PRE_CLOSURE,
                "spoken_reply": confirm,
                "messages": [AIMessage(content=confirm)],
            }
        if changes or detect_search_again_intent(user_text):
            return {
                "user_intent": "search",
                "trigger_search": True,
                "current_step": STEP_SEARCH,
            }
        if detect_exit_search_intent(user_text):
            return {
                "user_intent": "done",
                "exit_search_loop": True,
                "trigger_search": False,
                "current_step": STEP_EXIT,
                "spoken_reply": POLITE_EXIT_MSG,
                "messages": [AIMessage(content=POLITE_EXIT_MSG)],
            }
        return {"user_intent": "search", "trigger_search": True, "current_step": STEP_SEARCH}

    if data.get("wants_changes") and detect_exit_search_intent(user_text):
        return {
            "user_intent": "done",
            "exit_search_loop": True,
            "trigger_search": False,
            "current_step": STEP_EXIT,
            "spoken_reply": POLITE_EXIT_MSG,
            "messages": [AIMessage(content=POLITE_EXIT_MSG)],
        }

    if data.get("is_satisfied") and not data.get("wants_changes"):
        title = state.get("best_match_title") or "N/A"
        confirm = PROPERTY_CONFIRM_MSG
        return {
            "user_intent": "positive",
            "property_confirmed": True,
            "agreed_property_title": title,
            "user_intent_level": "High",
            "user_accepted_recommendation": True,
            "current_step": STEP_LEAD_CAPTURE,
            "trigger_search": False,
            "conversation_stage": STAGE_PRE_CLOSURE,
            "spoken_reply": confirm,
            "messages": [AIMessage(content=confirm)],
        }

    if data.get("wants_changes"):
        count = int(state.get("negotiation_count") or 0) + 1
        return {
            "user_intent": "search",
            "trigger_search": True,
            "current_step": STEP_SEARCH,
            "negotiation_count": count,
            "exit_search_loop": False,
        }

    return {
        "user_intent": "search",
        "trigger_search": True,
        "current_step": STEP_SEARCH,
    }


def _classify_pre_closure_intent(user_text: str) -> dict[str, Any]:
    prompt = f"""The agent asked: "{PRE_CLOSURE_MSG}"

User: \"\"\"{user_text}\"\"\"

Return JSON:
{{
  "wants_more": true if user wants other recommendations,
  "is_done": true if user says no / that's all / كدة تمام / خلاص
}}
"""

    try:
        data = gemini_json(prompt)
    except (genai_errors.APIError, json.JSONDecodeError, ValueError):
        lower = user_text.lower()
        done = any(w in lower for w in ("لا", "كدة تمام", "خلاص", "كفاية", "no", "that's all"))
        more = any(w in lower for w in ("أيوه", "اه", "كمان", "تاني", "yes", "more"))
        if done and not more:
            return {"user_intent": "done"}
        if more:
            return {"user_intent": "wants_more", "conversation_stage": STAGE_INTAKE}
        return {"user_intent": "done"}

    if data.get("is_done") and not data.get("wants_more"):
        return {"user_intent": "done", "user_intent_level": "High"}

    if data.get("wants_more"):
        return {"user_intent": "wants_more", "conversation_stage": STAGE_INTAKE}

    return {"user_intent": "done", "user_intent_level": "High"}


def search_node(state: GraphState) -> dict[str, Any]:
    """THE SEARCH — filler phrase, then Chroma RAG when trigger_search is True."""
    if state.get("exit_search_loop"):
        return {
            "trigger_search": False,
            "current_step": STEP_EXIT,
            "spoken_reply": POLITE_EXIT_MSG,
            "messages": [AIMessage(content=POLITE_EXIT_MSG)],
        }
    if not state.get("trigger_search"):
        return {}

    user_text = last_human_text(state)
    updates: dict[str, Any] = {
        "current_step": STEP_SEARCH,
        "messages": [AIMessage(content=pick_search_filler(state))],
    }
    regex_budget = preprocess_budget_regex(user_text)
    if regex_budget:
        updates["budget"] = regex_budget

    name = get_stored_name({**state, **updates})
    location = (state.get("location") or updates.get("location") or "").strip()
    budget = int(updates.get("budget") or state.get("budget") or 0)

    if not lead_keys_complete({**state, "cx_name": name, "location": location, "budget": budget}):
        return intake_node({**state, **updates})

    budget = normalize_budget(budget)
    search_query = state.get("search_query") or f"{location} property under {budget} EGP"
    property_type = (state.get("property_type") or "").strip()
    negotiation_mode = state.get("negotiation_mode") or ""
    reference_price = reference_price_from_state(state)
    soft_note = ""

    try:
        properties = chroma_search(
            search_query, location, budget, property_type=property_type
        )
        if not properties and property_type:
            properties = chroma_search(
                search_query, location, budget, property_type=""
            )
            if properties:
                soft_note = "alternate_type"
        properties = sort_properties_for_mode(properties, negotiation_mode)
    except Exception as exc:
        print(f"[warn] Chroma search failed: {exc}")
        properties = []

    cheaper_found = evaluate_cheaper_found(properties, negotiation_mode, reference_price)
    best_title = properties[0].get("title", "N/A") if properties else "N/A"

    print_demo_properties(
        properties,
        location=location,
        budget=budget,
        label="CHROMADB SEARCH RESULTS",
    )

    payload: dict[str, Any] = {
        "missing_info": False,
        "retrieved_properties": properties,
        "best_match_title": best_title,
        "cheaper_found": cheaper_found,
        "soft_search_note": soft_note,
        "current_step": STEP_RECOMMENDATION,
        "trigger_search": False,
        **updates,
    }
    return persist_lead_fields(state, payload)


def soft_search_node(state: GraphState) -> dict[str, Any]:
    """
    Zero-result fallback: broaden property type, then search nearby cheaper areas.
    """
    location = (state.get("location") or "").strip()
    budget = normalize_budget(int(state.get("budget") or 0))
    negotiation_mode = state.get("negotiation_mode") or ""
    reference_price = reference_price_from_state(state)

    fallback_strategies: list[tuple[str, str, bool, bool]] = [
        (
            "alternate_type",
            f"apartment flat condo townhouse duplex villa property {location} under {budget} EGP",
            True,
            True,
        ),
        (
            "alternate_type",
            f"residential home unit {location} budget friendly affordable",
            True,
            True,
        ),
        (
            "alternate_area",
            f"affordable cheaper residential property Cairo Egypt under {budget} EGP",
            False,
            True,
        ),
        (
            "alternate_area",
            f"lower price bargain property Egypt under {budget} EGP best value",
            False,
            False,
        ),
    ]

    properties: list[dict[str, Any]] = []
    note = ""
    for note_key, query, strict_loc, strict_budget in fallback_strategies:
        try:
            hits = search_properties(
                query,
                location if strict_loc else "",
                budget,
                strict_location=strict_loc,
                strict_budget=strict_budget,
            )
            hits = sort_properties_for_mode(hits, negotiation_mode)
        except Exception as exc:
            print(f"[warn] Soft search failed ({note_key}): {exc}")
            hits = []
        if hits:
            properties = hits
            note = note_key
            break

    cheaper_found = evaluate_cheaper_found(properties, negotiation_mode, reference_price)
    best_title = properties[0].get("title", "N/A") if properties else "N/A"

    print_demo_properties(
        properties,
        location=location,
        budget=budget,
        label=f"FALLBACK SEARCH ({note or 'broadened'})",
    )

    return persist_lead_fields(
        state,
        {
            "retrieved_properties": properties,
            "best_match_title": best_title,
            "cheaper_found": cheaper_found,
            "soft_search_note": note,
        },
    )


def _voice_fallback_pitch(prop: dict[str, Any], *, pivot: str = "") -> str:
    """Template pitch: property title + spoken price (no filler)."""
    title = str(prop.get("title", "") or "").strip()
    area = localize_area_phrase(str(prop.get("location", "")))
    ptype = localize_property_type(str(prop.get("type", "")))
    finish = localize_finish_hint(title)
    price = price_to_spoken_arabic(listing_price_egp(prop))
    finish_bit = f"، {finish}" if finish else ""
    title_lead = f"{title}، " if title else ""
    body = (
        f"عندنا {title_lead}{ptype} في {area}{finish_bit} — "
        f"{prop.get('bedroom', '—')} أوض نوم. السعر {price}، "
        f"تحب أقولك تفاصيل أكتر؟"
    )
    return f"{pivot} {body}".strip() if pivot else body


def final_response_node(state: GraphState) -> dict[str, Any]:
    properties = state.get("retrieved_properties") or []
    location = state.get("location") or ""
    budget = int(state.get("budget") or 0)
    negotiation_mode = state.get("negotiation_mode") or ""
    cheaper_found = bool(state.get("cheaper_found"))
    soft_note = state.get("soft_search_note") or ""
    pivot = pivot_lead_for_state(state, properties)

    cx_name = state.get("cx_name")

    def finish(reply: str, *, presented: bool = False, **extra: Any) -> dict[str, Any]:
        voice_text = finalize_voice_reply(reply)
        payload: dict[str, Any] = {
            "negotiation_mode": "",
            "soft_search_note": "",
            "spoken_reply": voice_text,
            **extra,
            "messages": [AIMessage(content=voice_text)],
        }
        if presented:
            payload["property_presented"] = True
        return payload

    # Cheaper-negotiation branch
    if negotiation_mode == "cheaper":
        if not properties or not cheaper_found:
            return finish(NEGOTIATOR_NOT_FOUND_MSG)

        top = properties[0]
        catalog = listing_voice_catalog([top], limit=1)
        prompt = build_broker_voice_prompt(
            catalog=catalog,
            location=location,
            budget=budget,
            cx_name=cx_name,
            pivot_lead=NEGOTIATOR_FOUND_LEAD,
            extra_instructions="- Emphasize that this option is better value within their budget.\n",
        )
        try:
            reply = gemini_text(prompt)
        except genai_errors.APIError as exc:
            reply = (
                QUOTA_SALES_PITCH_MSG
                if is_quota_error(exc)
                else _voice_fallback_pitch(top, pivot=NEGOTIATOR_FOUND_LEAD)
            )
        return finish(reply, presented=True)

    # Zero results — alternate area pitch
    if not properties:
        alt_hits: list[dict[str, Any]] = []
        try:
            alt_hits = search_properties(
                f"affordable residential property Egypt under {budget} EGP best value",
                "",
                budget,
                strict_location=False,
                strict_budget=True,
            )
        except Exception as exc:
            print(f"[warn] Alternate-area lookup failed: {exc}")

        if alt_hits:
            top = alt_hits[0]
            area = localize_area_phrase(str(top.get("location", "")))
            pivot_alt = (
                f"فهمتك، طيب الميزانية دي ممكن تجيب لنا حاجة لقطة في {area}، إيه رأيك؟"
            )
            catalog = listing_voice_catalog(alt_hits[:2])
            prompt = build_broker_voice_prompt(
                catalog=catalog,
                location=location,
                budget=budget,
                cx_name=cx_name,
                pivot_lead=pivot_alt,
                extra_instructions=(
                    "- No exact match in the client's requested area; pivot warmly to the listing area.\n"
                ),
            )
            try:
                reply = gemini_text(prompt)
            except genai_errors.APIError as exc:
                reply = (
                    QUOTA_SALES_PITCH_MSG
                    if is_quota_error(exc)
                    else _voice_fallback_pitch(top, pivot=pivot_alt)
                )
        else:
            budget_spoken = price_to_spoken_arabic(budget)
            reply = (
                f"والله مفيش حاجة في {localize_area_phrase(location)} ضمن "
                f"{budget_spoken} دلوقتي يا فندم. ممكن نوسّع المنطقة أو نظبط الميزانية؟"
            )
        return finish(reply, presented=bool(alt_hits))

    # Main sales pitch (top 1–2 listings)
    soft_lead = ""
    if soft_note == "alternate_type":
        soft_lead = SOFT_TYPE_LEAD
    elif soft_note == "alternate_area":
        soft_lead = SOFT_AREA_LEAD

    catalog = listing_voice_catalog(properties, limit=2)
    area_instruction = ""
    if soft_note == "alternate_area":
        area_instruction = (
            f"- Listings are outside {location}; pivot naturally to the compound/area in the JSON.\n"
        )

    prompt = build_broker_voice_prompt(
        catalog=catalog,
        location=location,
        budget=budget,
        cx_name=cx_name,
        pivot_lead=pivot or "",
        soft_lead=soft_lead if not pivot else "",
        extra_instructions=area_instruction,
    )

    try:
        reply = gemini_text(prompt)
    except genai_errors.APIError as exc:
        if is_quota_error(exc) and properties:
            reply = _voice_fallback_pitch(
                properties[0],
                pivot=pivot or soft_lead,
            )
        elif is_quota_error(exc):
            reply = QUOTA_SALES_PITCH_MSG
        else:
            print(f"[warn] Sales pitch failed: {exc}")
            reply = _voice_fallback_pitch(
                properties[0],
                pivot=pivot or soft_lead,
            )

    return finish(reply, presented=True)


def consultation_node(state: GraphState) -> dict[str, Any]:
    """RECOMMENDATION — ask if listings suit the client (decision gate next turn)."""
    pitch = (state.get("spoken_reply") or "").strip()
    combined = (
        f"{pitch} {RECOMMENDATION_DECISION_MSG}".strip()
        if pitch
        else RECOMMENDATION_DECISION_MSG
    )
    return {
        "conversation_stage": STAGE_NEGOTIATING,
        "current_step": STEP_RECOMMENDATION,
        "trigger_search": False,
        "negotiation_mode": "",
        "spoken_reply": combined,
        "messages": [AIMessage(content=RECOMMENDATION_DECISION_MSG)],
    }


def pre_closure_node(state: GraphState) -> dict[str, Any]:
    title = state.get("agreed_property_title") or state.get("best_match_title") or "N/A"
    return {
        "conversation_stage": STAGE_PRE_CLOSURE,
        "agreed_property_title": title,
        "user_intent_level": "High",
        "messages": [AIMessage(content=PRE_CLOSURE_MSG)],
    }


def matched_property_summary(state: GraphState) -> str:
    props = state.get("retrieved_properties") or []
    agreed_title = state.get("agreed_property_title") or state.get("best_match_title") or ""
    matched = next(
        (p for p in props if str(p.get("title", "")) == agreed_title),
        props[0] if props else {},
    )
    if not matched:
        return "لم يتم اختيار عقار بعد"
    price_egp = listing_price_egp(matched)
    type_ar = localize_property_type(str(matched.get("type", "")))
    area_ar = localize_area_phrase(str(matched.get("location", "")))
    finish = localize_finish_hint(str(matched.get("title", "")))
    finish_bit = f"، {finish}" if finish else ""
    return (
        f"{type_ar} في {area_ar}{finish_bit} — "
        f"{matched.get('bedroom', '—')} أوض — "
        f"{price_to_spoken_arabic(price_egp)}"
    )


def build_conversation_summary(state: GraphState) -> dict[str, Any]:
    """Sales-department handoff sheet (Arabic + structured fields for n8n email)."""
    name = get_stored_name(state) or "—"
    email = (state.get("cx_email") or "").strip() or "—"
    location = (state.get("location") or "").strip() or "—"
    budget = normalize_budget(int(state.get("budget") or 0))
    budget_display = f"{budget:,} EGP" if budget > 0 else "—"
    property_line = matched_property_summary(state)
    summary_ar = (
        f"عميل: {name} | إيميل: {email} | منطقة البحث: {location} | "
        f"الميزانية: {budget_display} | العقار المختار: {property_line}"
    )
    return {
        "cx_name": name,
        "cx_email": email,
        "preferred_location": location,
        "budget_egp": budget if budget > 0 else None,
        "budget_display": budget_display,
        "matched_property": property_line,
        "summary_ar": summary_ar,
        "summary_en": (
            f"Client {name} ({email}) looking in {location}, budget {budget_display}. "
            f"Selected: {property_line}"
        ),
    }


def build_lead_export(state: GraphState) -> dict[str, Any]:
    """Clean lead dict for n8n / demo terminal output."""
    sheet = build_conversation_summary(state)
    return {
        "cx_name": sheet["cx_name"],
        "cx_email": sheet["cx_email"],
        "budget": sheet["budget_egp"],
        "location": sheet["preferred_location"],
        "matched_property_summary": sheet["matched_property"],
        "conversation_summary_ar": sheet["summary_ar"],
        "conversation_summary_en": sheet["summary_en"],
    }


def build_n8n_lead_payload(state: GraphState) -> dict[str, Any]:
    """Full webhook envelope for n8n CRM automation."""
    export = build_lead_export(state)
    sales_priority = "High" if state.get("user_accepted_recommendation") else "Medium"
    return {
        "event": "sales_lead_captured",
        "source": "voice_agent_rag",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sales_priority": sales_priority,
        "department": "sales",
        "action": "email_lead_sheet",
        **export,
    }


def lead_capture_node(state: GraphState) -> dict[str, Any]:
    """Terminal node: emit JSON payload for n8n webhook ingestion."""
    payload = build_n8n_lead_payload(state)
    print("\n[LEAD_CAPTURE — n8n payload]")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print()
    return {
        "n8n_lead_payload": payload,
        "lead_captured": True,
        "current_step": STEP_CLOSING,
    }


def closing_node(state: GraphState) -> dict[str, Any]:
    """CLOSING / EXIT — polite goodbye; sales team handoff when lead captured."""
    if state.get("exit_search_loop") and not state.get("property_confirmed"):
        msg = POLITE_EXIT_MSG
    else:
        msg = CLOSING_TEAM_MSG
    return {
        "conversation_stage": STAGE_CLOSED,
        "current_step": STEP_EXIT,
        "trigger_search": False,
        "exit_search_loop": True,
        "spoken_reply": msg,
        "messages": [AIMessage(content=msg)],
    }


def final_signoff_node(state: GraphState) -> dict[str, Any]:
    sales_priority = "High" if state.get("user_accepted_recommendation") else "Medium"
    name = get_stored_name(state) or "فندم"
    msg = FINAL_SIGNOFF_MSG.format(name=name)
    return {
        "conversation_stage": STAGE_CLOSED,
        "current_step": STEP_CLOSING,
        "lead_captured": True,
        "user_intent_level": sales_priority,
        "spoken_reply": msg,
        "messages": [AIMessage(content=msg)],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def is_closing_conversation(state: GraphState) -> bool:
    intent = state.get("user_intent", "")
    if intent in ("goodbye", "done") or state.get("is_goodbye"):
        return True
    stage = state.get("conversation_stage") or ""
    if bool(state.get("property_presented")) and stage == STAGE_PRE_CLOSURE and intent == "done":
        return True
    return False


def route_after_response(
    state: GraphState,
) -> Literal["ask_for_email", "final_signoff", "__end__"]:
    """
    Email guardrail: before lead capture, require cx_email when the user ends the
  call or after a property was successfully presented.
    """
    if not is_closing_conversation(state):
        return "__end__"
    if not (state.get("cx_email") or "").strip():
        return "ask_for_email"
    return "final_signoff"


def route_after_greeting(
    state: GraphState,
) -> Literal["analyst", "post_search_intent", "ask_for_email", "closing", "post_email_intent"]:
    """Sales funnel entry — decision gate, intake, email, or exit."""
    if state.get("exit_search_loop") or state.get("current_step") == STEP_EXIT:
        return "closing"
    if state.get("lead_captured"):
        return "closing"
    if state.get("property_confirmed"):
        if (state.get("cx_email") or "").strip():
            return "closing"
        return "analyst"
    if (state.get("conversation_stage") or "") == STAGE_POST_EMAIL:
        return "post_email_intent"
    if (state.get("conversation_stage") or "") == STAGE_NEGOTIATING:
        return "post_search_intent"
    return "analyst"


def route_after_analyst(
    state: GraphState,
) -> Literal["search", "intake", "ask_for_email", "closing", "post_email_intent"]:
    """Route by current_step / trigger_search — never search after exit."""
    if state.get("exit_search_loop") or state.get("current_step") == STEP_EXIT:
        return "closing"
    if state.get("lead_captured"):
        return "closing"
    if (state.get("conversation_stage") or "") == STAGE_POST_EMAIL:
        return "post_email_intent"
    if state.get("property_confirmed") and (state.get("cx_email") or "").strip():
        return "post_email_intent"
    if state.get("property_confirmed") or state.get("current_step") == STEP_LEAD_CAPTURE:
        return "ask_for_email"
    if (
        state.get("trigger_search")
        and has_location_and_budget(state)
        and not state.get("exit_search_loop")
    ):
        return "search"
    if state.get("current_step") == STEP_INTAKE or not lead_keys_complete(state):
        return "intake"
    return "intake"


def route_after_post_search_intent(
    state: GraphState,
) -> Literal[
    "ask_for_email",
    "search",
    "closing",
    "final_signoff",
    "__end__",
]:
    """DECISION GATE routing — block search loop after user said enough."""
    if state.get("exit_search_loop"):
        return "closing"
    intent = state.get("user_intent", "search")
    if intent == "noop":
        return "__end__"
    if intent in ("goodbye", "done"):
        return "closing"
    if intent in ("positive", "satisfied"):
        return "ask_for_email"
    if intent in ("wants_changes", "wants_more", "search"):
        if state.get("exit_search_loop"):
            return "closing"
        if state.get("trigger_search"):
            return "search"
        return "__end__"
    if intent == "collect":
        return "__end__"
    return "__end__"


def route_after_search(
    state: GraphState,
) -> Literal["soft_search", "final_response", "__end__"]:
    if state.get("missing_info"):
        return "__end__"
    if not state.get("retrieved_properties"):
        return "soft_search"
    return "final_response"


def route_after_soft_search(
    state: GraphState,
) -> Literal["final_response"]:
    return "final_response"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("greeting", greeting_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("intake", intake_node)
    workflow.add_node("post_search_intent", intent_node)
    workflow.add_node("search", search_node)
    workflow.add_node("soft_search", soft_search_node)
    workflow.add_node("final_response", final_response_node)
    workflow.add_node("consultation", consultation_node)
    workflow.add_node("pre_closure", pre_closure_node)
    workflow.add_node("ask_for_name", ask_for_name_node)
    workflow.add_node("ask_for_email", ask_for_email_node)
    workflow.add_node("final_signoff", final_signoff_node)
    workflow.add_node("lead_capture", lead_capture_node)
    workflow.add_node("closing", closing_node)
    workflow.add_node("post_email_intent", post_email_intent_node)

    workflow.set_entry_point("greeting")
    workflow.add_conditional_edges(
        "greeting",
        route_after_greeting,
        {
            "analyst": "analyst",
            "post_search_intent": "post_search_intent",
            "ask_for_email": "ask_for_email",
            "closing": "closing",
            "post_email_intent": "post_email_intent",
        },
    )
    workflow.add_conditional_edges(
        "analyst",
        route_after_analyst,
        {
            "search": "search",
            "intake": "intake",
            "ask_for_email": "ask_for_email",
            "closing": "closing",
            "post_email_intent": "post_email_intent",
        },
    )
    workflow.add_edge("intake", END)
    workflow.add_conditional_edges(
        "post_search_intent",
        route_after_post_search_intent,
        {
            "ask_for_email": "ask_for_email",
            "search": "search",
            "closing": "closing",
            "final_signoff": "final_signoff",
            "__end__": END,
        },
    )
    workflow.add_edge("ask_for_name", END)
    workflow.add_conditional_edges(
        "ask_for_email",
        lambda s: "post_email_intent"
        if (s.get("cx_email") or "").strip()
        else "__end__",
        {"post_email_intent": "post_email_intent", "__end__": END},
    )
    workflow.add_conditional_edges(
        "post_email_intent",
        lambda s: "analyst"
        if (s.get("current_step") == STEP_INTAKE and not s.get("lead_captured"))
        else "lead_capture",
        {"analyst": "analyst", "lead_capture": "lead_capture"},
    )
    workflow.add_edge("closing", END)
    workflow.add_conditional_edges(
        "search",
        route_after_search,
        {
            "soft_search": "soft_search",
            "final_response": "final_response",
            "__end__": END,
        },
    )
    workflow.add_edge("soft_search", "final_response")
    workflow.add_edge("final_response", "consultation")
    workflow.add_conditional_edges(
        "consultation",
        route_after_response,
        {
            "ask_for_email": "ask_for_email",
            "final_signoff": "final_signoff",
            "__end__": END,
        },
    )
    workflow.add_edge("pre_closure", END)
    workflow.add_edge("final_signoff", "lead_capture")
    workflow.add_edge("lead_capture", END)

    return workflow.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def print_agent_reply(state: GraphState) -> None:
    last_human_idx = -1
    for i, msg in enumerate(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_idx = i
    for msg in reversed(state["messages"][last_human_idx + 1 :]):
        if isinstance(msg, AIMessage):
            print(f"\nAgent: {msg.content}\n")
            break


def main() -> int:
    if not CHROMA_PATH.is_dir():
        print(f"Error: ChromaDB not found at {CHROMA_PATH}. Run fast_index.py first.")
        return 1

    print("=" * 52)
    print("  Egyptian Real Estate Agent — Negotiation Loop")
    print("  Type 'quit' or 'exit' to leave.")
    print("=" * 52)

    graph = build_graph()
    config = {"configurable": {"thread_id": "demo-session-1"}}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            print_agent_reply(result)
        except GeminiQuotaExceeded:
            print(f"\nAgent: {QUOTA_BUSY_MSG}\n")
        except genai_errors.APIError as exc:
            if is_quota_error(exc):
                print(f"\nAgent: {QUOTA_BUSY_MSG}\n")
            else:
                print(f"\n[Error] Gemini API: {exc}\n")
        except Exception as exc:
            print(f"\n[Error] {exc}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
