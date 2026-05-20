import os
import re
import difflib
from typing import Optional

from dotenv import load_dotenv
from google import genai


load_dotenv()

_PACKAGE_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
if os.path.exists(_PACKAGE_ENV_PATH):
    load_dotenv(_PACKAGE_ENV_PATH, override=False)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)


VALID_LABELS = [
    "blue_cube",
    "blue_cylinder",
    "objects-yellow_box",
    "yellow_cylinder",
]

PICK_KEYWORDS = [
    "집어", "집어줘", "잡아", "잡아줘",
    "들어", "들어줘", "가져와", "가져와줘",
    "넣어", "넣어줘", "옮겨", "옮겨줘",
    "놔", "놔줘", "놓아", "놓아줘",
    "정리", "정리해", "정리해줘",
    "pick", "grab", "take",
]

DEFAULT_PLACE_DISTANCE_M = 0.15

ZONE_BY_LABEL = {
    "objects-yellow_box": "A",
    "yellow_cylinder": "A",
    "blue_cube": "B",
    "blue_cylinder": "B",
}

ZONE_RULES = [
    {
        "zone": "A",
        "terms": ["a구역", "a 구역", "에이구역", "에이 구역", "zone a", "a zone"],
    },
    {
        "zone": "B",
        "terms": ["b구역", "b 구역", "비구역", "비 구역", "zone b", "b zone"],
    },
]

DIRECTION_RULES = [
    {
        "axis": "y",
        "sign": -1.0,
        "display": "오른쪽",
        "terms": ["오른쪽", "오른 쪽", "우측", "오른편", "right"],
    },
    {
        "axis": "y",
        "sign": 1.0,
        "display": "왼쪽",
        "terms": ["왼쪽", "왼 쪽", "좌측", "왼편", "left"],
    },
    {
        "axis": "x",
        "sign": 1.0,
        "display": "앞쪽",
        "terms": ["앞으로", "앞쪽", "전방", "forward"],
    },
    {
        "axis": "x",
        "sign": -1.0,
        "display": "뒤쪽",
        "terms": ["뒤로", "뒤쪽", "후방", "backward", "back"],
    },
    {
        "axis": "z",
        "sign": 1.0,
        "display": "위쪽",
        "terms": ["위로", "위쪽", "up"],
    },
    {
        "axis": "z",
        "sign": -1.0,
        "display": "아래쪽",
        "terms": ["아래로", "아래쪽", "down"],
    },
]

LABEL_SYNONYMS = {
    "blue_cube": [
        "blue cube", "blue box", "blue block",
        "파란색 큐브", "파란색 블록",
        "파란 큐브", "파란 블록",
        "파란색 박스", "파란 박스",
        "파란색 직육면체", "파란 직육면체",
        "파란색 상자", "파란 상자",
        "파란색큐브", "파란색블록", "파란색박스",
        "파란큐브", "파란블록", "파란박스",
        "파란색직육면체"
    ],
    "blue_cylinder": [
        "blue cylinder",
        "파란색 원통", "파란색 실린더",
        "파란 원통", "파란 실린더",
        "파란색원통", "파란색실린더", "파란원통", "파란실린더"
    ],
    "objects-yellow_box": [
        "yellow box", "yellow block",
        "노란 박스", "노란색 박스",
        "노란 블록", "노란색 블록",
        "노란색 직육면체", "노란 직육면체",
        "노란 상자", "노란색 상자",
        "노란박스", "노란색박스",
        "노란블록", "노란색블록",
        "노란색직육면체", "노란직육면체",
        "노란상자", "노란색상자"
    ],
    "yellow_cylinder": [
        "yellow cylinder",
        "노란색 원통", "노란색 실린더",
        "노란 원통", "노란 실린더",
        "노란색원통", "노란색실린더", "노란원통", "노란실린더"
    ],
}

YES_WORDS = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "응", "네", "맞아", "그래"}
NO_WORDS = {"no", "n", "nope", "아니", "아니야", "취소"}

_last_suggestion = None


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def has_pick_intent(text: str) -> bool:
    normalized = normalize_text(text)
    compact = compact_text(text)
    for keyword in sorted(PICK_KEYWORDS, key=len, reverse=True):
        normalized_keyword = normalize_text(keyword)
        compact_keyword = compact_text(keyword)
        if re.search(_phrase_pattern(normalized_keyword), normalized):
            return True
        if compact_keyword and compact_keyword in compact:
            return True
    return False


def find_labels_exact(text: str) -> list[str]:
    normalized = normalize_text(text)
    compact = compact_text(text)
    matches = []

    for label_index, label in enumerate(VALID_LABELS):
        for term in _label_terms(label):
            for match in re.finditer(_phrase_pattern(term), normalized):
                matches.append(
                    {
                        "label": label,
                        "label_index": label_index,
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
            compact_term = compact_text(term)
            if not compact_term:
                continue
            start = compact.find(compact_term)
            while start >= 0:
                matches.append(
                    {
                        "label": label,
                        "label_index": label_index,
                        "start": start,
                        "end": start + len(compact_term),
                    }
                )
                start = compact.find(compact_term, start + 1)

    selected = []
    selected_labels = set()

    for match in sorted(matches, key=lambda item: (-(item["end"] - item["start"]), item["start"], item["label_index"])):
        if match["label"] in selected_labels:
            continue

        overlapping = [
            item
            for item in selected
            if not (match["end"] <= item["start"] or match["start"] >= item["end"])
        ]
        if overlapping:
            same_span = all(
                item["start"] == match["start"] and item["end"] == match["end"]
                for item in overlapping
            )
            if not same_span:
                continue

        selected.append(match)
        selected_labels.add(match["label"])

    selected.sort(key=lambda item: (item["start"], item["label_index"]))
    return [item["label"] for item in selected]


def fuzzy_label(text: str, candidates: list[str]) -> Optional[str]:
    candidate_labels = _ordered_valid_labels(candidates)
    if not candidate_labels:
        return None

    query = _strip_command_words(normalize_text(text))
    if not query:
        query = normalize_text(text)

    best_label = None
    best_score = 0.0

    for label in candidate_labels:
        for term in _label_terms(label):
            score = max(
                difflib.SequenceMatcher(None, query, term).ratio(),
                _partial_ratio(query, term),
            )
            if score > best_score:
                best_score = score
                best_label = label

    if best_score >= 0.76:
        return best_label
    return None


def visible_filter(labels: list[str], visible_objects: Optional[set[str]]) -> list[str]:
    if visible_objects is None:
        return labels

    visible_labels = set(_ordered_valid_labels(list(visible_objects)))
    return [label for label in labels if label in visible_labels]


def ask_gemini_chat(user_text: str) -> str:
    if gemini_client is None:
        return "Gemini API 키가 아직 설정되지 않았어. .env 파일에 GEMINI_API_KEY를 넣어줘."

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=(
                "너는 로봇팔 명령을 돕는 한국어 assistant야.\n"
                "사용자가 한국어로 말하면 한국어로 짧고 자연스럽게 답해.\n"
                "집기 명령은 실제 실행하지 말고, 필요한 정보가 부족하면 짧게 물어봐.\n\n"
                f"사용자: {user_text}"
            ),
        )
        return (response.text or "").strip() or "무슨 작업을 할지 말해줘."

    except Exception as e:
        return f"Gemini 호출 중 오류가 발생했어: {e}"


def llm_chat_or_pick(user_text: str, visible_objects: Optional[set[str]] = None) -> dict:
    global _last_suggestion

    normalized = normalize_text(user_text)
    if not normalized:
        return {"type": "chat", "reply": "명령을 입력해줘."}

    place_offset = parse_relative_place_offset(user_text)
    place_zone = None if place_offset else parse_zone_place(user_text)
    force_pick_flow = False

    if _last_suggestion is not None:
        suggestion = _last_suggestion

        if has_pick_intent(normalized) or find_labels_exact(normalized):
            _last_suggestion = None
            force_pick_flow = True
        elif _is_yes(normalized):
            label = suggestion["label"]
            _last_suggestion = None

            labels = visible_filter([label], visible_objects)
            if not labels:
                return {
                    "type": "chat",
                    "reply": f"지금 보이는 물체 중에서는 {_format_label(label)} 를 찾지 못했어.",
                }

            return _pick_queue(
                labels,
                suggestion.get("place_offset"),
                suggestion.get("place_zone"),
            )

        elif _is_no(normalized):
            _last_suggestion = None
            return {"type": "chat", "reply": "알겠어. 물체 이름을 조금 더 자세히 말해줘."}

        else:
            return {
                "type": "chat",
                "reply": f"{_format_label(suggestion['label'])} 를 말한 게 맞아? 맞으면 '네', 아니면 '아니'라고 말해줘.",
            }

    if force_pick_flow or has_pick_intent(normalized):
        labels = find_labels_exact(normalized)
        if labels:
            labels = visible_filter(labels, visible_objects)
            if not labels:
                return {
                    "type": "chat",
                    "reply": "지금 보이는 물체 중에서는 요청한 물체를 찾지 못했어.",
                }

            if len(labels) != 1:
                return {"type": "chat", "reply": "물체를 하나만 말해줘."}

            return _pick_queue(labels, place_offset, place_zone)

        candidates = _ordered_valid_labels(list(visible_objects)) if visible_objects else VALID_LABELS
        suggestion = fuzzy_label(normalized, candidates)
        if suggestion:
            _last_suggestion = {
                "label": suggestion,
                "place_offset": place_offset,
                "place_zone": place_zone,
            }
            return {
                "type": "chat",
                "reply": f"{_format_label(suggestion)} 를 말한 게 맞아? 맞으면 '네', 아니면 '아니'라고 말해줘.",
            }

        return {
            "type": "chat",
            "reply": "집을 물체를 다시 말해줘. 예: 노란색 박스 집어줘",
        }

    labels = visible_filter(find_labels_exact(normalized), visible_objects)
    if len(labels) == 1:
        _last_suggestion = {
            "label": labels[0],
            "place_offset": place_offset,
            "place_zone": place_zone,
        }
        return {
            "type": "chat",
            "reply": f"{_format_label(labels[0])} 를 집을까?",
        }
    if len(labels) > 1:
        return {"type": "chat", "reply": "물체를 하나만 말해줘."}

    return {"type": "chat", "reply": ask_gemini_chat(user_text)}


def parse_relative_place_offset(text: str) -> Optional[dict]:
    normalized = normalize_text(text)
    compact = compact_text(text)

    for rule in DIRECTION_RULES:
        if not _matches_direction(rule["terms"], normalized, compact):
            continue

        distance_m = _parse_place_distance_m(normalized, compact)
        if distance_m is None:
            distance_m = DEFAULT_PLACE_DISTANCE_M

        dx = dy = dz = 0.0
        value = rule["sign"] * distance_m
        if rule["axis"] == "x":
            dx = value
        elif rule["axis"] == "y":
            dy = value
        else:
            dz = value

        return {
            "place_mode": "relative",
            "place_dx": dx,
            "place_dy": dy,
            "place_dz": dz,
            "place_direction": rule["display"],
            "place_distance_m": distance_m,
        }

    return None


def parse_zone_place(text: str) -> Optional[str]:
    normalized = normalize_text(text)
    compact = compact_text(text)

    for rule in ZONE_RULES:
        if _matches_direction(rule["terms"], normalized, compact):
            return rule["zone"]

    return None


def _matches_direction(terms: list[str], normalized: str, compact: str) -> bool:
    for term in terms:
        normalized_term = normalize_text(term)
        compact_term = compact_text(term)
        if re.search(_phrase_pattern(normalized_term), normalized):
            return True
        use_compact = any(not char.isascii() for char in term) or bool(re.search(r"\s", term))
        if use_compact and compact_term and compact_term in compact:
            return True
    return False


def _parse_place_distance_m(normalized: str, compact: str) -> Optional[float]:
    compact_match = re.search(
        r"(\d+(?:\.\d+)?)(cm|센치|센티|센티미터|미터|m)",
        compact,
    )
    if compact_match:
        value = float(compact_match.group(1))
        unit = compact_match.group(2)
        if unit in ("cm", "센치", "센티", "센티미터"):
            return value / 100.0
        return value

    spaced_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(cm|센치|센티|센티미터|미터|m)",
        normalized,
    )
    if spaced_match:
        value = float(spaced_match.group(1))
        unit = spaced_match.group(2)
        if unit in ("cm", "센치", "센티", "센티미터"):
            return value / 100.0
        return value

    return None


def _label_terms(label: str) -> list[str]:
    terms = [
        label,
        label.replace("_", " "),
        label.replace("-", " "),
        label.replace("_", " ").replace("-", " "),
    ]
    terms.extend(LABEL_SYNONYMS.get(label, []))

    normalized_terms = []
    seen = set()
    for term in terms:
        normalized = normalize_text(term)
        if normalized and normalized not in seen:
            normalized_terms.append(normalized)
            seen.add(normalized)
    return normalized_terms


def _phrase_pattern(phrase: str) -> str:
    parts = normalize_text(phrase).split()
    if not parts:
        return r"$^"

    body = r"\s+".join(re.escape(part) for part in parts)
    prefix = r"(?<![a-z0-9])" if parts[0][0].isascii() and parts[0][0].isalnum() else ""
    suffix = r"(?![a-z0-9])" if parts[-1][-1].isascii() and parts[-1][-1].isalnum() else ""
    return prefix + body + suffix


def _ordered_valid_labels(candidates: list[str]) -> list[str]:
    normalized_candidates = {normalize_text(candidate) for candidate in candidates}
    return [
        label
        for label in VALID_LABELS
        if label in candidates or normalize_text(label) in normalized_candidates
    ]


def _strip_command_words(text: str) -> str:
    cleaned = normalize_text(text)

    for keyword in sorted(PICK_KEYWORDS, key=len, reverse=True):
        cleaned = re.sub(_phrase_pattern(keyword), " ", cleaned)

    cleaned = re.sub(r"\b(box|please|to|into|in)\b", " ", cleaned)
    cleaned = re.sub(r"(박스|상자|에|으로|로|를|을|좀)", " ", cleaned)
    return normalize_text(cleaned)


def _partial_ratio(first: str, second: str) -> float:
    first = normalize_text(first)
    second = normalize_text(second)
    if not first or not second:
        return 0.0

    if first == second or first in second or second in first:
        return 1.0

    shorter, longer = (first, second) if len(first) <= len(second) else (second, first)
    window_size = len(shorter)
    best = 0.0

    for index in range(0, len(longer) - window_size + 1):
        window = longer[index:index + window_size]
        best = max(best, difflib.SequenceMatcher(None, shorter, window).ratio())

    return best


def _format_label(label: str) -> str:
    return label.replace("_", " ").replace("-", " ")


def _format_labels(labels: list[str]) -> str:
    return ", ".join(_format_label(label) for label in labels)


def _default_zone_for_labels(labels: list[str]) -> Optional[str]:
    if len(labels) != 1:
        return None
    return ZONE_BY_LABEL.get(labels[0])


def _pick_queue(
    labels: list[str],
    place_offset: Optional[dict] = None,
    place_zone: Optional[str] = None,
) -> dict:
    global _last_suggestion

    _last_suggestion = None
    result = {
        "type": "pick_queue",
        "labels": labels,
        "reply": f"좋아. {_format_labels(labels)} 를 집을게.",
    }

    if place_offset:
        result.update(
            {
                "place_mode": place_offset["place_mode"],
                "place_dx": place_offset["place_dx"],
                "place_dy": place_offset["place_dy"],
                "place_dz": place_offset["place_dz"],
            }
        )
        distance_cm = int(round(place_offset["place_distance_m"] * 100.0))
        result["reply"] = (
            f"좋아. {_format_labels(labels)} 를 집어서 "
            f"{place_offset['place_direction']} {distance_cm}cm로 옮길게."
        )

    else:
        zone = place_zone or _default_zone_for_labels(labels)
        if zone:
            result.update(
                {
                    "place_mode": "zone",
                    "zone": zone,
                }
            )
            result["reply"] = (
                f"좋아. {_format_labels(labels)} 를 집어서 "
                f"{zone}구역에 놓을게."
            )

    return result


def _is_yes(text: str) -> bool:
    normalized = normalize_text(text)
    compact = re.sub(r"[\s.!?,~]+", "", normalized)
    words = set(normalized.split())
    return compact in YES_WORDS or bool(words & YES_WORDS)


def _is_no(text: str) -> bool:
    normalized = normalize_text(text)
    compact = re.sub(r"[\s.!?,~]+", "", normalized)
    words = set(normalized.split())
    return compact in NO_WORDS or bool(words & NO_WORDS)


if __name__ == "__main__":
    visible = {"blue_cylinder", "yellow_cylinder", "objects-yellow_box", "blue_cube"}

    while True:
        text = input("명령 입력: ")
        if text in ["q", "quit", "exit"]:
            break

        result = llm_chat_or_pick(text, visible_objects=visible)
        print(result)
