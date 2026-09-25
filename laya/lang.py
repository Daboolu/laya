"""无额外依赖的语言与文字检测，用于选择英文或多语言 checkpoint。"""
import re
import unicodedata
from typing import Dict, List, Optional, Union

_SCRIPT_RANGES = [
    ("greek", ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),
    ("cyrillic", ((0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F))),
    ("armenian", ((0x0530, 0x058F),)),
    ("hebrew", ((0x0590, 0x05FF),)),
    ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("devanagari", ((0x0900, 0x097F), (0xA8E0, 0xA8FF))),
    ("bengali", ((0x0980, 0x09FF),)),
    ("gurmukhi", ((0x0A00, 0x0A7F),)),
    ("gujarati", ((0x0A80, 0x0AFF),)),
    ("oriya", ((0x0B00, 0x0B7F),)),
    ("tamil", ((0x0B80, 0x0BFF),)),
    ("telugu", ((0x0C00, 0x0C7F),)),
    ("kannada", ((0x0C80, 0x0CFF),)),
    ("malayalam", ((0x0D00, 0x0D7F),)),
    ("sinhala", ((0x0D80, 0x0DFF),)),
    ("thai", ((0x0E00, 0x0E7F),)),
    ("lao", ((0x0E80, 0x0EFF),)),
    ("tibetan", ((0x0F00, 0x0FFF),)),
    ("myanmar", ((0x1000, 0x109F),)),
    ("georgian", ((0x10A0, 0x10FF),)),
    ("ethiopic", ((0x1200, 0x137F),)),
    ("khmer", ((0x1780, 0x17FF),)),
    ("hangul", ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF))),
    ("kana", ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF))),
    ("han", ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))),
]

_STOP = {
    "en": {"the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
           "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
           "will", "can", "would", "there", "their", "what", "which", "please", "we", "i"},
    "fr": {"le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
           "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce",
           "et", "du", "au", "ou", "je", "tu", "il", "elle", "ils", "elles", "mon", "ton",
           "ma", "ta", "sa", "mes", "tes", "ses", "ces", "deux", "trois", "très", "bien",
           "tout", "tous", "toute", "fait", "veux", "veut", "peux", "peut", "dois", "doit",
           "merci", "bonjour", "jour", "jours", "mois", "fois", "quand", "comment", "pourquoi",
           "alors", "donc"},
    "de": {"der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
           "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber",
           "ich", "wir", "mir", "mich", "dir", "dich", "uns", "mein", "meine", "meinen",
           "meinem", "meiner", "diese", "dieser", "diesen", "dieses", "einen", "einem", "einer",
           "wie", "wo", "wann", "welche", "im", "zum", "zur", "aus", "bei", "nach", "noch", "bitte",
           "heute", "jetzt", "kann", "kannst", "habe", "gibt", "wird",
           "in", "was"},
    "es": {"el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
           "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus",
           "la", "un", "y", "al", "lo", "le", "les", "su", "mi", "tu", "nos",
           "ni", "dos", "tres", "fue", "fueron", "ser", "tiene", "tienen", "tengo", "puede",
           "pueden", "quiero", "necesito", "hemos", "han", "sobre", "entre", "cuando", "donde",
           "porque", "aunque", "también", "ya", "eso", "esto", "esa", "ese", "nada", "algo",
           "aquí", "hoy", "gracias"},
    "pt": {"os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
           "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela",
           "o", "e", "na", "nas", "nos", "ao", "aos", "por", "foi", "era", "ser", "sou",
           "tem", "tenho", "pode", "podem", "quero", "preciso", "eu", "meu", "minha", "seu",
           "sua", "isso", "isto", "aqui", "ali", "como", "quando", "onde", "porque", "mais",
           "já", "ainda", "agora", "hoje", "ontem", "dois", "três", "tudo", "nada", "obrigado",
           "olá",
           "você", "vocês", "voce", "voces", "vc", "vcs", "nao", "sao", "ja", "até", "tá", "pra",
           "gostaria", "obrigada", "também", "tambem", "estou", "estamos", "meus", "minhas",
           "nosso", "nossa", "consigo", "cadê", "boa", "tarde", "noite",
           "depois", "antes", "então", "entao", "ninguém", "ninguem", "alguém", "alguem", "nenhum",
           "nenhuma", "estava", "ficou", "fiz", "deu"},
    "it": {"il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
           "questo", "questa", "anche", "come", "più", "sono", "nella", "alla",
           "la", "le", "un", "uno", "una", "e", "ed", "o", "da", "su", "tra", "fra", "mi",
           "ci", "ne", "ho", "hai", "ha", "abbiamo", "avete", "hanno", "era", "stato", "stata",
           "devo", "deve", "devono", "voglio", "vorrei", "mio", "mia", "tuo", "sua", "quando",
           "dove", "perche", "molto", "poco", "sempre", "mai", "già", "ancora", "adesso", "oggi",
           "ieri", "grazie", "ciao", "scusa",
           "nel", "nell", "negli", "sul", "sulla", "sulle", "dal", "dalla", "dallo", "dagli", "dei",
           "delle", "dello", "degli", "agli", "alle", "col"},
    "nl": {"het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
           "door", "maar", "ook", "worden", "deze", "naar", "wordt"},
    "ro": {"și", "să", "este", "sunt", "care", "pentru", "din", "dar", "după", "până", "fără",
           "ale", "lui", "în", "fost", "acum", "vreau", "trebuie", "foarte", "acest", "această",
           "acesta", "aceasta", "mi", "ți", "vă", "nu"},
    "bn": {"ami", "amar", "amake", "amra", "amader", "apni", "apnar", "apnake", "apnara",
           "tumi", "tomar", "tomake", "tomra", "tader", "ota", "eita", "oita",
           "ekta", "ei", "oi", "ki", "keno", "kivabe", "kibhabe", "kothay", "kokhon", "kobe",
           "koto", "kintu", "jodi", "tahole", "ar", "theke", "jonno", "sathe", "shathe", "diye",
           "niye", "moddhe", "kore", "korte", "korchi", "korsi", "korbo", "korechi", "koreche",
           "korun", "koren", "korlam", "hobe", "hoyeche", "hoise", "hocche", "hoyni",
           "chai", "chaina", "lagbe", "parchi", "parbo", "parchina", "peyechi", "paini",
           "dite", "dilam", "diyechi", "nai", "khub", "onek", "ekhon", "akhon", "ekhono",
           "abar", "ekbar", "duibar", "ajke", "kalke", "taka", "bhalo", "valo", "kharap",
           "shomossa", "somossa", "dhonnobad", "bhai", "shob", "keu", "kichu", "bolte", "bolun",
           "parben", "asbe", "jabe", "pabo", "ferot", "dorkar", "hoye", "geche", "gese"},

    "az": {"və", "ve", "bir", "bu", "üçün", "ucun", "ilə", "ile", "olan", "olub", "olmasa",
           "var", "yox", "yoxdur", "mən", "sən", "biz", "siz", "onlar", "daha", "çox", "cox",
           "hər", "nə", "kimi", "görə", "sonra", "əgər", "eger", "deyil", "lakin", "amma",
           "ancaq", "artıq", "artiq", "də", "isə", "həm", "yalnız", "yalniz"},
}
_NON_EN_DIACRITICS = set(
    "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ"
    "ăâîșțşţ"
    "ąćęłńśźż"
    "čďěňřšťůž"
    "őű"
    "ğı"
    "āēģīķļņūž"
    "đ"
    "ə"
)
_SHARED_WORDS = {w for w in {word for words in _STOP.values() for word in words}
                 if sum(w in words for words in _STOP.values()) > 1}

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_IDENTIFIER = re.compile(r"(?<![\w-])[\w-]*(?:[.@][\w-]+)+", re.UNICODE)


def _iter_text(state: Union[str, dict, list, None], _depth: int = 0) -> List[str]:
    if _depth > 6 or state is None:
        return []
    if isinstance(state, str):
        return [state]
    if isinstance(state, dict):
        out = []
        for v in state.values():
            out.extend(_iter_text(v, _depth + 1))
        return out
    if isinstance(state, (list, tuple)):
        out = []
        for v in state:
            out.extend(_iter_text(v, _depth + 1))
        return out
    return []


def state_text(state: Union[str, dict, list, None], max_chars: int = 4000) -> str:
    parts: List[str] = []
    budget = max_chars
    for leaf in _iter_text(state):
        if budget <= 0:
            break
        if len(leaf) > budget:
            parts.append(leaf[:budget])
            break
        parts.append(leaf)
        budget -= len(leaf) + 1
    return " ".join(parts)[:max_chars]


def _script_counts(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    latin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if cp < 0x02B0 or 0x1E00 <= cp <= 0x1EFF or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A:
            latin += 1
            continue
        for name, ranges in _SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            counts["other"] = counts.get("other", 0) + 1
    counts["latin"] = latin
    return counts


def _script_from_counts(counts: Dict[str, int]) -> str:
    if not any(counts.values()):
        return "unknown"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _profile_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if not total:
        return {}
    ordered: Dict[str, int] = {}
    if counts.get("latin"):
        ordered["latin"] = counts["latin"]
    for name, value in counts.items():
        if name != "latin":
            ordered[name] = value
    return {k: v / total for k, v in ordered.items() if v}


def detect_script(text: str) -> str:
    """返回主导文字系统；没有字母时返回 unknown。"""
    return _script_from_counts(_script_counts(text))


def script_profile(text: str) -> Dict[str, float]:
    """返回各文字系统占字母总数的比例。"""
    return _profile_from_counts(_script_counts(text))


NON_EN_DIACRITIC_RATE = 0.02

NON_LATIN_FRACTION = 0.2
NON_LATIN_MIN_FRACTION = 0.1
NON_LATIN_MIN_LETTERS = 10


def _script_of(ch: str) -> Optional[str]:
    cp = ord(ch)
    if cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A:
        return None
    for name, ranges in _SCRIPT_RANGES:
        if any(lo <= cp <= hi for lo, hi in ranges):
            return name
    return None


def _non_latin_words(text: str) -> List[str]:
    runs, cur, script = [], "", None
    for ch in text:
        if unicodedata.combining(ch):
            continue
        s = _script_of(ch)
        if s is not None and s == script:
            cur += ch
            continue
        if cur:
            runs.append(cur)
        cur, script = (ch, s) if s is not None else ("", None)
    if cur:
        runs.append(cur)
    return [w for w in runs if len(w) >= 2 and not w[0].isupper()]


def latin_profile(text: str) -> Dict[str, object]:
    words = _WORD.findall(_IDENTIFIER.sub(" ", text).replace("İ", "i").lower())
    lowered = text.lower()
    diac = sum(1 for ch in lowered if ch in _NON_EN_DIACRITICS)
    diac_rate = diac / max(1, len(lowered))
    non_english = diac_rate >= NON_EN_DIACRITIC_RATE
    if len(words) < 4:
        return {"language": None, "english_hits": 0, "diacritic_rate": diac_rate,
                "looks_non_english": non_english}

    scores = {lg: sum(1 for w in words if w in sw) for lg, sw in _STOP.items()}
    en = scores.get("en", 0)
    evidenced = {lg: s for lg, s in scores.items()
                 if lg != "en" and any(w not in _SHARED_WORDS for w in set(words) & _STOP[lg])}
    best_lg, best = max(evidenced.items(), key=lambda kv: kv[1], default=(None, 0))

    lang = None
    if best_lg and best >= max(2, en + 2):
        lang = best_lg
    elif best_lg and non_english and best >= max(2, en):
        lang = best_lg
    elif en and not non_english:
        lang = "en"
    return {"language": lang, "english_hits": en, "diacritic_rate": diac_rate,
            "looks_non_english": non_english}


def guess_latin_language(text: str) -> Optional[str]:
    """尽力判断拉丁文字语言；证据不足时返回 None。"""
    return latin_profile(text)["language"]


_CODE_LINE = re.compile(r"[=;{}\[\]]|\w\(")
_JOINED = re.compile(r"[^\W_][._/\\][^\W_]")
_LETTER_RUN = re.compile(r"[^\W\d_]{2,}")


def _non_english_segment(state: Union[str, dict, list, None], max_chars: int = 4000):
    seen = 0
    for leaf in _iter_text(state):
        for seg in leaf.split("\n"):
            if seen >= max_chars:
                return None
            seg = seg[:max_chars - seen]
            seen += len(seg)
            if _CODE_LINE.search(seg):
                continue
            prose = " ".join(tok for tok in seg.split() if not _JOINED.search(tok))
            if any(ch.islower() for ch in prose):
                prose = _LETTER_RUN.sub(lambda m: " " if m.group().isupper() else m.group(), prose)
            tokens = _WORD.findall(prose)
            if len(tokens) < 4:
                continue
            lang = latin_profile(prose)["language"]
            if lang not in (None, "en") and len({w.lower() for w in tokens} & _STOP.get(lang, set())) >= 2:
                return lang, seg.strip()
    return None


def analyse(state: Union[str, dict, list, None]) -> Dict[str, object]:
    """返回状态的完整路由检测结果。"""
    text = state_text(state)
    counts = _script_counts(text)
    prof = _profile_from_counts(counts)
    script = _script_from_counts(counts)
    non_latin = round(1.0 - prof.get("latin", 0.0), 4) if prof else 0.0
    n_non_latin = round(non_latin * sum(ch.isalpha() for ch in text))
    if script == "latin" and _non_latin_words(text) and (
            non_latin >= NON_LATIN_FRACTION or (
                non_latin >= NON_LATIN_MIN_FRACTION and n_non_latin >= NON_LATIN_MIN_LETTERS)):
        script = max((s for s in prof if s != "latin"), key=prof.get)
    if script == "unknown":
        return {"script": "unknown", "script_profile": prof, "language": None,
                "is_english": True, "language_undecided": True, "diacritic_rate": 0.0,
                "non_latin_fraction": 0.0, "mixed_segment": None}
    if script != "latin":
        return {"script": script, "script_profile": prof, "language": None,
                "is_english": False, "language_undecided": True, "diacritic_rate": 0.0,
                "non_latin_fraction": non_latin, "mixed_segment": None}
    prof_lat = latin_profile(text)
    lang = prof_lat["language"]
    undecided = lang is None
    english = lang == "en" or (undecided and not prof_lat["looks_non_english"])
    mixed = None
    leaves = _iter_text(state)
    if english and (len(leaves) > 1 or any("\n" in leaf for leaf in leaves)):
        found = _non_english_segment(state)
        if found:
            lang, mixed = found
            english, undecided = False, False
    return {"script": "latin", "script_profile": prof, "language": lang,
            "is_english": english, "language_undecided": undecided,
            "diacritic_rate": round(float(prof_lat["diacritic_rate"]), 4),
            "non_latin_fraction": non_latin, "mixed_segment": mixed}


def is_english(state: Union[str, dict, list, None]) -> bool:
    """英文 checkpoint 是否适合读取该状态。"""
    return bool(analyse(state)["is_english"])
