"""Term banks, Arabic-content detection, and the scraper query catalog.

Pure data plus stateless matchers - no imports from server.py (and there must
never be any: app modules are below server.py in the import graph).
"""
import re


ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
ARABIC_KEYWORDS = [
    "arabic", "arab", "عرب", "السعودية", "سعودي", "الإمارات", "إماراتي",
    "قطر", "قطري", "الكويت", "كويتي", "البحرين", "بحريني", "عمان", "عماني",
    "مصر", "مصري", "الأردن", "أردني", "لبنان", "لبناني", "العراق", "عراقي",
    "إيران", "إيراني", "فلسطين", "فلسطيني", "غزة", "القدس",
    "تركيا", "تركي", "اليمن", "يمني", "سوريا", "سوري",
    "المغرب", "مغربي", "تونس", "تونسي", "الجزائر", "جزائري", "ليبيا", "ليبي",
    "السودان", "سوداني", "موريتانيا", "موريتاني",
    "إسلام", "إسلامي", "مسلم", "قرآن", "حديث", "فقه", "شريعة",
    "عربية", "العربية",
]

def detect_arabic_content(text):
    """Return (is_relevant, keywords, has_script).

    The first value was `has_script or keywords`, but score_arabic_relevance bound
    it as `has_script` and then tagged the paper "arabic_script" and added +5. So a
    pure-English paper mentioning "arab" was reported as containing Arabic script
    with not one Arabic character in it. Script presence is now returned separately
    from keyword relevance.
    """
    if not text:
        return False, [], False
    has_script = bool(ARABIC_PATTERN.search(text))
    text_lower = text.lower()
    found_keywords = [kw for kw in ARABIC_KEYWORDS if _term_pattern(kw).search(text_lower)]
    return (has_script or bool(found_keywords)), found_keywords, has_script

def score_arabic_relevance(paper):
    title = (paper.get("title") or "").lower()
    summary = (paper.get("summary") or "").lower()
    text = title + " " + summary
    score = 0
    details = []
    _relevant, script_words, has_script = detect_arabic_content(text)
    if has_script:
        score += 5
        details.append("arabic_script")
    for kw in script_words:   # already word-boundary matched by detect_arabic_content
        score += 1
        if kw not in details:
            details.append(kw)
    country_terms = {
        "saudi": 3, "arabia": 3, "uae": 3, "emirati": 3, "qatar": 3, "kuwait": 3,
        "bahrain": 3, "oman": 3, "egypt": 3, "jordan": 3, "lebanon": 3, "iraq": 3,
        "iran": 2, "israel": 2, "palestine": 3, "gaza": 3, "turkey": 2, "yemen": 3,
        "syria": 3, "morocco": 3, "tunisia": 3, "algeria": 3, "libya": 3, "sudan": 3,
        "mena": 4, "middle east": 4, "gulf": 3, "arabian": 3,
    }
    for term, boost in country_terms.items():
        if _term_pattern(term).search(text):   # was substring: "oman" hit "Romania"
            score += boost
            details.append(term)
    return score, sorted(set(details))

BEHAVIOURAL_TERMS = [
    "nudge","nudging","behavior","behaviour","cognitive","decision",
    "choice","heuristic","bias","motivation","incentive","reward",
    "punishment","feedback","social norm","conformity","compliance",
    "attitude","perception","learning","memory","emotion","affect",
    "risk","trust","cooperation","altruism","fairness","justice",
    "intervention","policy","regulation","adherence","frame",
    "framing","priming","anchoring","loss aversion","prospect",
    "utility","preference","gamification","habit",
    "automaticity","self-control","willpower","attention","salience",
    "default","opt-in","opt-out","commitment","consistency",
    "reciprocity","authority","scarcity","social proof","liking",
]

# Split deliberately. The original single list mixed place names with generic
# sociology vocabulary, and since the dashboard's "Regional Keywords" panel ranks
# by raw frequency, the generic words buried every actual regional signal.
MENA_PLACE_TERMS = [
    "saudi","arabia","uae","emirati","dubai","abu dhabi","riyadh",
    "jeddah","mecca","medina","qatar","doha","kuwait","bahrain",
    "oman","muscat","egypt","cairo","jordan","amman","lebanon",
    "beirut","iraq","baghdad","iran","israel","palestine","gaza",
    "syria","yemen","morocco","tunisia","algeria","libya","sudan",
    "middle east","mena","gulf","gcc","levant","maghreb",
    "arab","arabic","islamic","muslim","vision 2030","neom",
]

# Retained, but reported separately as context rather than as "regional keywords".
MENA_CONTEXT_TERMS = [
    "conservative","liberal","tribe","tribal","honor","honour","shame",
    "collectivism","individualism","religion","cultural","context",
    "hijab","veil","gender","women","youth","unemployment",
    "diversification","oil","expat","foreign worker",
]

# Kept for backwards compatibility with anything referencing the old name.
MIDDLE_EAST_TERMS = MENA_PLACE_TERMS + MENA_CONTEXT_TERMS

SCRAPER_QUERIES = {
    "broad_behavioural": {"source": "arXiv", "desc": "Behavioral Science (broad)"},
    "broad_me": {"source": "arXiv", "desc": "Middle East (broad)"},
    "saudi": {"source": "arXiv", "desc": "Saudi Arabia"},
    "arab_psychology": {"source": "arXiv", "desc": "Arab Psychology"},
    "mena_health": {"source": "arXiv", "desc": "MENA Health"},
    "behavioural_economics": {"source": "arXiv", "desc": "Behavioral Economics"},
    "digital_behaviour": {"source": "arXiv", "desc": "Digital Behavior"},
    "nudge_policy": {"source": "arXiv", "desc": "Nudge Policy"},
    "arabic_health": {"source": "PubMed", "desc": "Arabic Health (PubMed)"},
    "cultural_psychology": {"source": "PubMed", "desc": "Cultural Psychology (PubMed)"},
    "com_b": {"source": "arXiv", "desc": "COM-B / Behaviour Change Wheel"},
    "tpb": {"source": "arXiv", "desc": "Theory of Planned Behaviour"},
    "hbm": {"source": "arXiv", "desc": "Health Belief Model"},
    "sct": {"source": "arXiv", "desc": "Social Cognitive Theory"},
    "sdt": {"source": "arXiv", "desc": "Self-Determination Theory"},
    "dual_process": {"source": "arXiv", "desc": "Dual Process / Kahneman"},
    "social_norms": {"source": "arXiv", "desc": "Social Norms"},
    "habit_formation": {"source": "arXiv", "desc": "Habit Formation"},
    "health_psychology": {"source": "arXiv", "desc": "Health Psychology"},
    "cultural_psych": {"source": "arXiv", "desc": "Cultural Psychology"},
    "mena_nudge": {"source": "arXiv", "desc": "Nudge + MENA"},
    "mena_health_behaviour": {"source": "arXiv", "desc": "Health Behaviour + MENA"},
    "mena_mental_health": {"source": "arXiv", "desc": "Mental Health + MENA"},
    "mena_women": {"source": "arXiv", "desc": "Women/Gender + MENA"},
    "mena_youth": {"source": "arXiv", "desc": "Youth + MENA"},
    "arabic_transliterated": {"source": "arXiv", "desc": "Arabic transliterated terms"},
    "mena_education": {"source": "arXiv", "desc": "Education + MENA"},
    "mena_business": {"source": "arXiv", "desc": "Business/Management + MENA"},
    "mena_technology": {"source": "arXiv", "desc": "Technology/Digital + MENA"},
    "mena_migration": {"source": "arXiv", "desc": "Migration/Refugee + MENA"},
    "crossref_psychology": {"source": "CrossRef", "desc": "Psychology journals (CrossRef)"},
    "crossref_health": {"source": "CrossRef", "desc": "Health behaviour (CrossRef)"},
    "crossref_mena": {"source": "CrossRef", "desc": "MENA + behaviour (CrossRef)"},
    "semanticscholar_broad": {"source": "SemanticScholar", "desc": "Behavioural science (SS)"},
    "semanticscholar_mena": {"source": "SemanticScholar", "desc": "MENA behavioural (SS)"},
}


_TERM_PATTERNS = {}


def _term_pattern(term):
    """Word-boundary matcher for a term, cached.

    Multi-word terms ("middle east", "abu dhabi") need internal whitespace to stay
    flexible; \\b at each end stops "mena" matching phenoMENA.
    """
    pat = _TERM_PATTERNS.get(term)
    if pat is None:
        pat = re.compile(r"\b" + r"\s+".join(re.escape(w) for w in term.split()) + r"\b")
        _TERM_PATTERNS[term] = pat
    return pat


def _count_terms(text, terms):
    counts = {t: len(_term_pattern(t).findall(text)) for t in terms}
    return sorted(((t, c) for t, c in counts.items() if c > 0), key=lambda x: -x[1])


BEHAVIOURAL_QUERY_PREFIX = "behaviour OR behavior OR cognitive OR decision OR bias OR motivation OR incentive OR nudge OR habit OR \"social norm\" OR intervention OR policy OR framing OR heuristic OR \"self-control\" OR willpower OR attention OR salience OR \"opt-in\" OR \"opt-out\" OR commitment OR consistency OR reciprocity OR authority OR scarcity OR \"social proof\""
