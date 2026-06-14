"""
Multi-source scraper for behavioural science papers with MENA focus.
Targets: arXiv, PubMed, Semantic Scholar, CrossRef, OpenAlex, bioRxiv, PsyArXiv, OSF, ResearchGate, LibGen, Anna's Archive.
Covers: behavioural models (COM-B, TPB, HBM, SCT, SDT, etc.),
          Arabic/MENA regional terms, broad behavioural science, grey literature.
"""
import requests
import xml.etree.ElementTree as ET
import time
import os
import json
import re
import argparse
import shutil
from pathlib import Path
from collections import Counter

try:
    from grey_sources import libgen_search, annas_archive_search
    HAS_GREY_SOURCES = True
except ImportError:
    HAS_GREY_SOURCES = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ── Query banks ────────────────────────────────────────────────────────────

# Behavioural models and frameworks — the "huge net"
BEHAVIOURAL_MODEL_QUERIES = {
    # Core behavioural science
    "com_b": 'all:"COM-B" OR all:"capability opportunity motivation" OR all:"behaviour change wheel" OR all:"Michie" AND all:"behaviour"',
    "tpb": 'all:"theory of planned behavior" OR all:"theory of planned behaviour" OR all:"Ajzen" AND all:"intention" OR all:"attitude"',
    "hbm": 'all:"health belief model" OR all:"Rosenstock" OR all:"perceived susceptibility" OR all:"perceived severity"',
    "sct": 'all:"social cognitive theory" OR all:"Bandura" AND all:"self-efficacy" OR all:"observational learning"',
    "sdt": 'all:"self-determination theory" OR all:"Deci" OR all:"Ryan" AND all:"intrinsic motivation" OR all:"autonomy"',
    "transtheoretical": 'all:"transtheoretical model" OR all:"stages of change" OR all:"Prochaska" OR all:"transtheoretical"',
    "bmi": 'all:"behavioural insights" OR all:"behavioural economics" OR all:"Thaler" OR all:"Sunstein" OR all:"nudge" OR all:"libertarian paternalism"',
    "dual_process": 'all:"dual process" OR all:"System 1" OR all:"System 2" OR all:"Kahneman" OR all:"heuristics and biases" OR all:"thinking fast"',
    "social_norms": 'all:"social norms" OR all:"Cialdini" OR all:"descriptive norms" OR all:"injunctive norms" OR all:"normative influence"',
    "habit_formation": 'all:"habit formation" OR all:"habit loop" OR all:"Wood" OR all:"Neal" OR all:"automaticity" OR all:"implementation intention"',
    "goal_setting": 'all:"goal setting" OR all:"Locke" OR all:"Latham" OR all:"goal commitment" OR all:"self-regulation"',
    "risk_perception": 'all:"risk perception" OR all:"Slovic" OR all:"risk communication" OR all:"health risk" OR all:"risk behaviour"',
    "technology_acceptance": 'all:"technology acceptance" OR all:"TAM" OR all:"Davis" OR all:"UTAUT" OR all:"perceived usefulness" OR all:"digital adoption"',
    "behaviour_change": 'all:"behaviour change" OR all:"behavior change" OR all:"intervention" OR all:"behavioural intervention" OR all:"health behaviour" OR all:"lifestyle change"',
    "decision_making": 'all:"decision making" OR all:"judgment" OR all:"decision science" OR all:"choice architecture" OR all:"decision support"',
    "cognitive_bias": 'all:"cognitive bias" OR all:"decision bias" OR all:"anchoring" OR all:"framing effect" OR all:"confirmation bias" OR all:"availability heuristic"',
    "motivation": 'all:"motivation" OR all:"intrinsic motivation" OR all:"extrinsic motivation" OR all:"incentive" OR all:"reward" OR all:"reinforcement"',
    "trust_cooperation": 'all:"trust" OR all:"cooperation" OR all:"social dilemma" OR all:"public goods" OR all:"reciprocity" OR all:"altruism" OR all:"prosocial"',
    "emotion_regulation": 'all:"emotion regulation" OR all:"affect" OR all:"emotional intelligence" OR all:"Gross" OR all:"reappraisal" OR all:"suppression"',
    "learning_behaviour": 'all:"learning" OR all:"reinforcement learning" OR all:"operant conditioning" OR all:"classical conditioning" OR all:"observational learning"',
    "addiction_behaviour": 'all:"addiction" OR all:"substance use" OR all:"smoking cessation" OR all:"alcohol" OR all:"drug use" OR all:"behavioural addiction" OR all:"gambling"',
    "health_psychology": 'all:"health psychology" OR all:"health behaviour" OR all:"patient adherence" OR all:"compliance" OR all:"self-management" OR all:"chronic disease"',
    "environmental_behaviour": 'all:"environmental behaviour" OR all:"pro-environmental" OR all:"sustainability" OR all:"energy conservation" OR all:"recycling" OR all:"climate behaviour"',
    "financial_behaviour": 'all:"financial behaviour" OR all:"financial decision" OR all:"savings behaviour" OR all:"investment behaviour" OR all:"consumer behaviour" OR all:"spending"',
    "organisational_behaviour": 'all:"organisational behaviour" OR all:"workplace" OR all:"employee motivation" OR all:"leadership" OR all:"team performance" OR all:"job satisfaction"',
    "cultural_psychology": 'all:"cultural psychology" OR all:"cross-cultural" OR all:"individualism" OR all:"collectivism" OR all:"Hofstede" OR all:"Markus" OR all:"Kitayama"',
    "developmental_behaviour": 'all:"developmental" OR all:"child development" OR all:"adolescent behaviour" OR all:"parenting" OR all:"attachment" OR all:"Piaget" OR all:"Vygotsky"',
    "social_influence": 'all:"social influence" OR all:"conformity" OR all:"obedience" OR all:"Milgram" OR all:"Asch" OR all:"peer influence" OR all:"group dynamics"',
    "persuasion": 'all:"persuasion" OR all:"persuasive technology" OR all:"Fogg" OR all:"cialdini" OR all:"influence" OR all:"attitude change" OR all:"message framing"',
    "wellbeing": 'all:"wellbeing" OR all:"well-being" OR all:"life satisfaction" OR all:"positive psychology" OR all:"Seligman" OR all:"PERMA" OR all:"flourishing" OR all:"happiness"',
    "mental_health": 'all:"mental health" OR all:"depression" OR all:"anxiety" OR all:"stress" OR all:"burnout" OR all:"psychological distress" OR all:"CBT" OR all:"cognitive behavioural therapy"',
}

# MENA / Arabic regional queries
MENA_QUERIES = {
    "mena_broad": 'all:"Middle East" OR all:"North Africa" OR all:"MENA" OR all:"Arab world" OR all:"Gulf" OR all:"Arabian Peninsula"',
    "saudi": 'all:"Saudi Arabia" OR all:"Saudi" OR all:"Riyadh" OR all:"Jeddah" OR all:"Vision 2030" OR all:"NEOM" OR all:"KSA"',
    "uae": 'all:"UAE" OR all:"United Arab Emirates" OR all:"Emirati" OR all:"Dubai" OR all:"Abu Dhabi"',
    "qatar": 'all:"Qatar" OR all:"Doha" OR all:"Qatari"',
    "egypt": 'all:"Egypt" OR all:"Cairo" OR all:"Egyptian"',
    "jordan": 'all:"Jordan" OR all:"Amman" OR all:"Jordanian"',
    "lebanon": 'all:"Lebanon" OR all:"Beirut" OR all:"Lebanese"',
    "iraq": 'all:"Iraq" OR all:"Baghdad" OR all:"Iraqi"',
    "iran": 'all:"Iran" OR all:"Tehran" OR all:"Iranian" OR all:"Persian"',
    "kuwait_bahrain_oman": 'all:"Kuwait" OR all:"Bahrain" OR all:"Oman" OR all:"Muscat" OR all:"Kuwaiti" OR all:"Bahraini"',
    "morocco_tunisia_algeria": 'all:"Morocco" OR all:"Tunisia" OR all:"Algeria" OR all:"Maghreb" OR all:"Moroccan" OR all:"Tunisian"',
    "palestine_israel": 'all:"Palestine" OR all:"Gaza" OR all:"West Bank" OR all:"Palestinian" OR all:"Israel" OR all:"Israeli"',
    "arab_culture": 'all:"Arab" AND (all:"culture" OR all:"cultural" OR all:"society" OR all:"social") OR all:"Arabic" AND (all:"behaviour" OR all:"behavior") OR all:"Islamic" AND (all:"psychology" OR all:"behaviour") OR all:"Muslim" AND (all:"health" OR all:"behaviour")',
    "mena_health": 'all:"Middle East" AND (all:"health" OR all:"healthcare" OR all:"public health") OR all:"Arab" AND (all:"health" OR all:"mental health") OR all:"Gulf" AND (all:"health" OR all:"disease")',
    "mena_education": 'all:"Middle East" AND (all:"education" OR all:"student" OR all:"university" OR all:"learning") OR all:"Arab" AND (all:"education" OR all:"school")',
    "mena_business": 'all:"Middle East" AND (all:"business" OR all:"management" OR all:"entrepreneurship" OR all:"economy") OR all:"Gulf" AND (all:"business" OR all:"oil" OR all:"diversification")',
    "mena_technology": 'all:"Middle East" AND (all:"technology" OR all:"digital" OR all:"internet" OR all:"social media" OR all:"AI") OR all:"Arab" AND (all:"technology" OR all:"digital")',
    "mena_youth": 'all:"Middle East" AND (all:"youth" OR all:"young" OR all:"adolescent" OR all:"generation") OR all:"Arab" AND (all:"youth" OR all:"young people")',
    "mena_women": 'all:"Middle East" AND (all:"women" OR all:"gender" OR all:"female" OR all:"woman") OR all:"Arab" AND (all:"women" OR all:"gender") OR all:"Gulf" AND (all:"women") OR all:"hijab" OR all:"veil"',
    "mena_migration": 'all:"Middle East" AND (all:"migration" OR all:"refugee" OR all:"displacement" OR all:"expatriate" OR all:"migrant") OR all:"Arab" AND (all:"refugee" OR all:"migration")',
    "arabic_transliterated": 'all:"Saudia" OR all:"Emirati" OR all:"Qatari" OR all:"Kuwaiti" OR all:"Bahraini" OR all:"Omani" OR all:"Yemeni" OR all:"Syrian" OR all:"Lebanese" OR all:"Jordanian" OR all:"Palestinian" OR all:"Iraqi" OR all:"Iranian" OR all:"Egyptian" OR all:"Libyan" OR all:"Tunisian" OR all:"Algerian" OR all:"Moroccan" OR all:"Sudanese" OR all:"Mauritanian"',
}

# Public health and clinical behaviour
PUBLIC_HEALTH_QUERIES = {
    "vaccination_behaviour": 'all:"vaccination" OR all:"vaccine hesitancy" OR all:"immunisation" OR all:"immunization" OR all:"COVID-19 vaccine"',
    "adherence_self_management": 'all:"medication adherence" OR all:"patient adherence" OR all:"self-management" OR all:"chronic disease" OR all:"diabetes"',
    "obesity_physical_activity": 'all:"obesity" OR all:"physical activity" OR all:"exercise behaviour" OR all:"sedentary behaviour" OR all:"diet"',
    "smoking_substance_use": 'all:"smoking cessation" OR all:"tobacco" OR all:"substance use" OR all:"alcohol" OR all:"addiction"',
    "mental_health_help_seeking": 'all:"mental health" OR all:"help-seeking" OR all:"depression" OR all:"anxiety" OR all:"stigma" OR all:"psychological distress"',
    "maternal_child_health": 'all:"maternal health" OR all:"child health" OR all:"breastfeeding" OR all:"parenting" OR all:"adolescent health"',
    "screening_prevention": 'all:"screening" OR all:"cancer screening" OR all:"preventive health" OR all:"risk perception" OR all:"health literacy"',
    "telehealth_digital_health": 'all:"telehealth" OR all:"mHealth" OR all:"mobile health" OR all:"eHealth" OR all:"digital health intervention"',
}

# Digital, technology and AI-mediated behaviour
DIGITAL_BEHAVIOUR_QUERIES = {
    "technology_adoption": 'all:"technology acceptance" OR all:"UTAUT" OR all:"perceived usefulness" OR all:"digital adoption"',
    "social_media_behaviour": 'all:"social media" OR all:"Facebook" OR all:"Instagram" OR all:"TikTok" OR all:"online behaviour"',
    "ai_human_behaviour": 'all:"artificial intelligence" OR all:"AI" OR all:"chatbot" OR all:"automation" OR all:"trust"',
    "internet_addiction": 'all:"internet addiction" OR all:"problematic internet use" OR all:"gaming disorder" OR all:"screen time"',
    "cybersecurity_behaviour": 'all:"cybersecurity" OR all:"privacy" OR all:"security behaviour" OR all:"phishing" OR all:"password"',
    "online_learning": 'all:"online learning" OR all:"e-learning" OR all:"MOOC" OR all:"learning analytics" OR all:"student engagement"',
}

# Education, learning and youth behaviour
EDUCATION_YOUTH_QUERIES = {
    "student_motivation": 'all:"student motivation" OR all:"academic motivation" OR all:"self-efficacy" OR all:"achievement goals"',
    "academic_procrastination": 'all:"academic procrastination" OR all:"self-regulated learning" OR all:"study habits"',
    "school_health_behaviour": 'all:"school" OR all:"adolescent" OR all:"health education" OR all:"peer influence"',
    "teacher_behaviour": 'all:"teacher" OR all:"educator" OR all:"classroom" OR all:"instructional behaviour"',
    "youth_risk_behaviour": 'all:"youth" OR all:"adolescent" OR all:"risk behaviour" OR all:"substance use" OR all:"violence"',
    "parenting_development": 'all:"parenting" OR all:"child development" OR all:"attachment" OR all:"family behaviour"',
}

# Economic, consumer and organisational behaviour
ECONOMIC_ORGANISATIONAL_QUERIES = {
    "consumer_behaviour": 'all:"consumer behaviour" OR all:"consumer behavior" OR all:"purchase intention" OR all:"brand attitude"',
    "financial_behaviour": 'all:"financial literacy" OR all:"saving behaviour" OR all:"investment behaviour" OR all:"spending" OR all:"debt"',
    "entrepreneurship": 'all:"entrepreneurship" OR all:"startup" OR all:"innovation adoption" OR all:"business behaviour"',
    "workplace_behaviour": 'all:"workplace" OR all:"employee" OR all:"job satisfaction" OR all:"turnover intention" OR all:"performance"',
    "leadership_teams": 'all:"leadership" OR all:"team" OR all:"psychological safety" OR all:"organisational commitment"',
    "burnout_wellbeing": 'all:"burnout" OR all:"work engagement" OR all:"wellbeing" OR all:"stress" OR all:"resilience"',
}

# Environmental, urban and sustainability behaviour
ENVIRONMENTAL_BEHAVIOUR_QUERIES = {
    "pro_environmental": 'all:"pro-environmental behaviour" OR all:"environmental behaviour" OR all:"sustainability"',
    "recycling_waste": 'all:"recycling" OR all:"waste sorting" OR all:"waste reduction" OR all:"circular economy"',
    "energy_water_conservation": 'all:"energy conservation" OR all:"water conservation" OR all:"resource conservation"',
    "transport_mobility": 'all:"transport behaviour" OR all:"mobility" OR all:"public transport" OR all:"active travel"',
    "climate_risk": 'all:"climate change" OR all:"climate risk" OR all:"adaptation" OR all:"risk perception"',
    "urban_behaviour": 'all:"urban" OR all:"smart city" OR all:"built environment" OR all:"walkability"',
}

# Culture, gender, migration and social context
CULTURE_GENDER_YOUTH_QUERIES = {
    "collectivism_individualism": 'all:"collectivism" OR all:"individualism" OR all:"cultural values" OR all:"social identity"',
    "religion_behaviour": 'all:"religion" OR all:"religious" OR all:"Muslim" OR all:"Islamic" OR all:"faith" OR all:"behaviour"',
    "gender_women": 'all:"gender" OR all:"women" OR all:"female" OR all:"gender norms" OR all:"empowerment"',
    "migration_refugee": 'all:"migration" OR all:"refugee" OR all:"displacement" OR all:"expatriate" OR all:"migrant"',
    "stigma_discrimination": 'all:"stigma" OR all:"discrimination" OR all:"social exclusion" OR all:"minority"',
    "honour_shame": 'all:"honour" OR all:"honor" OR all:"shame" OR all:"family reputation" OR all:"social norms"',
}

# Arabic/MENA scripts and transliterations
ARABIC_MENA_TERM_QUERIES = {
    "arabic_script_terms": 'all:"العربية" OR all:"سلوك" OR all:"الصحة" OR all:"التعليم" OR all:"السعودية" OR all:"مصر"',
    "arabic_transliteration": 'all:"Saudi" OR all:"Saudia" OR all:"Emirati" OR all:"Qatari" OR all:"Kuwaiti" OR all:"Bahraini" OR all:"Omani" OR all:"Yemeni" OR all:"Syrian" OR all:"Lebanese" OR all:"Jordanian" OR all:"Palestinian" OR all:"Iraqi" OR all:"Iranian" OR all:"Egyptian" OR all:"Libyan" OR all:"Tunisian" OR all:"Algerian" OR all:"Moroccan" OR all:"Sudanese"',
    "gulf_terms": 'all:"Gulf" OR all:"GCC" OR all:"Saudi Arabia" OR all:"UAE" OR all:"Qatar" OR all:"Kuwait" OR all:"Bahrain" OR all:"Oman"',
    "levant_terms": 'all:"Levant" OR all:"Jordan" OR all:"Lebanon" OR all:"Syria" OR all:"Palestine" OR all:"West Bank" OR all:"Gaza"',
    "north_africa_terms": 'all:"North Africa" OR all:"Maghreb" OR all:"Morocco" OR all:"Tunisia" OR all:"Algeria" OR all:"Libya" OR all:"Egypt"',
}

# Combined: behavioural models × MENA (for targeted searches)
COMBINED_QUERIES = {
    "mena_nudge": 'all:"nudge" AND (all:"Middle East" OR all:"Arab" OR all:"Saudi" OR all:"UAE" OR all:"Gulf")',
    "mena_health_behaviour": 'all:"health behaviour" AND (all:"Middle East" OR all:"Arab" OR all:"Saudi" OR all:"Egypt")',
    "mena_com_b": 'all:"COM-B" OR all:"behaviour change wheel" AND (all:"health" OR all:"intervention")',
    "mena_tpb": 'all:"theory of planned behavior" AND (all:"health" OR all:"Arab" OR all:"Middle East")',
    "mena_mental_health": 'all:"mental health" AND (all:"Middle East" OR all:"Arab" OR all:"Saudi" OR all:"refugee")',
    "mena_digital_behaviour": 'all:"digital" OR all:"social media" OR all:"internet" AND (all:"Arab" OR all:"Middle East" OR all:"youth")',
    "mena_consumer": 'all:"consumer behaviour" OR all:"consumer behavior" AND (all:"Middle East" OR all:"Arab" OR all:"Gulf")',
    "mena_organisational": 'all:"organisational behaviour" OR all:"workplace" AND (all:"Middle East" OR all:"Arab" OR all:"Gulf")',
}

ALL_QUERY_GROUPS = {
    "Behavioural models": BEHAVIOURAL_MODEL_QUERIES,
    "MENA and Arabic terms": MENA_QUERIES,
    "Public health behaviour": PUBLIC_HEALTH_QUERIES,
    "Digital and AI behaviour": DIGITAL_BEHAVIOUR_QUERIES,
    "Education and youth behaviour": EDUCATION_YOUTH_QUERIES,
    "Economic and organisational behaviour": ECONOMIC_ORGANISATIONAL_QUERIES,
    "Environmental and urban behaviour": ENVIRONMENTAL_BEHAVIOUR_QUERIES,
    "Culture, gender and migration": CULTURE_GENDER_YOUTH_QUERIES,
    "Arabic and MENA scripts": ARABIC_MENA_TERM_QUERIES,
    "Combined behavioural x MENA": COMBINED_QUERIES,
}

ALL_QUERIES = {}
for _group in ALL_QUERY_GROUPS.values():
    ALL_QUERIES.update(_group)


def load_existing_ids(data_dir):
    """Load all existing paper IDs from data/raw/papers_*.json files."""
    ids = set()
    for f in Path(data_dir).glob("papers_*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    for p in data:
                        pid = p.get("id") or p.get("entry_id")
                        if pid:
                            ids.add(pid)
                        doi = p.get("doi") or p.get("DOI")
                        if doi:
                            doi_norm = doi.lower().strip().replace("https://doi.org/", "").replace("doi:", "").strip()
                            ids.add(f"DOI:{doi_norm}")
        except Exception:
            pass
    return ids


class MultiSourceScraper:
    def __init__(self, delay=3):
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Hermes-Agent/1.0 (research; contact: academic)"
        })

    def _pbar(self, total, desc):
        if HAS_TQDM:
            return tqdm(total=total, desc=desc)
        else:
            return _SimplePbar(total, desc)

    # ── arXiv ─────────────────────────────────────────────────────────────
    def search_arxiv(self, query, max_results=100):
        papers = []
        start = 0
        pbar = self._pbar(min(max_results, 500), f"arXiv:{query[:20]}")
        while len(papers) < max_results:
            params = {
                'search_query': query, 'start': start,
                'max_results': min(100, max_results - len(papers)),
                'sortBy': 'submittedDate', 'sortOrder': 'descending'
            }
            try:
                r = self.session.get("http://export.arxiv.org/api/query", params=params, timeout=30)
                time.sleep(self.delay)
                root = ET.fromstring(r.text)
            except Exception as e:
                print(f"  arXiv error: {e}")
                break
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('{%s}entry' % ns['atom'])
            if not entries:
                break
            for entry in entries:
                eid = entry.find('{%s}id' % ns['atom']).text
                arxiv_id = eid.split('/')[-1]
                title = entry.find('{%s}title' % ns['atom']).text.strip().replace('\n', ' ')
                authors = [a.find('{%s}name' % ns['atom']).text for a in entry.findall('{%s}author' % ns['atom'])]
                summary = entry.find('{%s}summary' % ns['atom']).text.strip().replace('\n', ' ')
                published = entry.find('{%s}published' % ns['atom']).text
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                papers.append({
                    'id': arxiv_id, 'title': title, 'authors': authors,
                    'summary': summary, 'published': published,
                    'pdf_url': pdf_url, 'source': 'arXiv'
                })
                pbar.update(1)
                if len(papers) >= max_results:
                    break
            start += len(entries)
        pbar.close()
        return papers

    # ── PubMed ────────────────────────────────────────────────────────────
    def search_pubmed(self, query, max_results=100):
        papers = []
        pbar = self._pbar(min(max_results, 200), f"PubMed:{query[:20]}")
        try:
            # Search
            r = self.session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={'db': 'pubmed', 'retmode': 'json', 'retmax': max_results, 'term': query},
                timeout=30
            )
            ids = r.json().get('esearchresult', {}).get('idlist', [])
            if not ids:
                pbar.close()
                return []
            time.sleep(0.5)
            # Fetch
            r2 = self.session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={'db': 'pubmed', 'retmode': 'xml', 'id': ','.join(ids[:max_results])},
                timeout=30
            )
            root = ET.fromstring(r2.text)
            for article in root.findall('.//MedlineCitation'):
                pmid = article.find('PMID').text
                at = article.find('.//ArticleTitle')
                title = at.text if at is not None else "No title"
                abstract_el = article.find('.//Abstract/AbstractText')
                summary = abstract_el.text if abstract_el is not None else ""
                journal_el = article.find('.//Journal/Title')
                journal = journal_el.text if journal_el is not None else ""
                year_el = article.find('.//Journal/JournalIssue/PubDate/Year')
                if year_el is None:
                    year_el = article.find('.//ArticleDate/Year')
                year = year_el.text if year_el is not None else "2024"
                authors = []
                for a in article.findall('.//Author'):
                    ln = a.find('LastName')
                    fn = a.find('ForeName')
                    if ln is not None:
                        authors.append(f"{ln.text} {fn.text if fn is not None else ''}".strip())
                papers.append({
                    'id': f"PMID:{pmid}", 'title': title, 'authors': authors,
                    'summary': summary, 'published': f"{year}-01-01",
                    'pdf_url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    'source': 'PubMed', 'journal': journal
                })
                pbar.update(1)
        except Exception as e:
            print(f"  PubMed error: {e}")
        pbar.close()
        return papers

    # ── PubMed Central (PMC) ──────────────────────────────────────────────
    def search_pubmedcentral(self, query, max_results=100):
        """Search PubMed Central for full-text OA papers via E-utilities."""
        papers = []
        pbar = self._pbar(min(max_results, 200), f"PMC:{query[:20]}")
        try:
            r = self.session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={
                    'db': 'pmc', 'retmode': 'json', 'retmax': max_results,
                    'term': query + ' AND "open access"[filter]',
                    'sort': 'date'
                },
                timeout=30
            )
            ids = r.json().get('esearchresult', {}).get('idlist', [])
            if not ids:
                pbar.close()
                return []
            time.sleep(0.5)
            r2 = self.session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={'db': 'pmc', 'retmode': 'json', 'id': ','.join(ids[:max_results])},
                timeout=30
            )
            summaries = r2.json().get('result', {})
            for pmcid in ids[:max_results]:
                if pmcid not in summaries:
                    continue
                s = summaries[pmcid]
                title = s.get('title', 'No title')
                authors = [a.get('name', '') for a in s.get('authors', [])]
                abstract = s.get('abstract', '') or ""
                pub_date = s.get('pubdate', s.get('sortpubdate', ''))
                pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/pdf/"
                papers.append({
                    'id': f"PMC:{pmcid}",
                    'title': title,
                    'authors': authors,
                    'summary': abstract,
                    'published': pub_date,
                    'pdf_url': pdf_url,
                    'pdf_source': 'pmc',
                    'source': 'PubMedCentral',
                    'journal': s.get('fulljournalname', s.get('source', '')),
                    'pmcid': pmcid,
                })
                pbar.update(1)
        except Exception as e:
            print(f"  PMC error: {e}")
        pbar.close()
        return papers

    # ── Semantic Scholar ──────────────────────────────────────────────────
    def search_semanticscholar(self, query, max_results=100):
        papers = []
        pbar = self._pbar(min(max_results, 100), f"SemanticScholar:{query[:20]}")
        try:
            r = self.session.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={'query': query, 'limit': min(max_results, 100),
                        'fields': 'title,authors,abstract,url,publicationDate,year,venue,openAccessPdf'},
                timeout=30
            )
            if r.status_code == 200:
                data = r.json()
                for p in data.get('data', []):
                    paper_dict = {
                        'id': p.get('paperId', 'unknown'),
                        'title': p.get('title', ''),
                        'authors': [a.get('name', '') for a in p.get('authors', [])],
                        'summary': p.get('abstract', '') or '',
                        'published': p.get('publicationDate', '') or f"{p.get('year', 2024)}-01-01",
                        'pdf_url': p.get('url', '') or '',
                        'source': 'SemanticScholar',
                        'venue': p.get('venue', '')
                    }
                    oa_pdf = p.get('openAccessPdf')
                    if oa_pdf and oa_pdf.get('url'):
                        paper_dict['pdf_url'] = oa_pdf['url']
                        paper_dict['pdf_source'] = 'semanticscholar_oa'
                    papers.append(paper_dict)
                    pbar.update(1)
        except Exception as e:
            print(f"  SS error: {e}")
        pbar.close()
        return papers

    # ── CrossRef ──────────────────────────────────────────────────────────
    def search_crossref(self, query, max_results=100):
        papers = []
        pbar = self._pbar(min(max_results, 100), f"CrossRef:{query[:20]}")
        try:
            r = self.session.get(
                "https://api.crossref.org/works",
                params={'query': query, 'rows': min(max_results, 100), 'sort': 'published', 'order': 'desc',
                        'filter': 'type:journal-article'},
                timeout=30
            )
            if r.status_code == 200:
                items = r.json().get('message', {}).get('items', [])
                for item in items:
                    title_list = item.get('title', [])
                    title = title_list[0] if title_list else "No title"
                    authors = []
                    for a in item.get('author', []):
                        name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                        if name:
                            authors.append(name)
                    abstract = item.get('abstract', '') or ""
                    abstract = re.sub(r'<[^>]+>', '', abstract)
                    pub_date = item.get('published-print', item.get('published-online', {}))
                    date_parts = pub_date.get('date-parts', [[2024]])
                    year = date_parts[0][0] if date_parts else 2024
                    doi = item.get('DOI', '')
                    papers.append({
                        'id': f"DOI:{doi}", 'title': title, 'authors': authors,
                        'summary': abstract, 'published': f"{year}-01-01",
                        'pdf_url': f"https://doi.org/{doi}",
                        'source': 'CrossRef',
                        'journal': item.get('container-title', [''])[0] if item.get('container-title') else ''
                    })
                    pbar.update(1)
        except Exception as e:
            print(f"  CrossRef error: {e}")
        pbar.close()
        return papers

    # ── OpenAlex ───────────────────────────────────────────────────────────
    def search_openalex(self, query, max_results=100):
        papers = []
        pbar = self._pbar(min(max_results, 200), f"OpenAlex:{query[:20]}")
        try:
            r = self.session.get(
                "https://api.openalex.org/works",
                params={
                    'search': query,
                    'per-page': min(max_results, 200),
                    'sort': 'publication_date:desc',
                    'select': 'id,doi,title,authorships,publication_year,publication_date,abstract_inverted_index,primary_location,locations,host_venue,type,concepts'
                },
                timeout=30
            )
            if r.status_code == 200:
                items = r.json().get('results', [])
                for item in items:
                    doi = item.get('doi') or ''
                    if doi:
                        doi = doi.replace('https://doi.org/', '')
                    title = item.get('title') or 'No title'
                    authors = []
                    for a in item.get('authorships', [])[:20]:
                        author = a.get('author', {})
                        name = author.get('display_name') or ''
                        if name:
                            authors.append(name)
                    abstract = self._openalex_abstract(item.get('abstract_inverted_index'))
                    pub_date = item.get('publication_date') or f"{item.get('publication_year') or 2024}-01-01"
                    pdf_url = ''
                    if item.get('primary_location'):
                        pdf_url = item['primary_location'].get('landing_page_url') or ''
                    if not pdf_url and item.get('locations'):
                        for loc in item.get('locations', []):
                            if loc.get('pdf_url'):
                                pdf_url = loc.get('pdf_url')
                                break
                    papers.append({
                        'id': f"OpenAlex:{item.get('id', '').split('/')[-1]}:{doi}" if doi else f"OpenAlex:{item.get('id', '').split('/')[-1]}",
                        'title': title, 'authors': authors,
                        'summary': abstract, 'published': pub_date,
                        'pdf_url': pdf_url,
                        'source': 'OpenAlex',
                        'journal': item.get('host_venue', {}).get('display_name') if item.get('host_venue') else '',
                        'type': item.get('type', '')
                    })
                    pbar.update(1)
        except Exception as e:
            print(f"  OpenAlex error: {e}")
        pbar.close()
        return papers

    # ── bioRxiv ─────────────────────────────────────────────────────────────
    def search_biorxiv(self, query, max_results=100):
        """Search bioRxiv via their API."""
        papers = []
        pbar = self._pbar(min(max_results, 100), f"bioRxiv:{query[:20]}")
        try:
            r = self.session.get(
                "https://api.biorxiv.org/details/biorxiv/0/9999",
                params={'q': query, 'num_results': min(max_results, 100)},
                timeout=30
            )
            if r.status_code == 200:
                data = r.json()
                for item in data.get('collection', [])[:max_results]:
                    papers.append({
                        'id': f"bioRxiv:{item.get('doi', '')}",
                        'title': item.get('title', ''),
                        'authors': [a.strip() for a in item.get('authors', '').split(';') if a.strip()],
                        'summary': item.get('abstract', ''),
                        'published': item.get('date', ''),
                        'pdf_url': f"https://www.biorxiv.org/content/{item.get('doi', '')}.full.pdf" if item.get('doi') else '',
                        'pdf_source': 'biorxiv',
                        'source': 'bioRxiv',
                        'doi': item.get('doi', ''),
                    })
                    pbar.update(1)
        except Exception as e:
            print(f"  bioRxiv error: {e}")
        pbar.close()
        return papers

    # ── PsyArXiv ────────────────────────────────────────────────────────────
    def search_psyarxiv(self, query, max_results=100):
        """Search PsyArXiv via OSF API."""
        papers = []
        pbar = self._pbar(min(max_results, 100), f"PsyArXiv:{query[:20]}")
        try:
            r = self.session.get(
                "https://api.osf.io/v2/preprints/",
                params={
                    'filter[provider]': 'psyarxiv',
                    'page[size]': min(max_results, 100),
                    'filter[title]': query,
                },
                timeout=30
            )
            if r.status_code == 200:
                data = r.json()
                for item in data.get('data', [])[:max_results]:
                    attrs = item.get('attributes', {})
                    papers.append({
                        'id': f"PsyArXiv:{item.get('id', '')}",
                        'title': attrs.get('title', ''),
                        'authors': [],
                        'summary': attrs.get('description', ''),
                        'published': attrs.get('date_created', ''),
                        'pdf_url': attrs.get('links', {}).get('download', '') if attrs.get('links') else '',
                        'pdf_source': 'psyarxiv',
                        'source': 'PsyArXiv',
                    })
                    pbar.update(1)
        except Exception as e:
            print(f"  PsyArXiv error: {e}")
        pbar.close()
        return papers

    # ── OSF Preprints ───────────────────────────────────────────────────────
    def search_osf(self, query, max_results=100):
        """Search OSF Preprints (all providers)."""
        papers = []
        pbar = self._pbar(min(max_results, 100), f"OSF:{query[:20]}")
        try:
            r = self.session.get(
                "https://api.osf.io/v2/preprints/",
                params={
                    'page[size]': min(max_results, 100),
                    'filter[title]': query,
                },
                timeout=30
            )
            if r.status_code == 200:
                data = r.json()
                for item in data.get('data', [])[:max_results]:
                    attrs = item.get('attributes', {})
                    papers.append({
                        'id': f"OSF:{item.get('id', '')}",
                        'title': attrs.get('title', ''),
                        'authors': [],
                        'summary': attrs.get('description', ''),
                        'published': attrs.get('date_created', ''),
                        'pdf_url': attrs.get('links', {}).get('download', '') if attrs.get('links') else '',
                        'pdf_source': 'osf',
                        'source': 'OSF',
                    })
                    pbar.update(1)
        except Exception as e:
            print(f"  OSF error: {e}")
        pbar.close()
        return papers

    # ── ResearchGate ────────────────────────────────────────────────────────
    def search_researchgate(self, query, max_results=50):
        """Search ResearchGate for papers (author-uploaded PDFs)."""
        papers = []
        pbar = self._pbar(min(max_results, 50), f"ResearchGate:{query[:20]}")
        try:
            r = self.session.get(
                "https://www.researchgate.net/search/publication",
                params={'q': query, 'type': 'publication'},
                timeout=30,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml',
                }
            )
            if r.status_code == 200:
                html = r.text
                pub_pattern = re.compile(
                    r'"publicationType":"([^"]+)".*?"title":"([^"]+)".*?"doi":"([^"]*)".*?"url":"([^"]+)"',
                    re.DOTALL
                )
                for m in pub_pattern.finditer(html):
                    pub_type, title, doi, url = m.groups()
                    papers.append({
                        'id': f"RG:{doi or url.split('/')[-1]}",
                        'title': title.replace('\\n', ' '),
                        'authors': [],
                        'summary': '',
                        'published': '',
                        'pdf_url': url if url.endswith('.pdf') else '',
                        'source': 'ResearchGate',
                        'doi': doi,
                    })
                    pbar.update(1)
                    if len(papers) >= max_results:
                        break
        except Exception as e:
            print(f"  ResearchGate error: {e}")
        pbar.close()
        return papers

    # ── LibGen (grey literature) ────────────────────────────────────────────────
    def search_libgen(self, query, max_results=50):
        """Search Library Genesis for behavioural science books and papers."""
        if not HAS_GREY_SOURCES:
            return []
        papers = []
        pbar = self._pbar(min(max_results, 50), f"LibGen:{query[:20]}")
        try:
            results = libgen_search(query, max_results)
            for item in results:
                papers.append({
                    'id': f"LibGen:{item.get('md5', '')[:16]}",
                    'title': item.get('title', ''),
                    'authors': item.get('authors', '').split(';') if item.get('authors') else [],
                    'summary': '',
                    'published': '',
                    'pdf_url': item.get('download_url', ''),
                    'pdf_source': 'libgen',
                    'source': 'LibGen',
                    'md5': item.get('md5', ''),
                })
                pbar.update(1)
        except Exception as e:
            print(f"  LibGen error: {e}")
        pbar.close()
        return papers

    # ── Anna's Archive (grey literature) ───────────────────────────────────────
    def search_annas_archive(self, query, max_results=50):
        """Search Anna's Archive for academic papers and books."""
        if not HAS_GREY_SOURCES:
            return []
        papers = []
        pbar = self._pbar(min(max_results, 50), f"Anna:{query[:20]}")
        try:
            results = annas_archive_search(query, max_results)
            for item in results:
                papers.append({
                    'id': f"Anna:{item.get('md5', '')[:16]}",
                    'title': item.get('title', ''),
                    'authors': [],
                    'summary': '',
                    'published': '',
                    'pdf_url': item.get('download_url', ''),
                    'pdf_source': 'annas',
                    'source': 'AnnaArchive',
                    'md5': item.get('md5', ''),
                })
                pbar.update(1)
        except Exception as e:
            print(f"  Anna's Archive error: {e}")
        pbar.close()
        return papers

    def _openalex_abstract(self, inverted):
        if not inverted:
            return ''
        try:
            words = {}
            for word, positions in inverted.items():
                for pos in positions:
                    words[int(pos)] = word
            return ' '.join(words[i] for i in sorted(words.keys()))
        except Exception:
            return ''

    # ── Dedup + save ──────────────────────────────────────────────────────
    def dedup(self, papers):
        """Dedup by DOI first, then title normalization."""
        seen_dois = set()
        seen_titles = {}
        unique = []
        for p in papers:
            doi = p.get("doi") or p.get("DOI") or ""
            if doi:
                doi_norm = doi.lower().strip().replace("https://doi.org/", "").replace("doi:", "").strip()
                if doi_norm in seen_dois:
                    continue
                seen_dois.add(doi_norm)
            key = re.sub(r'[^\w]', '', p['title'].lower())[:100]
            if key and key in seen_titles:
                continue
            if key:
                seen_titles[key] = True
            unique.append(p)
        return unique

    def save(self, papers, tag):
        ts = int(time.time())
        out = self.data_dir / f"papers_{tag}_{ts}.json"
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        # Also copy as canonical
        canonical = self.data_dir / f"papers_{tag}.json"
        if canonical.exists():
            os.remove(canonical)
        shutil.copy(out, canonical)
        return out

    # ── Run all queries ───────────────────────────────────────────────────
    def run_all(self, queries, sources=None, max_per_query=50, incremental=False):
        """Run a dict of queries across multiple sources."""
        if sources is None:
            sources = ['arxiv', 'pubmed', 'semanticscholar', 'crossref', 'openalex']
        existing_ids = set()
        if incremental:
            existing_ids = load_existing_ids(self.data_dir)
            print(f"  Incremental mode: {len(existing_ids)} existing papers")
        all_papers = []
        for qname, qtext in queries.items():
            print(f"\n[{qname}]")
            batch = []
            if 'arxiv' in sources:
                print(f"  arXiv...")
                batch.extend(self.search_arxiv(qtext, max_per_query))
            if 'pubmed' in sources:
                print(f"  PubMed...")
                batch.extend(self.search_pubmed(qtext, max_per_query))
            if 'pubmedcentral' in sources:
                print(f"  PubMed Central...")
                batch.extend(self.search_pubmedcentral(qtext, max_per_query))
            if 'semanticscholar' in sources:
                print(f"  Semantic Scholar...")
                batch.extend(self.search_semanticscholar(qtext, max_per_query))
            if 'crossref' in sources:
                print(f"  CrossRef...")
                batch.extend(self.search_crossref(qtext, max_per_query))
            if 'openalex' in sources:
                print(f"  OpenAlex...")
                batch.extend(self.search_openalex(qtext, max_per_query))
            if 'biorxiv' in sources:
                print(f"  bioRxiv...")
                batch.extend(self.search_biorxiv(qtext, max_per_query))
            if 'psyarxiv' in sources:
                print(f"  PsyArXiv...")
                batch.extend(self.search_psyarxiv(qtext, max_per_query))
            if 'osf' in sources:
                print(f"  OSF...")
                batch.extend(self.search_osf(qtext, max_per_query))
            if 'researchgate' in sources:
                print(f"  ResearchGate...")
                batch.extend(self.search_researchgate(qtext, max_per_query))
            if 'libgen' in sources:
                print(f"  LibGen...")
                batch.extend(self.search_libgen(qtext, max_per_query))
            if 'annas' in sources:
                print(f"  Anna's Archive...")
                batch.extend(self.search_annas_archive(qtext, max_per_query))
            if incremental:
                filtered = [p for p in batch
                            if (p.get("id") not in existing_ids)
                            and (p.get("entry_id") not in existing_ids)]
                skipped = len(batch) - len(filtered)
                if skipped:
                    print(f"  Skipped {skipped} already-seen papers")
                all_papers.extend(filtered)
            else:
                all_papers.extend(batch)
        return self.dedup(all_papers)


class _SimplePbar:
    def __init__(self, total, desc):
        self.total = total
        self.desc = desc
        self.n = 0
        print(f"  {desc}: 0/{total}", end='\r')
    def update(self, n=1):
        self.n += n
        print(f"  {self.desc}: {self.n}/{self.total}", end='\r')
    def close(self):
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-source behavioural science scraper")
    parser.add_argument('-q', '--query', default=None,
                        help='Query key (e.g. com_b, tpb, mena_broad, saudi, etc.)')
    parser.add_argument('--group', default=None,
                        help='Query group key: behavioural_models, mena, public_health, digital, education, economic, environmental, culture, arabic, combined')
    parser.add_argument('--custom', default=None,
                        help='Custom query string. For arXiv, this is wrapped as all:"custom query".')
    parser.add_argument('--tag', default=None, help='Output tag; defaults to query/group/custom')
    parser.add_argument('--delay', type=float, default=3, help='Delay between arXiv requests')
    parser.add_argument('-n', '--num', type=int, default=50, help='Max results per query')
    parser.add_argument('--sources', default='arxiv,pubmed,pubmedcentral,semanticscholar,crossref,openalex,biorxiv,psyarxiv,osf,researchgate,libgen,annas',
                        help='Comma-separated sources (includes grey: libgen, annas)')
    parser.add_argument('--all-models', action='store_true', help='Run ALL behavioural model queries')
    parser.add_argument('--all-mena', action='store_true', help='Run ALL MENA queries')
    parser.add_argument('--all-combined', action='store_true', help='Run ALL combined queries')
    parser.add_argument('--everything', action='store_true', help='Run ALL queries across ALL sources')
    parser.add_argument('--list', action='store_true', help='List query groups and keys')
    parser.add_argument('--incremental', action='store_true',
                        help='Skip papers already in data/raw/ files')
    args = parser.parse_args()

    scraper = MultiSourceScraper(delay=args.delay)
    sources = [s.strip() for s in args.sources.split(',') if s.strip()]
    all_papers = []
    selected_group = None
    selected_queries = None
    mode_label = ''

    GROUP_ALIASES = {
        'behavioural_models': 'Behavioural models',
        'models': 'Behavioural models',
        'mena': 'MENA and Arabic terms',
        'arabic': 'Arabic and MENA scripts',
        'public_health': 'Public health behaviour',
        'health': 'Public health behaviour',
        'digital': 'Digital and AI behaviour',
        'education': 'Education and youth behaviour',
        'youth': 'Education and youth behaviour',
        'economic': 'Economic and organisational behaviour',
        'organisational': 'Economic and organisational behaviour',
        'environmental': 'Environmental and urban behaviour',
        'culture': 'Culture, gender and migration',
        'gender': 'Culture, gender and migration',
        'combined': 'Combined behavioural x MENA',
    }

    if args.list:
        print("Query groups:")
        for group, queries in ALL_QUERY_GROUPS.items():
            print(f"  {group}: {len(queries)}")
            for key in queries.keys():
                print(f"    - {key}")
        exit(0)

    if args.everything:
        mode_label = 'ALL queries across ALL sources'
        all_papers = scraper.run_all(ALL_QUERIES, sources, args.num, incremental=args.incremental)
    elif args.all_models:
        mode_label = 'all behavioural model queries'
        all_papers = scraper.run_all(BEHAVIOURAL_MODEL_QUERIES, sources, args.num, incremental=args.incremental)
    elif args.all_mena:
        mode_label = 'all MENA queries'
        all_papers = scraper.run_all(MENA_QUERIES, sources, args.num, incremental=args.incremental)
    elif args.all_combined:
        mode_label = 'all combined queries'
        all_papers = scraper.run_all(COMBINED_QUERIES, sources, args.num, incremental=args.incremental)
    elif args.group:
        group_name = GROUP_ALIASES.get(args.group, args.group)
        if group_name not in ALL_QUERY_GROUPS:
            raise SystemExit(f"Unknown group '{args.group}'. Available: {', '.join(ALL_QUERY_GROUPS.keys())}")
        selected_group = group_name
        selected_queries = ALL_QUERY_GROUPS[group_name]
        mode_label = f'all queries in group: {group_name}'
        all_papers = scraper.run_all(selected_queries, sources, args.num, incremental=args.incremental)
    elif args.query:
        selected_group = 'custom key'
        qtext = ALL_QUERIES.get(args.query, args.query)
        mode_label = f'query: {args.query}'
        if 'arxiv' in sources:
            all_papers.extend(scraper.search_arxiv(qtext, args.num))
        if 'pubmed' in sources:
            all_papers.extend(scraper.search_pubmed(qtext, args.num))
        if 'semanticscholar' in sources:
            all_papers.extend(scraper.search_semanticscholar(qtext, args.num))
        if 'crossref' in sources:
            all_papers.extend(scraper.search_crossref(qtext, args.num))
        if 'openalex' in sources:
            all_papers.extend(scraper.search_openalex(qtext, args.num))
    elif args.custom:
        selected_group = 'custom'
        qtext = args.custom
        mode_label = 'custom query'
        # For arXiv, wrap plain text in all:"..."
        arxiv_q = f'all:"{qtext}"'
        if 'arxiv' in sources:
            all_papers.extend(scraper.search_arxiv(arxiv_q, args.num))
        if 'pubmed' in sources:
            all_papers.extend(scraper.search_pubmed(qtext, args.num))
        if 'semanticscholar' in sources:
            all_papers.extend(scraper.search_semanticscholar(qtext, args.num))
        if 'crossref' in sources:
            all_papers.extend(scraper.search_crossref(qtext, args.num))
        if 'openalex' in sources:
            all_papers.extend(scraper.search_openalex(qtext, args.num))
    else:
        print("Available query groups:")
        for group, queries in ALL_QUERY_GROUPS.items():
            print(f"  {group}:")
            for key in queries.keys():
                print(f"    - {key}")
        print("\nUse -q <key>, --group <group>, --custom <text>, or --everything")
        parser.print_help()
        exit(0)

    all_papers = scraper.dedup(all_papers)
    tag = args.tag or args.query or args.group or args.custom or "multi"
    tag = re.sub(r'[^A-Za-z0-9_\-]+', '_', tag)[:80] or "multi"
    out = scraper.save(all_papers, tag)
    print(f"\nSaved {len(all_papers)} unique papers to {out}")
    print(f"Mode: {mode_label}")
    if selected_group:
        print(f"Group: {selected_group}")