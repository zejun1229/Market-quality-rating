"""
Vela Market Quality Rating System — Step 1  (v2 — calibrated)
Generation & Feature Extraction

Generates 3 historical venture markets (founding years 2005-2019),
extracts 7-dimension feature classifications via Claude, and saves
everything to reference_population.json.

v2 calibration changes
-----------------------
* MARKET_SEEDS now carry full knowledge briefs (specific dates, metrics,
  funding rounds, named customers) so Claude generation is anchored to
  the historical record rather than inference.
* Base profile prompt adds a Key Metrics field (hard numbers only) and an
  Exclusions field (what was NOT yet true) to eliminate forward projection.
* Dimension prompts now carry quantitative thresholds (timing adoption %),
  semantic clarifications (market_size = current annual spend not TAM,
  customer_readiness = primary target segment not general public,
  regulatory = formally enacted and enforced rules only), and a required
  contradicting_evidence field to surface and bound model uncertainty.
* Token budgets raised to support richer, fully-grounded outputs.
"""

import json
import os
import time
import sys
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Market seeds — full knowledge briefs for maximum generation fidelity
# ---------------------------------------------------------------------------

MARKET_SEEDS = [
    {
        "domain": "US on-demand ride-hailing / transportation network companies",
        "ref_year": 2010,
        "knowledge_brief": """
COMPANY TIMELINE (at reference year)
- UberCab (later Uber): beta May 2010, public launch July 5 2010 in San Francisco only.
  Service type: black car / livery only (not taxis). No UberX yet (that launched June 2012).
  Seed: ~$200K friends/family 2009. First Round Capital seed round (amount undisclosed) early 2010.
  Benchmark Capital $12M Series A announced Feb 2011 (invested Oct 2010).
  Lyft: did NOT exist in 2010; founded June 2012.
  Sidecar: did NOT exist in 2010; launched Feb 2012.

MARKET SIZE AT REFERENCE YEAR
- US taxi / black car / limo industry: ~$11B annual revenue (IBISWorld 2010 report).
- Uber's own revenue in 2010: negligible, tens of thousands of rides total in SF.
- Uber was NOT operating outside San Francisco in 2010.

INFRASTRUCTURE STATE (2010)
- iPhone 3GS released June 2009; GPS chips standard in smartphones.
- App Store launched July 2008; Google Maps API v3 released 2009.
- Braintree mobile payments API available; Stripe did NOT launch until 2011.
- 4G LTE networks: Verizon launched Dec 2010 (end of year); AT&T 4G not until 2011.
- AWS EC2 / S3: live since 2006, mature and used by startups.

CUSTOMER LANDSCAPE (2010)
- Primary beachhead: San Francisco tech-industry professionals (early adopters).
- General public awareness of app-based ride-hailing: effectively zero outside SF tech scene.
- Taxi-riding mainstream consumers had no awareness of Uber in 2010.

REGULATORY (2010)
- San Francisco Municipal Transportation Agency issued cease-and-desist to UberCab, Oct 2010.
  Uber renamed itself from UberCab and continued operating.
- No state-level TNC (Transportation Network Company) legislation existed anywhere in the US in 2010.
  California PUC did not classify TNCs until Sept 2013.
- Uber operated under pre-existing livery/black car regulations; no new regulatory category existed.

INVESTOR SENTIMENT (2010)
- "On-demand mobile services" thesis just emerging; Benchmark, First Round early believers.
- No sector label ("ride-sharing," "TNC") existed yet; press called it "Uber" or "app-for-cars."
""",
    },
    {
        "domain": "Consumer and enterprise cloud file storage and sync (Dropbox, Box era)",
        "ref_year": 2009,
        "knowledge_brief": """
COMPANY TIMELINE (at reference year)
- Dropbox: Y Combinator W2007. Public beta Sept 2008. 1 million registered users by April 2009.
  $7.2M Series A from Sequoia Capital, Oct 2008. Built entirely on AWS S3.
  Revenue model: freemium (2GB free / $9.99/mo Pro). No enterprise product in 2009.
- Box (Box.net): founded 2005, pivoted to enterprise in 2009. ~$6M raised by end 2009
  (investors: Meritech Capital, U.S. Venture Partners). Had paying enterprise customers but small.
- SugarSync: launched consumer sync product 2009; direct Dropbox competitor; raised ~$5M.
- Microsoft SkyDrive: existed as file HOSTING (not sync) since 2007; no desktop sync client in 2009.
  Windows Live Mesh (sync) launched 2008 but clunky; no mainstream adoption.
- Google: Google Docs existed; no cloud file sync product. Google Drive: launched April 2012.
- Apple MobileMe: launched July 2008 ($99/yr); sync unreliable, widely criticised. iCloud: Oct 2011.

INFRASTRUCTURE STATE (2009)
- AWS S3 launched March 2006; 2009 pricing ~$0.15/GB/month, highly reliable.
- US broadband penetration: ~63% of households (Pew Internet, 2009).
- USB flash drives (competitor): 4GB cost ~$10; 8GB ~$20 in 2009.

MARKET SIZE AT REFERENCE YEAR
- Dropbox was effectively pre-revenue in 2009 (freemium, very few paid conversions).
- Box had early enterprise contracts; estimated ARR well under $10M.
- Total cloud file sync category revenue: ~$20-50M annually (all vendors combined, 2009 estimate).
- No analyst firm had sized this market; IDC and Gartner coverage was minimal.

CUSTOMER LANDSCAPE (2009)
- Consumer beachhead: tech-savvy professionals and developers; strong word-of-mouth via referral loops.
- Enterprise buyers: cautious IT departments worried about data sovereignty, HIPAA compliance.
  Enterprise adoption was early-evaluator stage; no mainstream IT procurement process existed.

REGULATORY (2009)
- No cloud-specific regulations existed. HIPAA applied to healthcare data in cloud (indirect constraint).
- EU Data Protection Directive 1995 applied but enforcement weak.
- No FedRAMP, no GDPR (2018), no sector-specific cloud storage rules.
""",
    },
    {
        "domain": "B2B SaaS marketing automation platforms, mid-market segment",
        "ref_year": 2012,
        "knowledge_brief": """
COMPANY TIMELINE (at reference year)
- Marketo: founded 2006. ~$58M ARR by Q4 2012. ~2,000 customers. Filed for IPO March 2013.
  Raised $50M Series D (Aug 2012) at ~$400M valuation. Mid-market and enterprise focus.
- HubSpot: founded 2006. ~$52M revenue 2012. 8,000+ customers. SMB and lower mid-market focus.
  Raised $35M Series D (Mar 2011); IPO Oct 2014.
- Eloqua: acquired by Oracle for $871M (10.4x revenue), announced Dec 2012.
  ~1,200 enterprise customers; ~$84M ARR at acquisition. Enterprise tier leader.
- Pardot: acquired by ExactTarget for $95.5M, Oct 2012. ~1,000 customers, SMB/mid-market.
  Salesforce subsequently acquired ExactTarget for $2.5B (June 2013).
- Act-On: ~$20M ARR 2012; Series B $20M from Norwest Venture Partners.
- Silverpop, Aprimo, Neolane: competing in enterprise tier.
- Adobe acquired Neolane for $600M (June 2013; NOT yet completed at reference year).

MARKET SIZE AT REFERENCE YEAR
- Raab Associates estimated total marketing automation industry revenue at ~$325M in 2012.
- Growing rapidly: 50%+ YoY revenue growth across key vendors.
- Gartner Magic Quadrant for CRM Lead Management: first issued 2012 — validation of category.
- Forrester Wave for Marketing Automation Technology: 2012 edition published.

INFRASTRUCTURE STATE (2012)
- Salesforce.com: >$3B revenue FY2013; deeply embedded in mid-market/enterprise sales stacks.
- REST APIs and webhooks: standard; Zapier launched 2011. Deep CRM integrations mature.
- Cloud delivery: AWS, Azure, Heroku all mature; SaaS delivery fully proven.
- Email deliverability infrastructure: SendGrid, Mailgun, Return Path well established.

CUSTOMER LANDSCAPE (2012)
- Primary beachhead: marketing operations managers and CMOs at B2B tech/SaaS companies, 50-500 employees.
- Mainstream mid-market adoption clearly underway by 2012; budget line items established.
- Typical deal: $18K-$60K/yr for mid-market; $60K-$200K+/yr for enterprise.

REGULATORY (2012)
- CAN-SPAM Act (2003): applies to commercial email; well-understood compliance requirement.
- Canada CASL: enacted July 2014 (NOT law at reference year; was in public consultation 2012).
- No additional sector-specific regulations for marketing automation platforms in 2012.
- GDPR: not until May 2018.
""",
    },
]

# ---------------------------------------------------------------------------
# 7-Dimension schema — v2 calibrated scoring guides
# ---------------------------------------------------------------------------

DIMENSIONS = [
    {
        "name": "timing",
        "description": "Position on the technology adoption / diffusion S-curve at the reference year",
        "options": ["pre_chasm", "early_chasm", "early_majority", "late_majority", "peak"],
        "scoring_guide": (
            "Use these quantitative adoption-rate thresholds as primary anchors:\n"
            "  pre_chasm    = <2% category adoption; innovators only; no mainstream crossing attempt\n"
            "  early_chasm  = 2-8% adoption; visionary buyers actively using it; "
            "chasm crossing underway but majority still skeptical\n"
            "  early_majority = 8-35% adoption; pragmatist buyers are the primary growth driver; "
            "chasm demonstrably crossed\n"
            "  late_majority  = 35-65% adoption; conservative buyers now purchasing; "
            "growth rate decelerating\n"
            "  peak           = >65% adoption; market saturated; net new growth near zero\n"
            "If adoption data is unavailable, use: number of paying customers relative to total "
            "addressable buyer count as a proxy."
        ),
    },
    {
        "name": "competition",
        "description": "Competitive intensity and structural dynamics at the reference year",
        "options": ["nascent", "fragmented", "consolidating", "consolidated", "commoditized"],
        "scoring_guide": (
            "nascent       = 1-3 identifiable players; category not yet labelled by analysts or press\n"
            "fragmented    = 4-15+ players; no player >20% market share; no M&A wave yet\n"
            "consolidating = active M&A underway; 2-3 acquirers absorbing smaller players; "
            "analyst coverage naming 'leaders'\n"
            "consolidated  = 2-4 dominant players control >60% market share; minor players marginalized\n"
            "commoditized  = price is primary differentiator; gross margins under 40%; "
            "switching costs near zero"
        ),
    },
    {
        "name": "market_size",
        "description": (
            "Current annual revenue/spend in the category at the reference year. "
            "IMPORTANT: score current observable market spend, NOT speculative future TAM. "
            "Use actual reported revenue figures from known players, supplemented by analyst estimates. "
            "If the category revenue is near-zero (e.g. product just launched), score 'micro' "
            "even if the underlying industry being disrupted is large."
        ),
        "options": ["micro", "small", "medium", "large", "mega"],
        "scoring_guide": (
            "micro   = current annual category revenue/spend <$100M\n"
            "small   = current annual category revenue/spend $100M-$1B\n"
            "medium  = current annual category revenue/spend $1B-$10B\n"
            "large   = current annual category revenue/spend $10B-$100B\n"
            "mega    = current annual category revenue/spend >$100B\n"
            "Anchor your estimate to the KNOWN revenue of players in the specific "
            "product category, not the broader industry being disrupted."
        ),
    },
    {
        "name": "customer_readiness",
        "description": (
            "Buyer awareness and willingness to purchase among the PRIMARY TARGET SEGMENT "
            "(the beachhead / lead-adopter customer), not the general population. "
            "Score the most advanced buyer cohort, not the average."
        ),
        "options": ["unaware", "aware", "interested", "ready", "adopting"],
        "scoring_guide": (
            "unaware    = even the beachhead segment does not know the product category exists\n"
            "aware      = beachhead knows the category exists but is passive / skeptical\n"
            "interested = beachhead actively requesting demos, running trials, evaluating options\n"
            "ready      = beachhead has allocated budget; formal procurement or purchasing processes in place\n"
            "adopting   = beachhead is actively purchasing and deploying at scale; "
            "measurable installed base and repeat purchases\n"
            "Be explicit about which specific buyer segment you are scoring."
        ),
    },
    {
        "name": "regulatory",
        "description": "Regulatory environment at the reference year",
        "options": ["unregulated", "light_touch", "moderate", "heavy", "restricted"],
        "scoring_guide": (
            "Score based ONLY on formally enacted and actively enforced regulations at the reference year. "
            "Do not score based on anticipated future regulation or regulatory commentary.\n"
            "unregulated  = no sector-specific rules exist; existing general laws apply peripherally\n"
            "light_touch  = industry self-regulation OR existing rules from adjacent sectors applied "
            "loosely; no dedicated enforcement action yet\n"
            "moderate     = sector-specific rules formally enacted; compliance required; "
            "regulators actively enforcing; manageable burden\n"
            "heavy        = substantial compliance burden materially affecting product design, "
            "sales cycles, and operations\n"
            "restricted   = prohibitive rules; product banned or effectively blocked in key markets"
        ),
    },
    {
        "name": "infrastructure",
        "description": "Maturity of the underlying enabling infrastructure at the reference year",
        "options": ["non_existent", "emerging", "developing", "mature", "commoditized"],
        "scoring_guide": (
            "Score the *critical path* enablers — the infrastructure without which the product "
            "could not function. Name them explicitly in your rationale.\n"
            "non_existent  = critical enablers not yet built or publicly available\n"
            "emerging      = critical enablers exist but unreliable, expensive, or limited to "
            "specialist access\n"
            "developing    = critical enablers functional and accessible but still maturing "
            "(e.g. cost declining, reliability improving, coverage expanding)\n"
            "mature        = critical enablers robust, reliable, widely accessible, "
            "and affordable for target customers\n"
            "commoditized  = critical enablers are essentially free or near-free; "
            "multiple interchangeable providers; no meaningful barrier"
        ),
    },
    {
        "name": "market_structure",
        "description": "Clarity and stability of value chain roles and industry structure at the reference year",
        "options": ["undefined", "emerging", "forming", "defined", "mature"],
        "scoring_guide": (
            "undefined = no analyst or press category label; no recognizable value chain; "
            "no industry association or standards body\n"
            "emerging  = category label beginning to appear in press; first-mover advantages; "
            "value chain roles being pioneered by 1-2 players\n"
            "forming   = 2-4 distinct value chain tiers crystallizing (e.g. infra / platform / app); "
            "analyst firms beginning to cover; investment theses being articulated\n"
            "defined   = clear industry roles, buyer personas, integration standards, "
            "and competitive tiers; Gartner/Forrester coverage active\n"
            "mature    = established ecosystem; industry associations; certification bodies; "
            "procurement standards; 3rd-party integrators and resellers"
        ),
    },
]

# ---------------------------------------------------------------------------
# Claude helpers
# ---------------------------------------------------------------------------

def get_client() -> anthropic.Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("your_"):
        sys.exit(
            "ERROR: ANTHROPIC_API_KEY is not set or is a placeholder. "
            "Please update your .env file."
        )
    return anthropic.Anthropic(api_key=key)


def generate_base_profile(client: anthropic.Anthropic, seed: dict) -> dict:
    """Task A — Generate a high-fidelity plain-text historical market base profile."""
    prompt = (
        "You are a senior venture capital market analyst producing a reference-quality "
        "historical market profile. Your output will be used as ground truth for a "
        "quantitative market classification system — accuracy and precision are paramount.\n\n"
        "=== ASSIGNMENT ===\n"
        f"Market domain: {seed['domain']}\n"
        f"Reference year: {seed['ref_year']}\n\n"
        "=== VERIFIED KNOWLEDGE BRIEF ===\n"
        "The following facts are confirmed accurate. You MUST incorporate them and MUST NOT "
        "contradict them. Do not invent details that go beyond or conflict with this brief.\n\n"
        f"{seed['knowledge_brief'].strip()}\n\n"
        "=== OUTPUT FORMAT ===\n"
        "Write your response using EXACTLY these six labelled fields. "
        "Each field must be on its own line, starting with the label.\n\n"
        "Market: [concise market name, 4-8 words, no year]\n\n"
        "Context: [4-5 sentences. Cover: (1) the state of the market at the reference year, "
        "(2) the key technology or business model inflection point that created the opportunity, "
        "(3) investor sentiment, (4) one or two landmark events of the year. "
        "Every claim must be derivable from the knowledge brief above.]\n\n"
        "Buyers: [3 sentences. (1) Name the PRIMARY beachhead buyer segment with specifics — "
        "company size, role, industry vertical. (2) Describe their core pain point. "
        "(3) Describe their purchasing behaviour and budget allocation at this year.]\n\n"
        "Players: [3 sentences. Name specific companies with their approximate revenue/ARR, "
        "customer counts, or funding amounts where available from the brief. "
        "Distinguish incumbents from startups. State which major players had NOT yet entered.]\n\n"
        "Key Metrics: [A bulleted list of 4-6 hard data points from the knowledge brief. "
        "Format each as: - [metric name]: [value] ([source/year if known]). "
        "Include: funding rounds, user/customer counts, revenue figures, or market size estimates.]\n\n"
        "Exclusions: [2-3 sentences explicitly stating what was NOT yet true at this reference year. "
        "This must include: which major competitors had not yet launched, "
        "which regulatory frameworks had not yet been enacted, "
        "and which enabling technologies had not yet matured or shipped.]\n\n"
        f"Reference year: {seed['ref_year']}\n\n"
        "CRITICAL RULES:\n"
        "- All claims must be consistent with the knowledge brief.\n"
        "- Do not project any events that happened AFTER the reference year.\n"
        "- Be specific: use company names, dollar figures, and dates wherever possible.\n"
        "- Do not use hedging phrases like 'would eventually' or 'was poised to'."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1400,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text.strip()
    profile: dict = {
        "raw_text": raw_text,
        "ref_year": seed["ref_year"],
        "domain": seed["domain"],
    }

    # Parse labelled fields (handles multi-line values via section scanning)
    field_map = {
        "Market:":        "market_name",
        "Context:":       "context",
        "Buyers:":        "buyers",
        "Players:":       "players",
        "Key Metrics:":   "key_metrics",
        "Exclusions:":    "exclusions",
        "Reference year:":"reference_year_label",
    }
    current_field = None
    current_lines: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        matched = False
        for label, key in field_map.items():
            if stripped.startswith(label):
                # Save previous field
                if current_field:
                    profile[current_field] = " ".join(current_lines).strip()
                current_field = key
                current_lines = [stripped[len(label):].strip()]
                matched = True
                break
        if not matched and current_field:
            current_lines.append(stripped)
    if current_field:
        profile[current_field] = " ".join(current_lines).strip()

    return profile


def extract_dimension(
    client: anthropic.Anthropic, profile: dict, dimension: dict
) -> dict:
    """Task B — Extract a single dimension classification as structured JSON."""
    prompt = (
        "You are a senior quantitative venture capital researcher performing structured "
        "market classification. Your classifications will be cross-validated against "
        "independently retrieved evidence — precision and intellectual honesty are essential.\n\n"
        "=== MARKET PROFILE ===\n"
        f"{profile['raw_text']}\n\n"
        "=== DIMENSION TO CLASSIFY ===\n"
        f"Dimension:   {dimension['name']}\n"
        f"Description: {dimension['description']}\n\n"
        f"Scoring guide:\n{dimension['scoring_guide']}\n\n"
        "=== CLASSIFICATION TASK ===\n"
        f"Choose ONE value from this exact list: {dimension['options']}\n\n"
        "Return ONLY a valid JSON object — no markdown, no preamble, no trailing text.\n"
        "The JSON must have exactly these five fields:\n"
        "{\n"
        f'  "dimension": "{dimension["name"]}",\n'
        '  "classification": "<one value from the options list>",\n'
        '  "confidence": "<high|medium|low>",\n'
        '  "rationale": "<3 sentences: (1) the specific evidence from the profile that '
        'drives this classification, (2) where this market falls on the scoring guide scale, '
        '(3) the key metric or fact that most strongly anchors your choice>",\n'
        '  "contradicting_evidence": "<1-2 sentences: what evidence could argue for the '
        'adjacent classification above or below yours, and why you rejected it>"\n'
        "}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if model adds them
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.lower().startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Market processing
# ---------------------------------------------------------------------------

def process_market(client: anthropic.Anthropic, seed: dict, index: int) -> dict:
    """Run all Step 1 tasks for one market."""
    print(f"\n[Market {index + 1}/3]  {seed['domain'][:65]}  (ref {seed['ref_year']})")

    profile = generate_base_profile(client, seed)
    print(f"  + Profile: {profile.get('market_name', 'N/A')}")

    dimensions_result: dict = {}
    for dim in DIMENSIONS:
        print(f"  -> [{dim['name']}]...", end=" ", flush=True)
        time.sleep(0.4)
        try:
            result = extract_dimension(client, profile, dim)
            if result.get("classification") not in dim["options"]:
                result["classification"] = dim["options"][0]
                result["validation_warning"] = "Classification not in allowed options; defaulted."
            dimensions_result[dim["name"]] = result
            print(f"{result['classification']}  [{result.get('confidence','?')}]")
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            print(f"PARSE ERROR: {exc}")
            dimensions_result[dim["name"]] = {
                "dimension": dim["name"],
                "classification": "unknown",
                "confidence": "low",
                "rationale": "",
                "contradicting_evidence": "",
                "error": str(exc),
            }

    return {
        "id": f"market_{index + 1:03d}",
        "domain": seed["domain"],
        "ref_year": seed["ref_year"],
        "base_profile": profile,
        "dimensions": dimensions_result,
        "step1_complete": True,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  VELA MARKET QUALITY RATING — STEP 1  (v2 calibrated)")
    print("  Generation & Feature Extraction  (Claude)")
    print("=" * 65)

    client = get_client()
    population: list[dict] = []

    for i, seed in enumerate(MARKET_SEEDS):
        market_data = process_market(client, seed, i)
        population.append(market_data)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_population.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump({"schema_version": "1.1", "markets": population}, fh, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 65}")
    print(f"  Saved {len(population)} markets to reference_population.json")
    print(f"\n  7-Dimension Summary")
    print(f"  {'Market':<38}  {'Dimension':<24}  {'Class':<22}  [conf]")
    print(f"  {'-'*38}  {'-'*24}  {'-'*22}  ------")
    for m in population:
        name = m["base_profile"].get("market_name", m["domain"])[:38]
        for dim_name, dim_data in m["dimensions"].items():
            cl = dim_data.get("classification", "N/A")
            cf = dim_data.get("confidence", "?")
            print(f"  {name:<38}  {dim_name:<24}  {cl:<22}  [{cf}]")
            name = ""  # only print market name on first row

    print(f"\n{'=' * 65}")
    print("  Step 1 v2 complete. Run pipeline_step2.py to verify.")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
