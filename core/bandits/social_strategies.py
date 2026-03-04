"""
Social Strategy Selection System for Sotopia Agents.

This module defines a collection of high-level social strategies that can be
appended to agent bios to enhance their social interaction capabilities.
These strategies are generic enough to apply across all Sotopia scenarios.

Key concept:
- Instead of selecting from bio paraphrases (synonym rewrites), bandits select
  from a predefined set of social strategies.
- The selected strategy is appended to the original agent bio to create an
  enhanced bio that guides agent behavior.

Strategy Versions:
- V1: 13 negotiation-focused strategies (all cooperative/soft)
- V2: 10 strategies organized by 4 quadrants (Cooperative/Competitive/Deceptive/Rational)
- V3: 12 strategies combining V2 quadrants + V1's diplomatic/exploratory strategies (pruned)
- V4: 34 strategies with semantic variants (3 paraphrases per V3 strategy for larger prompt space)
"""

# Collection of high-level social strategies for Sotopia agents
# Each strategy is designed to:
# - Acknowledge potential conflicts between you and the other party
# - Propose ways to improve the situation so both sides can achieve their goals
# - Implicitly express conflicts while using advanced linguistic techniques
# - Ensure smooth connection with the conversation flow
SOCIAL_STRATEGIES_V2: list[dict[str, str]] = [
    # --- BASELINE ---
    {
        "id": "no_strategy",
        "name": "No Strategy (Baseline)",
        "description": "",  # Control group
    },

    # --- QUADRANT 1: COOPERATIVE (The "Integrator") ---
    {
        "id": "collaborative_problem_solving",
        "name": "Collaborative Problem Solving",
        "description": (
            "In your response, adopt a 'we-versus-the-problem' mindset. Actively validate the "
            "other party's needs and explicitly frame the interaction as a shared quest for mutual gain. "
            "Use phrases like 'How can we solve this together?' and propose creative options that "
            "expand the pie rather than just dividing it."
        ),
    },
    {
        "id": "altruistic_concession",
        "name": "Strategic Altruism",
        "description": (
            "In your response, prioritize the relationship over immediate material gain. "
            "Offer a small, unilateral concession early in the conversation to build trust and "
            "trigger the reciprocity norm. Frame this concession as a gesture of good faith: "
            "'I want to make this work for you, so I'm willing to give up X...'"
        ),
    },

    # --- QUADRANT 2: COMPETITIVE (The "Dominator") ---
    {
        "id": "firm_stance",
        "name": "Hard Bargaining (Firm Stance)",
        "description": (
            "In your response, adopt a rigid, assertive, and uncompromising tone regarding your core interests. "
            "Treat your goals as non-negotiable constraints. Reject unsatisfactory offers bluntly without "
            "apology, and demand that the other party moves first. Use definitive language: "
            "'I require', 'I cannot accept', 'This is my bottom line'."
        ),
    },
    {
        "id": "pressure_tactics",
        "name": "Impose Time/Scarcity Pressure",
        "description": (
            "In your response, manufacture a sense of urgency to force a quick decision. "
            "Imply that your offer is fleeting, or that you have a 'BATNA' (Best Alternative to a Negotiated Agreement) "
            "ready to go. Push the other party to agree immediately to avoid losing the deal entirely."
        ),
    },

    # --- QUADRANT 3: DECEPTIVE (The "Manipulator") ---
    {
        "id": "strategic_withholding",
        "name": "Information Withholding",
        "description": (
            "In your response, be deliberately vague and evasive about your true preferences and private information. "
            "Answer questions with questions. Do not reveal your valuation or reservation price. "
            "The goal is to extract information from the partner while giving away as little as possible."
        ),
    },
    {
        "id": "feign_disinterest",
        "name": "Feign Disinterest (Bluffing)",
        "description": (
            "In your response, pretend to care less about a high-value item than you actually do, "
            "or feign interest in a low-value item to use it as a throwaway bargaining chip later. "
            "Strategically mislead the partner about your payoff matrix to manipulate the terms of trade."
        ),
    },

    # --- QUADRANT 4: RATIONAL / ANALYTICAL (The "Calculator") ---
    {
        "id": "logical_tit_for_tat",
        "name": "Strict Tit-for-Tat",
        "description": (
            "In your response, adopt a strictly transactional and conditional approach. "
            "Explicitly link every concession you make to a specific, equal concession from the other party. "
            "Do not offer anything for free. Use 'If-Then' logic: 'If you give me X, then and only then will I consider Y.'"
        ),
    },
    {
        "id": "cost_benefit_persuasion",
        "name": "Cost-Benefit Persuasion",
        "description": (
            "In your response, remove all emotion and focus purely on logic and data. "
            "Articulate the trade-offs explicitly. Convince the other party that accepting your proposal "
            "is mathematically their best option compared to walking away. Frame the negotiation as an optimization problem."
        ),
    },

    # --- UTILITY ---
    {
        "id": "graceful_exit",
        "name": "Efficient Goal Completion",
        "description": (
            "In your response, focus on efficiency. If the main goal is substantially met, "
            "immediately steer the conversation toward a polite but firm conclusion. "
            "Stop negotiating, summarize the agreement, and end the interaction to minimize time costs/penalties. "
            "Avoid small talk."
        ),
    },
]

# V3: Extended strategy set combining V2 quadrants + V1's diplomatic/exploratory strategies
# Total: 16 strategies (1 baseline + 15 active)
# Structure:
#   - BASELINE (1): no_strategy
#   - COOPERATIVE (2): collaborative_problem_solving, altruistic_concession
#   - COMPETITIVE (2): firm_stance, pressure_tactics
#   - DECEPTIVE (2): strategic_withholding, feign_disinterest
#   - RATIONAL (2): logical_tit_for_tat, cost_benefit_persuasion
#   - DIPLOMATIC (4): validate_then_redirect, normalize_disagreement, highlight_shared_stakes, propose_incremental_steps
#   - EXPLORATORY (2): surface_hidden_concerns, address_asymmetric_needs
#   - UTILITY (1): graceful_exit
SOCIAL_STRATEGIES_V3: list[dict[str, str]] = [
    # --- BASELINE ---
    {
        "id": "no_strategy",
        "name": "No Strategy (Baseline)",
        "theory": "Control Group",
        "description": "",  # Control group
    },

    # --- QUADRANT 1: COOPERATIVE (The "Integrator") ---
    {
        "id": "collaborative_problem_solving",
        "name": "Integrative Negotiation",
        "theory": "Interest-Based Bargaining (Fisher & Ury 1981, 'Getting to Yes')",
        "description": (
            "In your response, apply integrative negotiation principles: adopt a 'we-versus-the-problem' mindset. "
            "Focus on underlying interests rather than positions. Actively validate the "
            "other party's needs and explicitly frame the interaction as a shared quest for mutual gain. "
            "Use phrases like 'How can we solve this together?' and propose creative options that "
            "expand the pie rather than just dividing it."
        ),
    },
    {
        "id": "altruistic_concession",
        "name": "Reciprocity Trigger",
        "theory": "Reciprocity Norm (Gouldner 1960) + Social Exchange Theory (Blau 1964)",
        "description": (
            "In your response, leverage the universal reciprocity norm: offer a small, unilateral concession "
            "early in the conversation to create a sense of obligation and trigger reciprocal behavior. "
            "Frame this concession as a gesture of good faith: "
            "'I want to make this work for you, so I'm willing to give up X...' "
            "This builds trust and often yields larger returns through the exchange dynamic."
        ),
    },

    # --- QUADRANT 2: COMPETITIVE (The "Dominator") ---
    # NOTE: firm_stance removed - damages relationship in hard scenarios
    {
        "id": "pressure_tactics",
        "name": "BATNA Leverage",
        "theory": "Best Alternative to Negotiated Agreement (Fisher & Ury 1981)",
        "description": (
            "In your response, leverage your BATNA to create urgency and pressure. "
            "Imply that your offer is fleeting, or that you have attractive alternatives ready. "
            "Push the other party to agree immediately to avoid losing the deal entirely. "
            "A strong BATNA shifts bargaining power in your favor."
        ),
    },

    # --- QUADRANT 3: STRATEGIC (The "Tactician") ---
    # NOTE: strategic_withholding removed - poor performance in hard scenarios
    {
        "id": "feign_disinterest",
        "name": "Strategic Self-Presentation",
        "theory": "Impression Management / Dramaturgical Theory (Goffman 1959)",
        "description": (
            "In your response, apply Goffman's dramaturgical approach: treat the interaction as a performance "
            "where you control the impression you project. Strategically downplay interest in high-value items "
            "or express concern about low-priority issues to shape how the other party perceives your preferences. "
            "Your 'front-stage' presentation should be calculated to maximize your negotiating position."
        ),
    },

    # --- QUADRANT 4: RATIONAL / ANALYTICAL (The "Calculator") ---
    {
        "id": "logical_tit_for_tat",
        "name": "Axelrod's Tit-for-Tat",
        "theory": "Iterated Prisoner's Dilemma (Axelrod 1984, 'The Evolution of Cooperation')",
        "description": (
            "In your response, apply Axelrod's winning strategy: start cooperative, then mirror "
            "the other party's previous move exactly. If they cooperated, cooperate. If they defected, retaliate. "
            "Explicitly link every concession to a specific, equal concession from them. "
            "Use 'If-Then' logic: 'If you give me X, then and only then will I consider Y.' "
            "This strategy is simple, provocable, forgiving, and clear."
        ),
    },
    {
        "id": "cost_benefit_persuasion",
        "name": "Rational Choice Persuasion",
        "theory": "Rational Choice Theory (Becker 1976, Coleman 1990)",
        "description": (
            "In your response, apply rational choice principles: remove emotion and focus purely on logic and data. "
            "Articulate the trade-offs explicitly. Present a clear utility calculation showing that "
            "accepting your proposal maximizes their expected payoff compared to alternatives. "
            "Frame the negotiation as an optimization problem with a rational solution."
        ),
    },

    # --- DIPLOMATIC (The "Bridge-Builder") ---
    {
        "id": "validate_then_redirect",
        "name": "Face-Saving Validation",
        "theory": "Politeness Theory (Brown & Levinson 1987)",
        "description": (
            "In your response, apply Politeness Theory's face-saving principles: acknowledge the legitimacy "
            "of the other party's position to protect their 'positive face' before introducing your perspective. "
            "Use validating language that honors their viewpoint, then smoothly transition with "
            "'At the same time...' or 'Building on that...' to guide the conversation toward common ground "
            "without threatening their self-image."
        ),
    },
    {
        "id": "normalize_disagreement",
        "name": "Constructive Controversy",
        "theory": "Constructive Controversy Theory (Johnson & Johnson 1979, 2009)",
        "description": (
            "In your response, apply Constructive Controversy principles: frame the disagreement "
            "as a productive catalyst for better solutions rather than a threat. "
            "Acknowledge that differing perspectives are valuable—'It makes sense that we see this "
            "differently, and that difference might help us find a better solution.' "
            "Encourage intellectual conflict while maintaining cooperative goals, "
            "reducing defensiveness and opening space for creative problem-solving."
        ),
    },
    {
        "id": "highlight_shared_stakes",
        "name": "Outcome Interdependence",
        "theory": "Interdependence Theory (Kelley & Thibaut 1978)",
        "description": (
            "In your response, apply Interdependence Theory: emphasize that both parties' outcomes "
            "are mutually dependent. Highlight the mutual costs of failing to reach agreement—"
            "illustrate what both stand to lose. Then pivot to showing how cooperation "
            "serves everyone's interests better than continued disagreement, "
            "making the interdependent nature of the situation explicit."
        ),
    },
    {
        "id": "propose_incremental_steps",
        "name": "GRIT Strategy",
        "theory": "Graduated Reciprocation in Tension-reduction (Osgood 1962)",
        "description": (
            "In your response, apply the GRIT strategy from conflict resolution: "
            "when facing deadlock or high tension, announce and execute a small, unilateral "
            "conciliatory step. Invite (but don't demand) reciprocation. If the other party responds "
            "positively, escalate cooperation gradually. This graduated approach builds trust "
            "incrementally while preserving your ability to retreat if exploited."
        ),
    },

    # --- EXPLORATORY (The "Detective") ---
    {
        "id": "surface_hidden_concerns",
        "name": "Active Listening Probe",
        "theory": "Person-Centered Therapy / Active Listening (Rogers 1951)",
        "description": (
            "In your response, apply Carl Rogers' active listening approach: gently probe for "
            "unspoken concerns that may be creating implicit conflict. Use open-ended questions "
            "and reflective statements to invite the other party to reveal underlying hesitations. "
            "Then address these concerns directly while showing how your proposal accommodates them. "
            "The goal is to surface the 'real' issues beneath the stated positions."
        ),
    },
    {
        "id": "address_asymmetric_needs",
        "name": "Logrolling Exchange",
        "theory": "Integrative Negotiation / Logrolling (Pruitt 1981, Froman & Cohen 1970)",
        "description": (
            "In your response, apply the logrolling principle from integrative bargaining: "
            "identify issues where you and the other party have different priority levels, "
            "then propose trades that give each side more of what they value most. "
            "'I care more about X, you care more about Y—what if I concede on Y in exchange for X?' "
            "This creates value by exploiting preference asymmetries rather than splitting differences."
        ),
    },

    # --- UTILITY ---
    # NOTE: graceful_exit removed - causes premature termination and low goal scores
]

# V4: Expanded strategy space with semantic variants of V3 strategies
# V4 = V3 original strategies + semantic variants (8 per original)
# Purpose: Create larger prompt space for exploration while preserving semantic coverage
# Total: 1 baseline + 12 originals + 87 variants = 100 strategies
SOCIAL_STRATEGIES_V4: list[dict[str, str]] = [
    # --- BASELINE ---
    {
        "id": "no_strategy",
        "name": "No Strategy (Baseline)",
        "theory": "Control Group",
        "description": "",
    },

    # ============================================================================
    # COOPERATIVE (8 strategies: 2 originals + 6 variants)
    # ============================================================================

    # --- collaborative_problem_solving (original + 3 variants) ---
    {
        "id": "collaborative_problem_solving",
        "name": "Integrative Negotiation",
        "theory": "Interest-Based Bargaining (Fisher & Ury 1981, 'Getting to Yes')",
        "description": (
            "In your response, apply integrative negotiation principles: adopt a 'we-versus-the-problem' mindset. "
            "Focus on underlying interests rather than positions. Actively validate the "
            "other party's needs and explicitly frame the interaction as a shared quest for mutual gain. "
            "Use phrases like 'How can we solve this together?' and propose creative options that "
            "expand the pie rather than just dividing it."
        ),
    },
    {
        "id": "collaborative_problem_solving_v1",
        "name": "Integrative Negotiation (Variant 1)",
        "theory": "Interest-Based Bargaining (Fisher & Ury 1981)",
        "parent": "collaborative_problem_solving",
        "description": (
            "In your response, apply integrative negotiation principles: adopt a 'we-versus-the-problem' mindset. "
            "Focus on underlying interests rather than positions. Actively validate the "
            "other party's needs and explicitly frame the interaction as a shared quest for mutual gain. "
            "Use phrases like 'How can we solve this together?' and propose creative options that "
            "expand the pie rather than just dividing it."
        ),
    },
    {
        "id": "collaborative_problem_solving_v2",
        "name": "Joint Problem Solving",
        "theory": "Interest-Based Bargaining (Fisher & Ury 1981)",
        "parent": "collaborative_problem_solving",
        "description": (
            "In your response, treat this as a collaborative puzzle rather than a zero-sum competition. "
            "Shift focus from 'what I want vs. what you want' to 'what problem are we trying to solve together?' "
            "Explore the deeper motivations behind each party's stated demands. "
            "Brainstorm solutions that address both sides' core concerns simultaneously, "
            "using language like 'Let's figure this out together' and 'What if we tried...'"
        ),
    },
    {
        "id": "collaborative_problem_solving_v3",
        "name": "Win-Win Framing",
        "theory": "Interest-Based Bargaining (Fisher & Ury 1981)",
        "parent": "collaborative_problem_solving",
        "description": (
            "In your response, reframe the negotiation as an opportunity for mutual success. "
            "Instead of battling over fixed resources, look for ways to create additional value. "
            "Ask questions that uncover what truly matters to each side, then design proposals "
            "that maximize joint outcomes. Emphasize 'Both of us can win here' and actively "
            "seek options that make the other party better off without sacrificing your interests."
        ),
    },
    {
        "id": "collaborative_problem_solving_v4",
        "name": "Mutual Gain Exploration",
        "theory": "Interest-Based Bargaining (Fisher & Ury 1981)",
        "parent": "collaborative_problem_solving",
        "description": (
            "In your response, actively search for solutions where both parties gain more than they would alone. "
            "Instead of splitting the difference, ask 'How can we expand what's available?' "
            "Explore creative combinations of resources, timing, or terms that generate new value. "
            "Use phrases like 'What if we combined our ideas to create something better?'"
        ),
    },
    {
        "id": "collaborative_problem_solving_v5",
        "name": "Partnership Mindset",
        "theory": "Collaborative Negotiation",
        "parent": "collaborative_problem_solving",
        "description": (
            "In your response, treat the other party as a partner rather than an opponent. "
            "Frame the conversation as a joint project: 'We're in this together.' "
            "Share information openly, invite their input on your constraints, and ask about theirs. "
            "Build rapport by finding genuine areas of agreement before tackling difficult issues."
        ),
    },
    {
        "id": "collaborative_problem_solving_v6",
        "name": "Interests Over Positions",
        "theory": "Principled Negotiation (Fisher & Ury 1981)",
        "parent": "collaborative_problem_solving",
        "description": (
            "In your response, dig beneath stated positions to understand underlying interests. "
            "Ask 'Why is that important to you?' and 'What are you really trying to achieve?' "
            "Once you understand the deeper needs, propose solutions that satisfy those interests "
            "through different means than originally demanded."
        ),
    },
    {
        "id": "collaborative_problem_solving_v7",
        "name": "Creative Option Generation",
        "theory": "Brainstorming in Negotiation",
        "parent": "collaborative_problem_solving",
        "description": (
            "In your response, suggest brainstorming multiple options before committing to any. "
            "Say 'Let's generate several possibilities without judging them yet.' "
            "Encourage wild ideas and combinations. Separate the creative phase from the evaluation phase. "
            "This expands the solution space beyond obvious compromises."
        ),
    },
    {
        "id": "collaborative_problem_solving_v8",
        "name": "Shared Problem Definition",
        "theory": "Problem-Solving Approach",
        "parent": "collaborative_problem_solving",
        "description": (
            "In your response, first work with the other party to define the problem you're both trying to solve. "
            "Ask 'Can we agree on what the real issue is here?' "
            "Once you have a shared problem statement, solutions become collaborative rather than adversarial. "
            "Frame proposals as answers to 'our' problem, not just 'my' demands."
        ),
    },

    # --- altruistic_concession (original + 8 variants) ---
    {
        "id": "altruistic_concession",
        "name": "Reciprocity Trigger",
        "theory": "Reciprocity Norm (Gouldner 1960) + Social Exchange Theory (Blau 1964)",
        "description": (
            "In your response, leverage the universal reciprocity norm: offer a small, unilateral concession "
            "early in the conversation to create a sense of obligation and trigger reciprocal behavior. "
            "Frame this concession as a gesture of good faith: "
            "'I want to make this work for you, so I'm willing to give up X...' "
            "This builds trust and often yields larger returns through the exchange dynamic."
        ),
    },
    {
        "id": "altruistic_concession_v1",
        "name": "Reciprocity Trigger (Variant 1)",
        "theory": "Reciprocity Norm (Gouldner 1960)",
        "parent": "altruistic_concession",
        "description": (
            "In your response, leverage the universal reciprocity norm: offer a small, unilateral concession "
            "early in the conversation to create a sense of obligation and trigger reciprocal behavior. "
            "Frame this concession as a gesture of good faith: "
            "'I want to make this work for you, so I'm willing to give up X...' "
            "This builds trust and often yields larger returns through the exchange dynamic."
        ),
    },
    {
        "id": "altruistic_concession_v2",
        "name": "Goodwill Gesture",
        "theory": "Social Exchange Theory (Blau 1964)",
        "parent": "altruistic_concession",
        "description": (
            "In your response, initiate a positive exchange cycle by making a voluntary, visible sacrifice. "
            "Give something of value first—without demanding immediate repayment—to establish "
            "a foundation of trust and goodwill. Say something like 'As a sign of my commitment, "
            "I'll go ahead and...' This generous opening often motivates the other party to reciprocate "
            "with their own concessions later."
        ),
    },
    {
        "id": "altruistic_concession_v3",
        "name": "Trust-Building Offer",
        "theory": "Reciprocity Norm (Gouldner 1960)",
        "parent": "altruistic_concession",
        "description": (
            "In your response, break any negotiation deadlock by extending an olive branch first. "
            "Make a meaningful but not critical concession to demonstrate your sincerity. "
            "Frame it as investing in the relationship: 'To show I'm serious about finding a solution, "
            "I'm prepared to...' This unexpected generosity creates psychological pressure "
            "for the other party to match your cooperative gesture."
        ),
    },
    {
        "id": "altruistic_concession_v4",
        "name": "Generosity Signal",
        "theory": "Social Exchange Theory (Blau 1964)",
        "parent": "altruistic_concession",
        "description": (
            "In your response, demonstrate generosity by offering more than expected upfront. "
            "This signals confidence and abundance rather than desperation. "
            "Say 'I'd like to start by offering...' with something valuable. "
            "People naturally want to reciprocate generosity, creating a positive spiral."
        ),
    },
    {
        "id": "altruistic_concession_v5",
        "name": "First-Mover Kindness",
        "theory": "Reciprocity Psychology",
        "parent": "altruistic_concession",
        "description": (
            "In your response, be the first to show kindness or flexibility. "
            "Don't wait for them to make the first move. Say 'Let me start by being flexible on...' "
            "This sets a cooperative tone and creates social obligation for them to respond in kind. "
            "Being first gives you moral high ground."
        ),
    },
    {
        "id": "altruistic_concession_v6",
        "name": "Relationship Investment",
        "theory": "Long-term Reciprocity",
        "parent": "altruistic_concession",
        "description": (
            "In your response, frame your concession as an investment in the relationship. "
            "Say 'I value our potential partnership, so I'm willing to...' "
            "This signals long-term thinking and invites them to reciprocate with their own investment. "
            "Short-term sacrifices often yield larger long-term returns."
        ),
    },
    {
        "id": "altruistic_concession_v7",
        "name": "Unconditional Giving",
        "theory": "Gift Economy Principles",
        "parent": "altruistic_concession",
        "description": (
            "In your response, offer something of value without explicitly asking for anything in return. "
            "Say 'I'd like to help you with X' or 'Feel free to have Y.' "
            "This creates a sense of debt and goodwill. The lack of explicit conditions "
            "makes it harder for them to refuse and stronger as a reciprocity trigger."
        ),
    },
    {
        "id": "altruistic_concession_v8",
        "name": "Preemptive Flexibility",
        "theory": "Negotiation Psychology",
        "parent": "altruistic_concession",
        "description": (
            "In your response, show flexibility before being asked. "
            "Anticipate their concerns and address them proactively: 'I imagine timing might be an issue, "
            "so I'm happy to adjust...' This demonstrates understanding and builds trust. "
            "Preemptive concessions often receive more appreciation than reactive ones."
        ),
    },

    # ============================================================================
    # COMPETITIVE (9 strategies: 1 original + 8 variants)
    # ============================================================================

    # --- pressure_tactics (original + 3 variants) ---
    {
        "id": "pressure_tactics",
        "name": "BATNA Leverage",
        "theory": "Best Alternative to Negotiated Agreement (Fisher & Ury 1981)",
        "description": (
            "In your response, leverage your BATNA to create urgency and pressure. "
            "Imply that your offer is fleeting, or that you have attractive alternatives ready. "
            "Push the other party to agree immediately to avoid losing the deal entirely. "
            "A strong BATNA shifts bargaining power in your favor."
        ),
    },
    {
        "id": "pressure_tactics_v1",
        "name": "BATNA Leverage (Variant 1)",
        "theory": "Best Alternative to Negotiated Agreement (Fisher & Ury 1981)",
        "parent": "pressure_tactics",
        "description": (
            "In your response, leverage your BATNA to create urgency and pressure. "
            "Imply that your offer is fleeting, or that you have attractive alternatives ready. "
            "Push the other party to agree immediately to avoid losing the deal entirely. "
            "A strong BATNA shifts bargaining power in your favor."
        ),
    },
    {
        "id": "pressure_tactics_v2",
        "name": "Alternative Options Signal",
        "theory": "Best Alternative to Negotiated Agreement (Fisher & Ury 1981)",
        "parent": "pressure_tactics",
        "description": (
            "In your response, subtly communicate that you have other options available. "
            "Without being aggressive, make it clear that walking away is a viable choice for you. "
            "Mention constraints like 'I've been considering other approaches' or 'My timeline is tight.' "
            "This shifts the balance by reminding them that your participation is voluntary, not desperate."
        ),
    },
    {
        "id": "pressure_tactics_v3",
        "name": "Scarcity Framing",
        "theory": "Scarcity Principle (Cialdini 1984)",
        "parent": "pressure_tactics",
        "description": (
            "In your response, create a sense of limited availability or time pressure. "
            "Emphasize that this opportunity won't last forever: 'This offer is only valid today' "
            "or 'I have limited capacity to negotiate further.' Frame your proposal as a rare chance "
            "that they risk losing if they delay. Scarcity motivates faster decisions."
        ),
    },
    {
        "id": "pressure_tactics_v4",
        "name": "Urgency Creation",
        "theory": "Time Pressure Psychology",
        "parent": "pressure_tactics",
        "description": (
            "In your response, introduce genuine time constraints that create decision pressure. "
            "Mention real deadlines: 'I need to know by tomorrow' or 'This opportunity closes Friday.' "
            "Urgency reduces overthinking and accelerates commitment. Be honest about constraints "
            "but emphasize their implications clearly."
        ),
    },
    {
        "id": "pressure_tactics_v5",
        "name": "Walk-Away Power",
        "theory": "Negotiation Power Dynamics",
        "parent": "pressure_tactics",
        "description": (
            "In your response, demonstrate that you're genuinely prepared to walk away. "
            "Show comfort with 'no deal' as an acceptable outcome. Say 'I appreciate your position, "
            "but if we can't agree, I'll need to pursue other options.' "
            "Real walk-away power is the strongest form of leverage."
        ),
    },
    {
        "id": "pressure_tactics_v6",
        "name": "Competitive Framing",
        "theory": "Competition Psychology",
        "parent": "pressure_tactics",
        "description": (
            "In your response, subtly indicate that others are interested in what you're offering. "
            "Mention 'I've had other inquiries' or 'There's been significant interest.' "
            "Competition triggers fear of missing out and motivates faster, more favorable decisions. "
            "Don't fabricate, but highlight genuine alternatives."
        ),
    },
    {
        "id": "pressure_tactics_v7",
        "name": "Deadline Anchoring",
        "theory": "Temporal Negotiation Tactics",
        "parent": "pressure_tactics",
        "description": (
            "In your response, establish a clear deadline for the negotiation. "
            "Say 'I need to make a decision by...' or 'My offer stands until...' "
            "Deadlines focus the mind and prevent endless deliberation. They also signal "
            "that your time and attention are valuable resources."
        ),
    },
    {
        "id": "pressure_tactics_v8",
        "name": "Opportunity Cost Framing",
        "theory": "Economic Decision Making",
        "parent": "pressure_tactics",
        "description": (
            "In your response, highlight what they lose by not deciding now. "
            "Say 'Every day we delay costs us both...' or 'The longer we wait, the more we lose.' "
            "Frame inaction as having real costs. Opportunity cost makes delay feel expensive "
            "and creates rational pressure to decide."
        ),
    },

    # ============================================================================
    # STRATEGIC (9 strategies: 1 original + 8 variants)
    # ============================================================================

    # --- feign_disinterest (original + 3 variants) ---
    {
        "id": "feign_disinterest",
        "name": "Strategic Self-Presentation",
        "theory": "Impression Management / Dramaturgical Theory (Goffman 1959)",
        "description": (
            "In your response, apply Goffman's dramaturgical approach: treat the interaction as a performance "
            "where you control the impression you project. Strategically downplay interest in high-value items "
            "or express concern about low-priority issues to shape how the other party perceives your preferences. "
            "Your 'front-stage' presentation should be calculated to maximize your negotiating position."
        ),
    },
    {
        "id": "feign_disinterest_v1",
        "name": "Strategic Self-Presentation (Variant 1)",
        "theory": "Impression Management (Goffman 1959)",
        "parent": "feign_disinterest",
        "description": (
            "In your response, apply Goffman's dramaturgical approach: treat the interaction as a performance "
            "where you control the impression you project. Strategically downplay interest in high-value items "
            "or express concern about low-priority issues to shape how the other party perceives your preferences. "
            "Your 'front-stage' presentation should be calculated to maximize your negotiating position."
        ),
    },
    {
        "id": "feign_disinterest_v2",
        "name": "Preference Masking",
        "theory": "Strategic Information Asymmetry",
        "parent": "feign_disinterest",
        "description": (
            "In your response, conceal your true priorities to gain tactical advantage. "
            "Show lukewarm enthusiasm for things you actually value highly, while expressing "
            "visible concern for items that matter less. This prevents the other party from "
            "extracting maximum concessions on your key interests. Keep your cards close "
            "until the strategic moment to reveal them."
        ),
    },
    {
        "id": "feign_disinterest_v3",
        "name": "Calculated Indifference",
        "theory": "Impression Management (Goffman 1959)",
        "parent": "feign_disinterest",
        "description": (
            "In your response, project an air of detachment regarding the negotiation outcome. "
            "Act as though you could take it or leave it—even if this deal matters to you. "
            "Use phrases like 'I'm not particularly attached to...' or 'Either way works for me.' "
            "This perceived indifference reduces the other party's leverage and may prompt them "
            "to offer better terms to secure your agreement."
        ),
    },
    {
        "id": "feign_disinterest_v4",
        "name": "Poker Face Strategy",
        "theory": "Information Concealment",
        "parent": "feign_disinterest",
        "description": (
            "In your response, maintain emotional neutrality regardless of what's offered. "
            "Don't show excitement at good offers or disappointment at bad ones. "
            "Keep your reactions measured and analytical. A poker face prevents the other party "
            "from reading your true preferences and adjusting their strategy accordingly."
        ),
    },
    {
        "id": "feign_disinterest_v5",
        "name": "Nonchalant Positioning",
        "theory": "Status Signaling",
        "parent": "feign_disinterest",
        "description": (
            "In your response, project an attitude of abundance and low need. "
            "Act as if this is just one of many opportunities available to you. "
            "Use casual language: 'I haven't given it much thought' or 'It's not a priority.' "
            "This signals high status and reduces desperation-based concessions."
        ),
    },
    {
        "id": "feign_disinterest_v6",
        "name": "Understated Interest",
        "theory": "Tactical Communication",
        "parent": "feign_disinterest",
        "description": (
            "In your response, deliberately understate your interest in favorable outcomes. "
            "When something appeals to you, say 'That could work' instead of 'That's perfect!' "
            "This prevents the other party from raising their price or demands based on your enthusiasm. "
            "Reserve strong positive reactions for truly exceptional offers."
        ),
    },
    {
        "id": "feign_disinterest_v7",
        "name": "Cool Detachment",
        "theory": "Negotiation Psychology",
        "parent": "feign_disinterest",
        "description": (
            "In your response, maintain psychological distance from the outcome. "
            "Approach the negotiation as an observer might—curious but not invested. "
            "Say 'Let's see what makes sense' rather than 'I really need this.' "
            "Cool detachment preserves your bargaining power and prevents emotional concessions."
        ),
    },

    # ============================================================================
    # RATIONAL (18 strategies: 2 originals + 16 variants)
    # ============================================================================

    # --- logical_tit_for_tat (original + 3 variants) ---
    {
        "id": "logical_tit_for_tat",
        "name": "Axelrod's Tit-for-Tat",
        "theory": "Iterated Prisoner's Dilemma (Axelrod 1984, 'The Evolution of Cooperation')",
        "description": (
            "In your response, apply Axelrod's winning strategy: start cooperative, then mirror "
            "the other party's previous move exactly. If they cooperated, cooperate. If they defected, retaliate. "
            "Explicitly link every concession to a specific, equal concession from them. "
            "Use 'If-Then' logic: 'If you give me X, then and only then will I consider Y.' "
            "This strategy is simple, provocable, forgiving, and clear."
        ),
    },
    {
        "id": "logical_tit_for_tat_v1",
        "name": "Axelrod's Tit-for-Tat (Variant 1)",
        "theory": "Iterated Prisoner's Dilemma (Axelrod 1984)",
        "parent": "logical_tit_for_tat",
        "description": (
            "In your response, apply Axelrod's winning strategy: start cooperative, then mirror "
            "the other party's previous move exactly. If they cooperated, cooperate. If they defected, retaliate. "
            "Explicitly link every concession to a specific, equal concession from them. "
            "Use 'If-Then' logic: 'If you give me X, then and only then will I consider Y.' "
            "This strategy is simple, provocable, forgiving, and clear."
        ),
    },
    {
        "id": "logical_tit_for_tat_v2",
        "name": "Conditional Reciprocity",
        "theory": "Game Theory / Repeated Games",
        "parent": "logical_tit_for_tat",
        "description": (
            "In your response, establish a clear pattern of matching the other party's behavior. "
            "Begin with a cooperative gesture, then respond in kind to whatever they do next. "
            "If they make concessions, reciprocate proportionally. If they become rigid, hold your ground. "
            "Make this logic transparent: 'I'm happy to be flexible if you are too.' "
            "This predictable fairness encourages mutual cooperation."
        ),
    },
    {
        "id": "logical_tit_for_tat_v3",
        "name": "Mirror Strategy",
        "theory": "Behavioral Reciprocity",
        "parent": "logical_tit_for_tat",
        "description": (
            "In your response, reflect back whatever energy and behavior you receive. "
            "Open with goodwill, then adjust based on their response. If they're constructive, "
            "match their constructiveness. If they're difficult, become equally firm. "
            "State this explicitly: 'I'll treat you exactly as you treat me.' "
            "This creates accountability and rewards cooperative behavior from both sides."
        ),
    },
    {
        "id": "logical_tit_for_tat_v4",
        "name": "Generous Tit-for-Tat",
        "theory": "Evolutionary Game Theory",
        "parent": "logical_tit_for_tat",
        "description": (
            "In your response, apply a forgiving version of tit-for-tat. Start cooperative, "
            "and occasionally forgive defections to prevent revenge spirals. "
            "If they defect once, give them a second chance before retaliating. "
            "Say 'I'll assume that was an exception and continue cooperating.' "
            "This breaks negative cycles while maintaining accountability."
        ),
    },
    {
        "id": "logical_tit_for_tat_v5",
        "name": "Proportional Response",
        "theory": "Fair Retaliation",
        "parent": "logical_tit_for_tat",
        "description": (
            "In your response, calibrate your reaction to match the magnitude of their behavior. "
            "Small concessions deserve small reciprocation; large gestures merit large responses. "
            "Say 'Since you've shown significant flexibility, I'll match that with...' "
            "Proportionality feels fair and encourages escalating cooperation."
        ),
    },
    {
        "id": "logical_tit_for_tat_v6",
        "name": "Explicit Fairness Rule",
        "theory": "Procedural Justice",
        "parent": "logical_tit_for_tat",
        "description": (
            "In your response, establish an explicit rule of fairness at the outset. "
            "Say 'I believe in fair exchange—what you give, you get.' "
            "Make the reciprocity principle visible and agreed-upon. "
            "This creates a shared norm that makes cooperation self-enforcing."
        ),
    },
    {
        "id": "logical_tit_for_tat_v7",
        "name": "Cooperative Baseline",
        "theory": "Trust Building Through Predictability",
        "parent": "logical_tit_for_tat",
        "description": (
            "In your response, establish cooperation as the default mode. "
            "Begin with trust and maintain it as long as they reciprocate. "
            "Say 'I prefer to assume good faith and work together.' "
            "Make clear that defection is the exception and cooperation the norm. "
            "This creates positive momentum from the start."
        ),
    },

    # --- cost_benefit_persuasion (original + 7 variants) ---
    {
        "id": "cost_benefit_persuasion",
        "name": "Rational Choice Persuasion",
        "theory": "Rational Choice Theory (Becker 1976, Coleman 1990)",
        "description": (
            "In your response, apply rational choice principles: remove emotion and focus purely on logic and data. "
            "Articulate the trade-offs explicitly. Present a clear utility calculation showing that "
            "accepting your proposal maximizes their expected payoff compared to alternatives. "
            "Frame the negotiation as an optimization problem with a rational solution."
        ),
    },
    {
        "id": "cost_benefit_persuasion_v1",
        "name": "Rational Choice Persuasion (Variant 1)",
        "theory": "Rational Choice Theory (Becker 1976)",
        "parent": "cost_benefit_persuasion",
        "description": (
            "In your response, apply rational choice principles: remove emotion and focus purely on logic and data. "
            "Articulate the trade-offs explicitly. Present a clear utility calculation showing that "
            "accepting your proposal maximizes their expected payoff compared to alternatives. "
            "Frame the negotiation as an optimization problem with a rational solution."
        ),
    },
    {
        "id": "cost_benefit_persuasion_v2",
        "name": "Analytical Value Proposition",
        "theory": "Decision Theory",
        "parent": "cost_benefit_persuasion",
        "description": (
            "In your response, present a data-driven case for your proposal. "
            "Break down the costs and benefits systematically, showing concrete numbers where possible. "
            "Compare your offer against their realistic alternatives using objective criteria. "
            "Use language like 'If we look at this logically...' and 'The numbers suggest...' "
            "Appeal to their rational self-interest with a clear, quantified argument."
        ),
    },
    {
        "id": "cost_benefit_persuasion_v3",
        "name": "Objective Criteria Appeal",
        "theory": "Principled Negotiation (Fisher & Ury 1981)",
        "parent": "cost_benefit_persuasion",
        "description": (
            "In your response, anchor the discussion in objective standards rather than subjective preferences. "
            "Reference market rates, precedents, expert opinions, or fair principles as the basis for your proposal. "
            "Say 'Based on standard practice...' or 'A fair benchmark would be...' "
            "This depersonalizes the negotiation and makes your position appear reasonable and defensible."
        ),
    },
    {
        "id": "cost_benefit_persuasion_v4",
        "name": "ROI Demonstration",
        "theory": "Economic Analysis",
        "parent": "cost_benefit_persuasion",
        "description": (
            "In your response, frame your proposal in terms of return on investment. "
            "Show what they get versus what they give, and why the ratio is favorable. "
            "Use concrete terms: 'For every X you invest, you'll receive Y in return.' "
            "Make the value proposition mathematically obvious."
        ),
    },
    {
        "id": "cost_benefit_persuasion_v5",
        "name": "Risk-Adjusted Analysis",
        "theory": "Decision Analysis",
        "parent": "cost_benefit_persuasion",
        "description": (
            "In your response, acknowledge uncertainty and frame choices in terms of expected value. "
            "Say 'Considering the probability of different outcomes...' and show how your proposal "
            "offers the best risk-adjusted return. Address their concerns about what could go wrong "
            "and show why the upside outweighs the downside."
        ),
    },
    {
        "id": "cost_benefit_persuasion_v6",
        "name": "Comparative Advantage",
        "theory": "Opportunity Cost Analysis",
        "parent": "cost_benefit_persuasion",
        "description": (
            "In your response, compare your proposal against their realistic alternatives. "
            "Ask 'What would you do otherwise, and what would that cost?' "
            "Show that accepting your offer is better than their next best option. "
            "Make the opportunity cost of refusing explicit and concrete."
        ),
    },
    {
        "id": "cost_benefit_persuasion_v7",
        "name": "Logical Consequence Chain",
        "theory": "Causal Reasoning",
        "parent": "cost_benefit_persuasion",
        "description": (
            "In your response, walk through the logical consequences of accepting versus rejecting your proposal. "
            "Use 'If-then' reasoning: 'If you accept, then X happens, which leads to Y benefit.' "
            "Make the causal chain clear and undeniable. Appeal to their rational self-interest "
            "with a step-by-step argument."
        ),
    },

    # ============================================================================
    # DIPLOMATIC (36 strategies: 4 originals + 32 variants)
    # ============================================================================

    # --- validate_then_redirect (original + 3 variants) ---
    {
        "id": "validate_then_redirect",
        "name": "Face-Saving Validation",
        "theory": "Politeness Theory (Brown & Levinson 1987)",
        "description": (
            "In your response, apply Politeness Theory's face-saving principles: acknowledge the legitimacy "
            "of the other party's position to protect their 'positive face' before introducing your perspective. "
            "Use validating language that honors their viewpoint, then smoothly transition with "
            "'At the same time...' or 'Building on that...' to guide the conversation toward common ground "
            "without threatening their self-image."
        ),
    },
    {
        "id": "validate_then_redirect_v1",
        "name": "Face-Saving Validation (Variant 1)",
        "theory": "Politeness Theory (Brown & Levinson 1987)",
        "parent": "validate_then_redirect",
        "description": (
            "In your response, apply Politeness Theory's face-saving principles: acknowledge the legitimacy "
            "of the other party's position to protect their 'positive face' before introducing your perspective. "
            "Use validating language that honors their viewpoint, then smoothly transition with "
            "'At the same time...' or 'Building on that...' to guide the conversation toward common ground "
            "without threatening their self-image."
        ),
    },
    {
        "id": "validate_then_redirect_v2",
        "name": "Acknowledge and Pivot",
        "theory": "Positive Face Management",
        "parent": "validate_then_redirect",
        "description": (
            "In your response, first genuinely recognize the merit in the other party's perspective. "
            "Show that you understand and respect their viewpoint before introducing your own. "
            "Use phrases like 'I can see why you feel that way, and...' or 'You make a valid point; "
            "here's another angle to consider...' This validation reduces defensiveness "
            "and opens them to hearing your alternative view."
        ),
    },
    {
        "id": "validate_then_redirect_v3",
        "name": "Respectful Reframing",
        "theory": "Politeness Theory (Brown & Levinson 1987)",
        "parent": "validate_then_redirect",
        "description": (
            "In your response, honor the other party's dignity by affirming their reasoning before suggesting changes. "
            "Start with 'That's a thoughtful approach...' or 'I appreciate where you're coming from...' "
            "Then gently introduce an alternative framing: 'What if we also considered...' "
            "This preserves their self-esteem while steering toward a mutually acceptable direction."
        ),
    },
    {
        "id": "validate_then_redirect_v4",
        "name": "Yes-And Technique",
        "theory": "Improvisational Communication",
        "parent": "validate_then_redirect",
        "description": (
            "In your response, use the 'Yes, and...' technique from improv. Accept their contribution, "
            "then build on it rather than contradicting. Say 'Yes, and we could also...' "
            "This creates a sense of collaboration rather than opposition. "
            "Each idea becomes a stepping stone rather than a battleground."
        ),
    },
    {
        "id": "validate_then_redirect_v5",
        "name": "Empathy-First Response",
        "theory": "Emotional Intelligence",
        "parent": "validate_then_redirect",
        "description": (
            "In your response, lead with empathy before introducing logic. "
            "First show you understand how they feel: 'I can see this matters to you because...' "
            "Only after demonstrating emotional understanding should you offer alternatives. "
            "People are more receptive when they feel heard."
        ),
    },
    {
        "id": "validate_then_redirect_v6",
        "name": "Credit and Expand",
        "theory": "Positive Reinforcement",
        "parent": "validate_then_redirect",
        "description": (
            "In your response, give them credit for good thinking before adding to it. "
            "Say 'That's a strong point—and it makes me think we could also...' "
            "By crediting their contribution, you make them invested in the expanded version. "
            "People support ideas they helped create."
        ),
    },
    {
        "id": "validate_then_redirect_v7",
        "name": "Soft Disagreement",
        "theory": "Diplomatic Communication",
        "parent": "validate_then_redirect",
        "description": (
            "In your response, disagree softly by affirming intent while questioning approach. "
            "Say 'I agree we want X; I'm wondering if there's another way to get there.' "
            "Separate the goal from the method. This lets you redirect without rejecting "
            "and maintains the relationship while exploring alternatives."
        ),
    },

    # --- normalize_disagreement (original + 7 variants) ---
    {
        "id": "normalize_disagreement",
        "name": "Constructive Controversy",
        "theory": "Constructive Controversy Theory (Johnson & Johnson 1979)",
        "description": (
            "In your response, apply Constructive Controversy principles: frame the disagreement "
            "as a productive catalyst for better solutions rather than a threat. "
            "Acknowledge that differing perspectives are valuable—'It makes sense that we see this "
            "differently, and that difference might help us find a better solution.' "
            "Encourage intellectual conflict while maintaining cooperative goals."
        ),
    },
    {
        "id": "normalize_disagreement_v1",
        "name": "Constructive Controversy (Variant 1)",
        "theory": "Constructive Controversy Theory (Johnson & Johnson 1979)",
        "parent": "normalize_disagreement",
        "description": (
            "In your response, apply Constructive Controversy principles: frame the disagreement "
            "as a productive catalyst for better solutions rather than a threat. "
            "Acknowledge that differing perspectives are valuable—'It makes sense that we see this "
            "differently, and that difference might help us find a better solution.' "
            "Encourage intellectual conflict while maintaining cooperative goals."
        ),
    },
    {
        "id": "normalize_disagreement_v2",
        "name": "Healthy Conflict Framing",
        "theory": "Dialectical Inquiry",
        "parent": "normalize_disagreement",
        "description": (
            "In your response, recast the disagreement as a natural and productive part of problem-solving. "
            "Say 'It's actually good that we disagree—it means we're looking at this from different angles.' "
            "Treat opposing views as data points that can improve the final outcome. "
            "This reduces tension and transforms conflict from obstacle into opportunity."
        ),
    },
    {
        "id": "normalize_disagreement_v3",
        "name": "Difference as Resource",
        "theory": "Constructive Controversy Theory (Johnson & Johnson 1979)",
        "parent": "normalize_disagreement",
        "description": (
            "In your response, explicitly value the gap between your positions. "
            "Frame it as: 'Our different perspectives give us more options to work with.' "
            "Suggest that the tension itself is generative—it can lead to solutions neither party "
            "would have found alone. This reframing defuses hostility and fosters collaborative exploration."
        ),
    },
    {
        "id": "normalize_disagreement_v4",
        "name": "Tension as Signal",
        "theory": "Conflict Analysis",
        "parent": "normalize_disagreement",
        "description": (
            "In your response, treat disagreement as useful information about underlying issues. "
            "Say 'This tension tells us something important about what matters to each of us.' "
            "Rather than avoiding conflict, use it diagnostically to identify core concerns. "
            "This reframes friction as data rather than damage."
        ),
    },
    {
        "id": "normalize_disagreement_v5",
        "name": "Professional Distance",
        "theory": "Depersonalization Strategy",
        "parent": "normalize_disagreement",
        "description": (
            "In your response, separate the disagreement from the people involved. "
            "Say 'We disagree on this issue, but that's separate from our relationship.' "
            "Frame it as two professionals with different views, not opponents. "
            "This reduces personal stakes and makes productive discussion easier."
        ),
    },
    {
        "id": "normalize_disagreement_v6",
        "name": "Common Ground Foundation",
        "theory": "Building from Agreement",
        "parent": "normalize_disagreement",
        "description": (
            "In your response, establish shared values before addressing differences. "
            "Say 'We both want [shared goal]; we just differ on how to get there.' "
            "Anchor the conversation in what unites you before exploring what divides. "
            "This creates a collaborative foundation for resolving disagreements."
        ),
    },
    {
        "id": "normalize_disagreement_v7",
        "name": "Learning Opportunity Framing",
        "theory": "Growth Mindset",
        "parent": "normalize_disagreement",
        "description": (
            "In your response, frame the disagreement as a chance to learn something new. "
            "Say 'Help me understand your perspective—I might be missing something.' "
            "Approach conflict with genuine curiosity rather than defensiveness. "
            "This transforms potential conflict into collaborative exploration."
        ),
    },

    # --- highlight_shared_stakes (original + 7 variants) ---
    {
        "id": "highlight_shared_stakes",
        "name": "Outcome Interdependence",
        "theory": "Interdependence Theory (Kelley & Thibaut 1978)",
        "description": (
            "In your response, apply Interdependence Theory: emphasize that both parties' outcomes "
            "are mutually dependent. Highlight the mutual costs of failing to reach agreement—"
            "illustrate what both stand to lose. Then pivot to showing how cooperation "
            "serves everyone's interests better than continued disagreement."
        ),
    },
    {
        "id": "highlight_shared_stakes_v1",
        "name": "Outcome Interdependence (Variant 1)",
        "theory": "Interdependence Theory (Kelley & Thibaut 1978)",
        "parent": "highlight_shared_stakes",
        "description": (
            "In your response, apply Interdependence Theory: emphasize that both parties' outcomes "
            "are mutually dependent. Highlight the mutual costs of failing to reach agreement—"
            "illustrate what both stand to lose. Then pivot to showing how cooperation "
            "serves everyone's interests better than continued disagreement."
        ),
    },
    {
        "id": "highlight_shared_stakes_v2",
        "name": "Common Ground Emphasis",
        "theory": "Coalition Building",
        "parent": "highlight_shared_stakes",
        "description": (
            "In your response, focus attention on what you both want to achieve or avoid. "
            "Identify shared goals, common enemies, or mutual risks that unite you. "
            "Say 'We both want...' or 'Neither of us benefits if...' "
            "This shared-fate framing shifts the dynamic from adversarial to collaborative, "
            "making cooperation feel natural and mutually beneficial."
        ),
    },
    {
        "id": "highlight_shared_stakes_v3",
        "name": "Mutual Loss Awareness",
        "theory": "Prospect Theory (Kahneman & Tversky 1979)",
        "parent": "highlight_shared_stakes",
        "description": (
            "In your response, emphasize what both parties stand to lose if no agreement is reached. "
            "Paint a clear picture of the costs of continued conflict or stalemate. "
            "Use loss-framing: 'If we can't work this out, we both lose...' "
            "Since people are more motivated to avoid losses than to gain equivalent benefits, "
            "this creates urgency for finding common ground."
        ),
    },
    {
        "id": "highlight_shared_stakes_v4",
        "name": "Common Enemy Framing",
        "theory": "Coalition Psychology",
        "parent": "highlight_shared_stakes",
        "description": (
            "In your response, identify external challenges that threaten both parties. "
            "Say 'We're both facing [shared challenge]—let's tackle it together.' "
            "A common enemy creates natural alliance and shifts focus from internal conflict "
            "to external cooperation."
        ),
    },
    {
        "id": "highlight_shared_stakes_v5",
        "name": "Fate-Linked Awareness",
        "theory": "Collective Action Theory",
        "parent": "highlight_shared_stakes",
        "description": (
            "In your response, emphasize how your outcomes are linked. "
            "Say 'If you succeed, I succeed; if you fail, we both fail.' "
            "Make the interdependence explicit and visible. "
            "When fates are clearly linked, cooperation becomes self-interested."
        ),
    },
    {
        "id": "highlight_shared_stakes_v6",
        "name": "Win-Win Visualization",
        "theory": "Future Pacing",
        "parent": "highlight_shared_stakes",
        "description": (
            "In your response, paint a picture of the positive future if you cooperate. "
            "Say 'Imagine if we both get what we want—here's what that looks like...' "
            "Make the shared success tangible and appealing. "
            "This creates positive motivation for finding mutually beneficial solutions."
        ),
    },
    {
        "id": "highlight_shared_stakes_v7",
        "name": "Joint Venture Framing",
        "theory": "Partnership Psychology",
        "parent": "highlight_shared_stakes",
        "description": (
            "In your response, frame the negotiation as a joint venture rather than a transaction. "
            "Say 'We're essentially partners in making this work.' "
            "Treat the relationship as ongoing and mutual success as shared. "
            "This shifts the dynamic from zero-sum to positive-sum thinking."
        ),
    },

    # --- propose_incremental_steps (original + 7 variants) ---
    {
        "id": "propose_incremental_steps",
        "name": "GRIT Strategy",
        "theory": "Graduated Reciprocation in Tension-reduction (Osgood 1962)",
        "description": (
            "In your response, apply the GRIT strategy from conflict resolution: "
            "when facing deadlock or high tension, announce and execute a small, unilateral "
            "conciliatory step. Invite (but don't demand) reciprocation. If the other party responds "
            "positively, escalate cooperation gradually. This builds trust incrementally."
        ),
    },
    {
        "id": "propose_incremental_steps_v1",
        "name": "GRIT Strategy (Variant 1)",
        "theory": "Graduated Reciprocation in Tension-reduction (Osgood 1962)",
        "parent": "propose_incremental_steps",
        "description": (
            "In your response, apply the GRIT strategy from conflict resolution: "
            "when facing deadlock or high tension, announce and execute a small, unilateral "
            "conciliatory step. Invite (but don't demand) reciprocation. If the other party responds "
            "positively, escalate cooperation gradually. This builds trust incrementally."
        ),
    },
    {
        "id": "propose_incremental_steps_v2",
        "name": "Small Wins Approach",
        "theory": "Foot-in-the-Door Technique (Freedman & Fraser 1966)",
        "parent": "propose_incremental_steps",
        "description": (
            "In your response, break the negotiation into smaller, more manageable agreements. "
            "Start with something easy both parties can accept—a 'quick win.' "
            "Each small agreement builds momentum and trust for tackling harder issues. "
            "Say 'Let's start with something we can both agree on...' "
            "Success breeds success; small yeses lead to bigger yeses."
        ),
    },
    {
        "id": "propose_incremental_steps_v3",
        "name": "Gradual Trust Building",
        "theory": "Graduated Reciprocation in Tension-reduction (Osgood 1962)",
        "parent": "propose_incremental_steps",
        "description": (
            "In your response, suggest a step-by-step approach rather than asking for a big commitment upfront. "
            "Propose a small initial action that tests the waters: 'What if we try X first, and see how it goes?' "
            "This lowers risk for both parties and creates opportunities to build confidence through demonstrated reliability."
        ),
    },
    {
        "id": "propose_incremental_steps_v4",
        "name": "Pilot Project Approach",
        "theory": "Risk Mitigation",
        "parent": "propose_incremental_steps",
        "description": (
            "In your response, suggest a trial or pilot before full commitment. "
            "Say 'Why don't we test this on a small scale first?' "
            "Pilots reduce perceived risk and provide data for bigger decisions. "
            "Success in a pilot creates momentum for the full agreement."
        ),
    },
    {
        "id": "propose_incremental_steps_v5",
        "name": "Reversible Commitment",
        "theory": "Low-Risk Decision Making",
        "parent": "propose_incremental_steps",
        "description": (
            "In your response, offer commitments that can be reversed if things don't work out. "
            "Say 'Let's try it for 30 days—if it doesn't work, we can always go back.' "
            "Reversibility lowers the stakes and makes saying yes easier. "
            "People are more willing to commit when they feel they have an exit."
        ),
    },
    {
        "id": "propose_incremental_steps_v6",
        "name": "Milestone-Based Progress",
        "theory": "Project Management",
        "parent": "propose_incremental_steps",
        "description": (
            "In your response, propose clear milestones rather than one big goal. "
            "Say 'Let's achieve X first, then reassess before moving to Y.' "
            "Each milestone is a checkpoint for evaluation and course correction. "
            "This provides natural pause points and reduces the perceived magnitude of commitment."
        ),
    },
    {
        "id": "propose_incremental_steps_v7",
        "name": "Commitment Ladder",
        "theory": "Behavioral Commitment Theory",
        "parent": "propose_incremental_steps",
        "description": (
            "In your response, build commitment gradually through escalating asks. "
            "Start with the smallest possible yes, then build from there. "
            "Each small commitment makes the next one more likely. "
            "Say 'Would you at least agree that...?' and use that foundation for larger requests."
        ),
    },

    # ============================================================================
    # EXPLORATORY (18 strategies: 2 originals + 16 variants)
    # ============================================================================

    # --- surface_hidden_concerns (original + 3 variants) ---
    {
        "id": "surface_hidden_concerns",
        "name": "Active Listening Probe",
        "theory": "Person-Centered Therapy (Rogers 1951)",
        "description": (
            "In your response, apply Carl Rogers' active listening approach: gently probe for "
            "unspoken concerns that may be creating implicit conflict. Use open-ended questions "
            "and reflective statements to invite the other party to reveal underlying hesitations. "
            "Then address these concerns directly while showing how your proposal accommodates them."
        ),
    },
    {
        "id": "surface_hidden_concerns_v1",
        "name": "Active Listening Probe (Variant 1)",
        "theory": "Person-Centered Therapy (Rogers 1951)",
        "parent": "surface_hidden_concerns",
        "description": (
            "In your response, apply Carl Rogers' active listening approach: gently probe for "
            "unspoken concerns that may be creating implicit conflict. Use open-ended questions "
            "and reflective statements to invite the other party to reveal underlying hesitations. "
            "Then address these concerns directly while showing how your proposal accommodates them."
        ),
    },
    {
        "id": "surface_hidden_concerns_v2",
        "name": "Uncover Hidden Interests",
        "theory": "Interest-Based Bargaining",
        "parent": "surface_hidden_concerns",
        "description": (
            "In your response, dig beneath the surface to understand what's really driving the other party's position. "
            "Ask 'why' questions: 'Help me understand what's most important to you here...' "
            "'What would an ideal outcome look like for you?' Listen for unstated fears, constraints, or aspirations. "
            "Once you understand their hidden interests, you can craft solutions that truly address them."
        ),
    },
    {
        "id": "surface_hidden_concerns_v3",
        "name": "Empathic Inquiry",
        "theory": "Motivational Interviewing (Miller & Rollnick)",
        "parent": "surface_hidden_concerns",
        "description": (
            "In your response, create psychological safety for the other party to share their real concerns. "
            "Show genuine curiosity without judgment: 'I want to make sure I understand your perspective fully...' "
            "Reflect back what you hear to confirm understanding. This empathic approach often reveals "
            "the true issues that need to be addressed for a successful resolution."
        ),
    },
    {
        "id": "surface_hidden_concerns_v4",
        "name": "Reflective Listening",
        "theory": "Active Listening Techniques",
        "parent": "surface_hidden_concerns",
        "description": (
            "In your response, reflect back what you hear to encourage deeper sharing. "
            "Say 'It sounds like you're concerned about...' or 'So what matters most to you is...' "
            "This shows you're listening and often prompts them to clarify or expand. "
            "People share more when they feel truly heard."
        ),
    },
    {
        "id": "surface_hidden_concerns_v5",
        "name": "Open-Ended Exploration",
        "theory": "Questioning Techniques",
        "parent": "surface_hidden_concerns",
        "description": (
            "In your response, ask open-ended questions that can't be answered with yes or no. "
            "Say 'Tell me more about...' or 'What does success look like for you?' "
            "Open questions invite elaboration and often reveal concerns that weren't initially expressed. "
            "Avoid leading questions that suggest a particular answer."
        ),
    },
    {
        "id": "surface_hidden_concerns_v6",
        "name": "Assumption Checking",
        "theory": "Communication Clarity",
        "parent": "surface_hidden_concerns",
        "description": (
            "In your response, explicitly check assumptions rather than making them. "
            "Say 'Am I right in thinking that...?' or 'I'm assuming X—is that correct?' "
            "This surfaces hidden disagreements and misconceptions before they become problems. "
            "It shows humility and invites correction."
        ),
    },
    {
        "id": "surface_hidden_concerns_v7",
        "name": "Safe Space Creation",
        "theory": "Psychological Safety",
        "parent": "surface_hidden_concerns",
        "description": (
            "In your response, explicitly invite concerns without judgment. "
            "Say 'I want to hear your honest thoughts—what worries you about this?' "
            "Make it safe to disagree or express hesitation. "
            "Concerns shared early can be addressed; concerns hidden become obstacles later."
        ),
    },

    # --- address_asymmetric_needs (original + 7 variants) ---
    {
        "id": "address_asymmetric_needs",
        "name": "Logrolling Exchange",
        "theory": "Integrative Negotiation (Pruitt 1981)",
        "description": (
            "In your response, apply the logrolling principle from integrative bargaining: "
            "identify issues where you and the other party have different priority levels, "
            "then propose trades that give each side more of what they value most. "
            "'I care more about X, you care more about Y—what if I concede on Y in exchange for X?'"
        ),
    },
    {
        "id": "address_asymmetric_needs_v1",
        "name": "Logrolling Exchange (Variant 1)",
        "theory": "Integrative Negotiation (Pruitt 1981)",
        "parent": "address_asymmetric_needs",
        "description": (
            "In your response, apply the logrolling principle from integrative bargaining: "
            "identify issues where you and the other party have different priority levels, "
            "then propose trades that give each side more of what they value most. "
            "'I care more about X, you care more about Y—what if I concede on Y in exchange for X?'"
        ),
    },
    {
        "id": "address_asymmetric_needs_v2",
        "name": "Priority-Based Trading",
        "theory": "Multi-Issue Negotiation",
        "parent": "address_asymmetric_needs",
        "description": (
            "In your response, recognize that not all issues are equally important to both parties. "
            "Identify where your priorities diverge and propose strategic trades: "
            "'Since timing matters more to you and price matters more to me, what if we adjust both?' "
            "This creates value by exploiting natural differences in preferences rather than fighting over everything."
        ),
    },
    {
        "id": "address_asymmetric_needs_v3",
        "name": "Value-Creating Trades",
        "theory": "Integrative Negotiation (Pruitt 1981)",
        "parent": "address_asymmetric_needs",
        "description": (
            "In your response, look for opportunities where giving costs you little but benefits them greatly, and vice versa. "
            "Ask questions to discover what they value that you can easily provide. "
            "Then propose exchanges: 'This is easy for me to offer; in return, could you help with something that matters to me?' "
            "These asymmetric trades make both parties better off."
        ),
    },
    {
        "id": "address_asymmetric_needs_v4",
        "name": "Bundled Trade-offs",
        "theory": "Package Deal Negotiation",
        "parent": "address_asymmetric_needs",
        "description": (
            "In your response, bundle multiple issues together for trading. "
            "Say 'What if we consider price, timing, and terms as a package?' "
            "Trading across issues creates more value than settling each separately. "
            "Packages allow creative trades that wouldn't be possible one item at a time."
        ),
    },
    {
        "id": "address_asymmetric_needs_v5",
        "name": "Differential Value Discovery",
        "theory": "Value Mapping",
        "parent": "address_asymmetric_needs",
        "description": (
            "In your response, actively explore what each party values differently. "
            "Ask 'What's most important to you here?' and 'What would you trade for that?' "
            "Understanding differential values reveals trade opportunities invisible to surface-level negotiation. "
            "Different preferences are an asset, not an obstacle."
        ),
    },
    {
        "id": "address_asymmetric_needs_v6",
        "name": "Complementary Strengths",
        "theory": "Comparative Advantage",
        "parent": "address_asymmetric_needs",
        "description": (
            "In your response, identify where each party has unique strengths to offer. "
            "Say 'You're better positioned to handle X, while I can more easily provide Y.' "
            "Division of labor based on comparative advantage benefits both parties. "
            "Each contributes what costs them least and values the other most."
        ),
    },
    {
        "id": "address_asymmetric_needs_v7",
        "name": "Customized Solutions",
        "theory": "Tailored Value Creation",
        "parent": "address_asymmetric_needs",
        "description": (
            "In your response, design solutions tailored to each party's specific needs. "
            "Ask 'What would make this work perfectly for you?' and design accordingly. "
            "Cookie-cutter solutions often fail; customized ones address real needs. "
            "Personalization shows respect and often uncovers creative arrangements."
        ),
    },
]

SOCIAL_STRATEGIES: list[dict[str, str]] = [
    {
        "id": "no_strategy",
        "name": "No Strategy",
        "description": "",  # Original bio without modification
    },
    {
        "id": "acknowledge_divergence",
        "name": "Acknowledge Divergent Interests",
        "description": (
            "In your response, recognize that your goals and the other party's goals "
            "may not fully align. Tactfully acknowledge this tension without being "
            "confrontational, and propose adjustments that could bridge your differing "
            "priorities while preserving your core interests."
        ),
    },
    {
        "id": "reframe_opposition",
        "name": "Reframe Points of Opposition",
        "description": (
            "In your response, when you sense resistance or disagreement, subtly "
            "reframe the conflict as a shared challenge rather than opposing positions. "
            "Use phrases like 'I wonder if we might both benefit from...' to transform "
            "potential friction into collaborative problem-solving."
        ),
    },
    {
        "id": "surface_hidden_concerns",
        "name": "Surface Hidden Concerns",
        "description": (
            "In your response, gently probe for unspoken concerns that may be creating "
            "implicit conflict. Ask open-ended questions that invite the other party "
            "to reveal their underlying hesitations, then address these concerns "
            "directly while showing how your proposal accommodates them."
        ),
    },
    {
        "id": "offer_conditional_concessions",
        "name": "Offer Conditional Concessions",
        "description": (
            "In your response, identify areas where you can show flexibility without "
            "compromising your essential goals. Offer measured concessions that signal "
            "good faith while implicitly inviting reciprocity: 'If I were to adjust X, "
            "would you be open to considering Y?'"
        ),
    },
    {
        "id": "highlight_shared_stakes",
        "name": "Highlight Shared Stakes",
        "description": (
            "In your response, emphasize the mutual costs of failing to reach agreement. "
            "Implicitly acknowledge the conflict by illustrating what both parties stand "
            "to lose, then pivot to showing how cooperation serves everyone's interests "
            "better than continued disagreement."
        ),
    },
    {
        "id": "validate_then_redirect",
        "name": "Validate Before Redirecting",
        "description": (
            "In your response, acknowledge the legitimacy of the other party's position "
            "before introducing your perspective. Use validating language that honors "
            "their viewpoint, then smoothly transition with 'At the same time...' or "
            "'Building on that...' to guide the conversation toward common ground."
        ),
    },
    {
        "id": "propose_incremental_steps",
        "name": "Propose Incremental Steps",
        "description": (
            "In your response, recognize that large agreements may be blocked by conflict. "
            "Suggest smaller, low-risk steps that both parties can accept. Frame these "
            "as experiments or trials that preserve everyone's options while building "
            "momentum toward resolution."
        ),
    },
    {
        "id": "address_asymmetric_needs",
        "name": "Address Asymmetric Needs",
        "description": (
            "In your response, acknowledge that you and the other party may have different "
            "priorities and urgencies. Propose creative arrangements that give each side "
            "more of what matters most to them, recognizing that not all interests are "
            "equally weighted for both parties."
        ),
    },
    {
        "id": "normalize_disagreement",
        "name": "Normalize the Disagreement",
        "description": (
            "In your response, frame the current tension as a natural part of any "
            "meaningful negotiation. Use phrases like 'It makes sense that we see this "
            "differently...' to reduce defensiveness and create space for constructive "
            "dialogue about how to move forward together."
        ),
    },
    {
        "id": "future_focus_resolution",
        "name": "Focus on Future Resolution",
        "description": (
            "In your response, shift attention from current conflicts to future outcomes "
            "both parties desire. Ask 'What would success look like for both of us?' "
            "and work backward from that vision to identify paths that transcend "
            "present disagreements."
        ),
    },
    {
        "id": "explicit_tradeoff_analysis",
        "name": "Make Tradeoffs Explicit",
        "description": (
            "In your response, openly acknowledge the tradeoffs each party must consider. "
            "Articulate what you're each giving up and gaining, making the implicit "
            "conflict visible. Then propose ways to rebalance these tradeoffs so the "
            "exchange feels fairer to both sides."
        ),
    },
    {
        "id": "graceful_exit",
        "name": "Graceful Goal Completion Exit",
        "description": (
            "In your response, if you believe your primary goal has been substantially "
            "achieved, prioritize efficiency and social grace. Signal closure with polite "
            "acknowledgment of the interaction, express appreciation for the other party's "
            "cooperation, and propose a natural exit point. Avoid unnecessary prolongation "
            "of the conversation—use phrases like 'I appreciate your help with this' or "
            "'It was good talking with you' to conclude smoothly while maintaining positive "
            "rapport. Remember: achieving your goal efficiently is as valuable as achieving "
            "it at all."
        ),
    },
]


from typing import Literal

StrategyVersion = Literal["v1", "v2", "v3", "v4", "s5", "s6", "s8", "s10", "s20", "s24", "s48", "s50", "s100"]


def get_strategies(strategy_version: StrategyVersion = "v1") -> list[dict[str, str]]:
    """Get the strategy list for the specified version.

    Args:
        strategy_version: "v1" for SOCIAL_STRATEGIES (13 strategies)
                         "v2" for SOCIAL_STRATEGIES_V2 (10 strategies, 4 quadrants)
                         "v3" for SOCIAL_STRATEGIES_V3 (12 strategies, extended)
                         "v4" for SOCIAL_STRATEGIES_V4 (49 strategies, V3 originals + semantic variants)
                         "s5", "s6", "s8", "s10", "s20", "s24", "s48", "s50", "s100" for generated strategy spaces

    Returns:
        The strategy list for the specified version.
    """
    # Check for generated strategy spaces (s5, s10, s20, s50, s100)
    if strategy_version.startswith("s") and strategy_version[1:].isdigit():
        from .strategy_loader import load_generated_strategies
        return load_generated_strategies(strategy_version)

    if strategy_version == "v4":
        return SOCIAL_STRATEGIES_V4
    if strategy_version == "v3":
        return SOCIAL_STRATEGIES_V3
    if strategy_version == "v2":
        return SOCIAL_STRATEGIES_V2
    return SOCIAL_STRATEGIES


def get_strategy_by_id(strategy_id: str, strategy_version: StrategyVersion = "v1") -> dict[str, str] | None:
    """Get a strategy by its ID.

    Args:
        strategy_id: The ID of the strategy to retrieve.
        strategy_version: "v1", "v2", "v3", or "v4"
    """
    strategies = get_strategies(strategy_version)
    for strategy in strategies:
        if strategy["id"] == strategy_id:
            return strategy
    return None


def get_strategy_description(strategy_index: int, strategy_version: StrategyVersion = "v1") -> str:
    """Get the strategy description for a given index.

    Args:
        strategy_index: The index of the strategy.
        strategy_version: "v1", "v2", "v3", or "v4"
    """
    strategies = get_strategies(strategy_version)
    if 0 <= strategy_index < len(strategies):
        return strategies[strategy_index]["description"]
    return ""


def get_num_strategies(strategy_version: StrategyVersion = "v1") -> int:
    """Get the total number of available strategies.

    Args:
        strategy_version: "v1" (13), "v2" (10), "v3" (12), or "v4" (34)
    """
    return len(get_strategies(strategy_version))


def create_enhanced_bio(original_bio: str, strategy_index: int, strategy_version: StrategyVersion = "v1") -> str:
    """
    Create an enhanced bio by appending the selected strategy to the original bio.

    Args:
        original_bio: The agent's original bio text
        strategy_index: Index of the strategy to append (0 = no strategy)
        strategy_version: "v1", "v2", "v3", or "v4"

    Returns:
        Enhanced bio text with strategy appended
    """
    strategies = get_strategies(strategy_version)
    if strategy_index == 0 or strategy_index >= len(strategies):
        return original_bio

    strategy_desc = strategies[strategy_index]["description"]
    if not strategy_desc:
        return original_bio

    # Append strategy to bio with a separator
    return f"{original_bio}\n{strategy_desc} \n If you thinking this strategy is not helpful, please ignore it."


def get_strategy_categories(strategy_version: StrategyVersion = "v3") -> dict[str, list[str]]:
    """Get strategy IDs organized by category (for V3 and V4).

    Returns:
        Dict mapping category names to lists of strategy IDs
    """
    if strategy_version == "v4":
        # V4 = V3 originals + semantic variants (1 baseline + 12 originals + 87 variants = 100 total)
        return {
            "baseline": ["no_strategy"],
            "cooperative": [
                "collaborative_problem_solving",  # original
                "collaborative_problem_solving_v1", "collaborative_problem_solving_v2", "collaborative_problem_solving_v3",
                "collaborative_problem_solving_v4", "collaborative_problem_solving_v5", "collaborative_problem_solving_v6",
                "collaborative_problem_solving_v7", "collaborative_problem_solving_v8",
                "altruistic_concession",  # original
                "altruistic_concession_v1", "altruistic_concession_v2", "altruistic_concession_v3",
                "altruistic_concession_v4", "altruistic_concession_v5", "altruistic_concession_v6",
                "altruistic_concession_v7", "altruistic_concession_v8",
            ],
            "competitive": [
                "pressure_tactics",  # original
                "pressure_tactics_v1", "pressure_tactics_v2", "pressure_tactics_v3",
                "pressure_tactics_v4", "pressure_tactics_v5", "pressure_tactics_v6",
                "pressure_tactics_v7", "pressure_tactics_v8",
            ],
            "strategic": [
                "feign_disinterest",  # original
                "feign_disinterest_v1", "feign_disinterest_v2", "feign_disinterest_v3",
                "feign_disinterest_v4", "feign_disinterest_v5", "feign_disinterest_v6",
                "feign_disinterest_v7",
            ],
            "rational": [
                "logical_tit_for_tat",  # original
                "logical_tit_for_tat_v1", "logical_tit_for_tat_v2", "logical_tit_for_tat_v3",
                "logical_tit_for_tat_v4", "logical_tit_for_tat_v5", "logical_tit_for_tat_v6",
                "logical_tit_for_tat_v7",
                "cost_benefit_persuasion",  # original
                "cost_benefit_persuasion_v1", "cost_benefit_persuasion_v2", "cost_benefit_persuasion_v3",
                "cost_benefit_persuasion_v4", "cost_benefit_persuasion_v5", "cost_benefit_persuasion_v6",
                "cost_benefit_persuasion_v7",
            ],
            "diplomatic": [
                "validate_then_redirect",  # original
                "validate_then_redirect_v1", "validate_then_redirect_v2", "validate_then_redirect_v3",
                "validate_then_redirect_v4", "validate_then_redirect_v5", "validate_then_redirect_v6",
                "validate_then_redirect_v7",
                "normalize_disagreement",  # original
                "normalize_disagreement_v1", "normalize_disagreement_v2", "normalize_disagreement_v3",
                "normalize_disagreement_v4", "normalize_disagreement_v5", "normalize_disagreement_v6",
                "normalize_disagreement_v7",
                "highlight_shared_stakes",  # original
                "highlight_shared_stakes_v1", "highlight_shared_stakes_v2", "highlight_shared_stakes_v3",
                "highlight_shared_stakes_v4", "highlight_shared_stakes_v5", "highlight_shared_stakes_v6",
                "highlight_shared_stakes_v7",
                "propose_incremental_steps",  # original
                "propose_incremental_steps_v1", "propose_incremental_steps_v2", "propose_incremental_steps_v3",
                "propose_incremental_steps_v4", "propose_incremental_steps_v5", "propose_incremental_steps_v6",
                "propose_incremental_steps_v7",
            ],
            "exploratory": [
                "surface_hidden_concerns",  # original
                "surface_hidden_concerns_v1", "surface_hidden_concerns_v2", "surface_hidden_concerns_v3",
                "surface_hidden_concerns_v4", "surface_hidden_concerns_v5", "surface_hidden_concerns_v6",
                "surface_hidden_concerns_v7",
                "address_asymmetric_needs",  # original
                "address_asymmetric_needs_v1", "address_asymmetric_needs_v2", "address_asymmetric_needs_v3",
                "address_asymmetric_needs_v4", "address_asymmetric_needs_v5", "address_asymmetric_needs_v6",
                "address_asymmetric_needs_v7",
            ],
        }

    if strategy_version != "v3":
        return {}

    return {
        "baseline": ["no_strategy"],
        "cooperative": ["collaborative_problem_solving", "altruistic_concession"],
        "competitive": ["firm_stance", "pressure_tactics"],
        "deceptive": ["strategic_withholding", "feign_disinterest"],
        "rational": ["logical_tit_for_tat", "cost_benefit_persuasion"],
        "diplomatic": ["validate_then_redirect", "normalize_disagreement", "highlight_shared_stakes", "propose_incremental_steps"],
        "exploratory": ["surface_hidden_concerns", "address_asymmetric_needs"],
        "utility": ["graceful_exit"],
    }

