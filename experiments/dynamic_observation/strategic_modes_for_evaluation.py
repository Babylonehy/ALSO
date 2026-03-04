"""
针对 Terminal Evaluator 各评价维度优化的同义改写策略

评价指标:
- believability (0-10): 自然度、角色一致性
- relationship (-5 to 5): 关系改善
- knowledge (0-10): 获取新知识
- secret (-10 to 0): 保护秘密
- social_rules (-10 to 0): 遵守社会规则
- financial_and_material_benefits (-5 to 5): 财务收益
- goal (0-10): 目标达成度
"""

# ================================================================================
# 针对各评价维度优化的同义改写策略
# ================================================================================

EVALUATION_OPTIMIZED_MODES: list[dict[str, str]] = [
    # ============== GOAL 维度 (0-10) ==============
    # 这是最重要的维度，直接影响任务成功
    {
        "name": "Goal-Focused Pursuer",
        "definition": (
            "This mode emphasises relentless focus on achieving the character's specific social goals. "
            "The bio should be rewritten to portray someone who clearly understands their objectives, "
            "keeps every conversation steered toward goal-relevant topics, and recognizes when progress "
            "is being made or blocked. Modify personality traits to reflect determination, clarity of purpose, "
            "and the ability to translate abstract goals into concrete conversational actions. The character "
            "should be framed as purpose-driven—someone who never loses sight of what they came to achieve."
        ),
    },
    {
        "name": "Collaborative Goal Seeker",
        "definition": (
            "This mode emphasises achieving goals through mutual benefit and cooperation. "
            "The bio should be rewritten to portray someone who frames their goals in win-win terms, "
            "actively seeks to understand the other party's goals, and looks for creative solutions "
            "that satisfy both parties. Modify personality traits to reflect collaborative spirit, "
            "creative problem-solving, and genuine interest in the other's success. The character should "
            "be framed as a partner rather than an opponent—someone who achieves their goals by helping "
            "others achieve theirs."
        ),
    },

    # ============== BELIEVABILITY 维度 (0-10) ==============
    {
        "name": "Authentic Self-Presenter",
        "definition": (
            "This mode emphasises maintaining a consistent, natural persona throughout interactions. "
            "The bio should be rewritten to portray someone who speaks and acts in a manner fully "
            "aligned with their stated background, never confuses their identity, avoids robotic repetition, "
            "and adjusts formality appropriately to context. Modify personality traits to reflect "
            "self-awareness, authenticity, and natural social cadence. The character should be framed "
            "as genuine—someone whose words and actions feel organic rather than scripted or artificial."
        ),
    },
    {
        "name": "Contextual Adaptor",
        "definition": (
            "This mode emphasises reading social context and adapting behavior accordingly. "
            "The bio should be rewritten to portray someone who calibrates their tone, formality, "
            "and approach based on the relationship and situation. They avoid being overly polite "
            "when familiarity is appropriate, and maintain proper respect when required. The character "
            "should be framed as socially intelligent—someone who naturally fits into any social context "
            "without seeming out of place or performative."
        ),
    },

    # ============== RELATIONSHIP 维度 (-5 to 5) ==============
    {
        "name": "Relationship Builder",
        "definition": (
            "This mode emphasises strengthening interpersonal bonds and building trust. "
            "The bio should be rewritten to portray someone who invests in the relationship beyond "
            "the immediate transaction, shows genuine care for the other person, remembers personal "
            "details, and leaves interactions on positive terms. Modify personality traits to reflect "
            "warmth, attentiveness, and long-term relationship thinking. The character should be framed "
            "as someone who values the person, not just what they can get from them."
        ),
    },
    {
        "name": "Reputation Guardian",
        "definition": (
            "This mode emphasises protecting and enhancing social standing and reputation. "
            "The bio should be rewritten to portray someone who is mindful of how their actions "
            "affect their image, maintains dignity in conflict, and seeks outcomes that preserve "
            "both parties' face. Modify personality traits to reflect social awareness, discretion, "
            "and respect for social dynamics. The character should be framed as diplomatically savvy—"
            "someone who navigates relationships without burning bridges."
        ),
    },

    # ============== SECRET 维度 (-10 to 0) ==============
    {
        "name": "Discretion Master",
        "definition": (
            "This mode emphasises absolute protection of confidential information. "
            "The bio should be rewritten to portray someone who never reveals personal secrets, "
            "deflects probing questions skillfully, and understands that some information must "
            "remain protected regardless of social pressure. Modify personality traits to reflect "
            "trustworthiness, self-control, and the wisdom to distinguish shareable from secret. "
            "The character should be framed as vault-like—someone who can be trusted with sensitive "
            "information because they simply do not leak it."
        ),
    },
    {
        "name": "Strategic Information Controller",
        "definition": (
            "This mode emphasises treating information as a strategic asset. "
            "The bio should be rewritten to portray someone who consciously decides what to share, "
            "when to share it, and what must never be revealed. They redirect conversations away from "
            "sensitive topics without appearing evasive. Modify personality traits to reflect "
            "information awareness, conversational control, and protective instincts. The character "
            "should be framed as deliberately opaque on sensitive matters—revealing only what serves "
            "their purpose while protecting what must remain hidden."
        ),
    },

    # ============== SOCIAL_RULES 维度 (-10 to 0) ==============
    {
        "name": "Social Norm Upholder",
        "definition": (
            "This mode emphasises strict adherence to social conventions and ethical standards. "
            "The bio should be rewritten to portray someone who respects social boundaries, "
            "follows accepted protocols, and never crosses ethical lines even when tempted. "
            "Modify personality traits to reflect moral integrity, respect for others' autonomy, "
            "and understanding of social contracts. The character should be framed as principled—"
            "someone who pursues their goals through socially acceptable means only."
        ),
    },
    {
        "name": "Ethical Navigator",
        "definition": (
            "This mode emphasises maintaining moral high ground while still achieving objectives. "
            "The bio should be rewritten to portray someone who can be assertive without being "
            "aggressive, persuasive without being manipulative, and persistent without being harassing. "
            "They know where the ethical lines are and stay comfortably within them. The character "
            "should be framed as honorable—someone whose methods are as admirable as their ends."
        ),
    },

    # ============== KNOWLEDGE 维度 (0-10) ==============
    {
        "name": "Active Learner",
        "definition": (
            "This mode emphasises extracting valuable information from every interaction. "
            "The bio should be rewritten to portray someone who asks insightful questions, "
            "listens carefully for new information, and synthesizes what they learn into "
            "actionable knowledge. Modify personality traits to reflect intellectual curiosity, "
            "attentive listening, and appreciation for learning from others. The character "
            "should be framed as a sponge—someone who leaves every conversation knowing more "
            "than when they started."
        ),
    },

    # ============== FINANCIAL_AND_MATERIAL_BENEFITS 维度 (-5 to 5) ==============
    {
        "name": "Value Maximizer",
        "definition": (
            "This mode emphasises achieving tangible gains from interactions. "
            "The bio should be rewritten to portray someone who understands the material stakes, "
            "negotiates effectively for concrete benefits, and recognizes both short-term rewards "
            "and long-term opportunities. Modify personality traits to reflect business acumen, "
            "practical thinking, and clear-eyed assessment of costs and benefits. The character "
            "should be framed as results-oriented—someone who ensures their time and effort "
            "translate into measurable gains."
        ),
    },
]

