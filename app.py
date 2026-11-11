from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


APP_TITLE = "Bank AI Compliance Assistant"
APP_SUBTITLE = "银行 AI 应用合规风险智能评估助手"
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_FILE = BASE_DIR / "sample_cases.json"
LEGAL_REVIEW_DATE = "2026-11-10"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODELS = ["deepseek-flash", "deepseek-v4-pro"]
DEEPSEEK_DOC_URL = "https://api-docs.deepseek.com/quick_start/agent_integrations/"

DISCLAIMER_SHORT = (
    "仅供模拟评估、面试演示和内部初筛；不构成法律意见、监管结论或上线审批依据。"
)
APPLICABILITY_NOTICE = (
    "规则仅反映截至 2026 年 11 月 10 日核验的部分公开法规要求。具体适用性取决于业务场景、"
    "主体角色、数据分类分级、数据流向、服务对象及服务范围，应由法务、合规、数据安全等专业人员复核。"
)

LEGAL_SOURCES = [
    {
        "name": "《中华人民共和国个人信息保护法》",
        "url": "https://www.cac.gov.cn/2021-08/20/c_1631050028355286.htm",
        "publisher": "国家互联网信息办公室（来源：中国人大网）",
    },
    {
        "name": "《中华人民共和国数据安全法》",
        "url": "https://flk.npc.gov.cn/detail?fileId=&id=ff80818179f5e0800179f885c7e70392&title=%E4%B8%AD%E5%8D%8E%E4%BA%BA%E6%B0%91%E5%85%B1%E5%92%8C%E5%9B%BD%E6%95%B0%E6%8D%AE%E5%AE%89%E5%85%A8%E6%B3%95&type=",
        "publisher": "国家法律法规数据库",
    },
    {
        "name": "《生成式人工智能服务管理暂行办法》",
        "url": "https://www.cac.gov.cn/2023-07/13/c_1690898327029107.htm",
        "publisher": "国家互联网信息办公室",
    },
]

LEVEL_BANDS = [
    (24, "低", "LOW", "#0f766e"),
    (49, "中", "MEDIUM", "#b7791f"),
    (74, "高", "HIGH", "#c2410c"),
    (100, "极高", "VERY HIGH", "#b91c1c"),
]

BOOLEAN_FIELDS = [
    "generative_ai",
    "personal_financial_info",
    "sensitive_personal_info",
    "identifiable_customer_input",
    "automated_decision",
    "material_rights_impact",
    "third_party_model",
    "cross_border_processing",
    "important_data",
    "cii_operator",
    "human_review",
    "veto_control",
    "prompt_logging",
    "output_logging",
    "approval_logging",
    "pipia_completed",
    "fact_verification",
    "rag_citations",
    "fairness_test",
    "customer_explanation",
    "appeal_correction_channel",
    "field_whitelist",
    "masking",
    "environment_isolation",
    "vendor_due_diligence",
    "data_processing_agreement",
    "training_prohibited",
    "audit_right",
    "deletion_certificate",
    "red_team_test",
    "version_management",
    "safety_assessment_filing",
]

TEXT_FIELDS = [
    "name",
    "department",
    "stage",
    "description",
    "audience",
    "public_service_scope",
    "public_opinion_capability",
]

DEFAULT_CASE = {
    "name": "待评估 AI 应用",
    "department": "业务部门",
    "stage": "方案设计",
    "description": "请描述模型用途、输入输出、数据流和人工决策边界。",
    "audience": "员工",
    "generative_ai": True,
    "public_service_scope": "待判断",
    "public_opinion_capability": "待判断",
    "personal_financial_info": False,
    "sensitive_personal_info": False,
    "identifiable_customer_input": False,
    "automated_decision": False,
    "material_rights_impact": False,
    "third_party_model": False,
    "cross_border_processing": False,
    "important_data": False,
    "cii_operator": False,
    "human_review": True,
    "veto_control": True,
    "prompt_logging": True,
    "output_logging": True,
    "approval_logging": True,
    "pipia_completed": False,
    "fact_verification": True,
    "rag_citations": True,
    "fairness_test": True,
    "customer_explanation": True,
    "appeal_correction_channel": True,
    "field_whitelist": True,
    "masking": True,
    "environment_isolation": True,
    "vendor_due_diligence": True,
    "data_processing_agreement": True,
    "training_prohibited": True,
    "audit_right": True,
    "deletion_certificate": True,
    "red_team_test": True,
    "version_management": True,
    "safety_assessment_filing": False,
}


def bool_text(value: bool) -> str:
    return "是" if value else "否"


def has_personal_data(case: dict) -> bool:
    return any(
        case.get(key, False)
        for key in (
            "personal_financial_info",
            "sensitive_personal_info",
            "identifiable_customer_input",
        )
    )


def is_customer_facing(case: dict) -> bool:
    return case.get("audience") in {"客户", "混合"}


def genai_rules_may_apply(case: dict) -> bool:
    return bool(case.get("generative_ai")) and case.get("public_service_scope") != "否"


def missing_labels(case: dict, field_labels: dict[str, str]) -> list[str]:
    return [label for key, label in field_labels.items() if not case.get(key, False)]


def logging_fact(case: dict) -> str:
    missing = missing_labels(
        case,
        {
            "prompt_logging": "提示词日志",
            "output_logging": "输出日志",
            "approval_logging": "审批/人工复核日志",
        },
    )
    return "审计留痕存在缺口：" + "、".join(missing) + "未落实。"


def hallucination_fact(case: dict) -> str:
    missing = missing_labels(
        case,
        {"fact_verification": "事实核验/幻觉拦截", "rag_citations": "RAG 来源引用"},
    )
    return "高影响或客户侧生成内容缺少：" + "、".join(missing) + "。"


def rights_channel_fact(case: dict) -> str:
    missing = missing_labels(
        case,
        {
            "customer_explanation": "客户解释机制",
            "appeal_correction_channel": "申诉/更正渠道",
        },
    )
    return "客户权利保障存在缺口：" + "、".join(missing) + "未落实。"


def data_controls_fact(case: dict) -> str:
    missing = missing_labels(
        case,
        {
            "field_whitelist": "数据字段白名单",
            "masking": "脱敏/去标识化",
            "environment_isolation": "环境与权限隔离",
        },
    )
    return "个人信息安全控制存在缺口：" + "、".join(missing) + "未落实。"


def vendor_controls_fact(case: dict) -> str:
    missing = missing_labels(
        case,
        {
            "vendor_due_diligence": "供应商尽调",
            "data_processing_agreement": "数据处理协议",
            "training_prohibited": "训练禁用约定",
            "audit_right": "审计权",
            "deletion_certificate": "删除/返还证明",
        },
    )
    return "第三方治理条款存在缺口：" + "、".join(missing) + "未落实。"


def important_data_fact(case: dict) -> str:
    operator = "已标识为关键信息基础设施运营者" if case.get("cii_operator") else "关基运营者身份未勾选"
    return f"项目勾选涉及重要数据且存在境外处理；{operator}，需按实际主体及数据目录进一步判定。"


def filing_fact(case: dict) -> str:
    scope = case.get("public_service_scope", "待判断")
    capability = case.get("public_opinion_capability", "待判断")
    return (
        f"面向境内公众服务范围：{scope}；舆论属性或社会动员能力：{capability}；"
        "安全评估/算法备案状态未确认完成。"
    )


def pipia_required(case: dict) -> bool:
    personal = has_personal_data(case)
    return bool(
        case.get("sensitive_personal_info")
        or case.get("automated_decision")
        or (case.get("identifiable_customer_input") and case.get("third_party_model"))
        or (case.get("identifiable_customer_input") and case.get("cross_border_processing"))
        or (personal and case.get("material_rights_impact"))
    )


@dataclass(frozen=True)
class RiskRule:
    rule_id: str
    dimension: str
    points: int
    trigger_text: str
    condition: Callable[[dict], bool]
    fact: Callable[[dict], str]
    risk: str
    legal_basis: str
    remediation: str
    english_title: str


RULES: list[RiskRule] = [
    RiskRule(
        "R01",
        "个人金融信息",
        4,
        "处理个人金融信息",
        lambda c: bool(c.get("personal_financial_info")),
        lambda c: "项目确认处理个人金融信息；该类别并不当然全部等同于敏感个人信息，需按字段逐项分类。",
        "金融场景数据通常与客户权益高度相关，字段范围、用途和保存期限需要最小化。",
        "《个人信息保护法》第6条（目的明确、直接相关、最小范围）及第51条（分类、加密/去标识化、权限等措施）。",
        "建立字段级数据清单和白名单，仅传递业务必需字段；默认脱敏/去标识化，并与生产核心数据区隔离。",
        "Personal financial information",
    ),
    RiskRule(
        "R02",
        "敏感个人信息",
        7,
        "处理敏感个人信息",
        lambda c: bool(c.get("sensitive_personal_info")),
        lambda c: "项目确认处理敏感个人信息；金融账户、生物识别、医疗健康、行踪轨迹等属于法定列举类型。",
        "泄露或非法使用可能对人格尊严、人身或财产安全造成较高影响，需证明特定目的与充分必要性。",
        "《个人信息保护法》第28—30条（敏感信息定义、特定目的和充分必要、严格保护、单独同意及增强告知）及第55—56条（PIPIA）。",
        "逐字段确认处理基础、单独同意或其他法定情形及增强告知；采用强加密、分级授权、全链路审计并完成PIPIA。",
        "Sensitive personal information",
    ),
    RiskRule(
        "R03",
        "可识别客户信息入模",
        7,
        "向模型输入可识别客户信息",
        lambda c: bool(c.get("identifiable_customer_input")),
        lambda c: "提示词、上下文、附件或检索结果中会向模型传入可识别客户信息。",
        "输入内容可能被外部处理、记录或超范围利用，且自由文本容易夹带非必要字段。",
        "《个人信息保护法》第6、21、23、51条；《生成式人工智能服务管理暂行办法》第11条仅在该办法适用且属于提供者时关注输入与使用记录保护。",
        "在模型网关实施字段白名单、DLP检测和自动脱敏；阻断身份证号、账号等原文入模，并核验处理角色、日志期限和删除机制。",
        "Identifiable customer data in model input",
    ),
    RiskRule(
        "R04",
        "自动化决策",
        6,
        "利用个人信息进行自动化决策",
        lambda c: bool(c.get("automated_decision")),
        lambda c: "模型输出会直接或实质性参与自动化决策。",
        "需要保证决策透明、公平、公正，并验证是否存在不合理差别待遇。",
        "《个人信息保护法》第24条（自动化决策透明、公平、公正及重大影响决定的权利保障）及第55—56条（PIPIA）。",
        "把模型定位为可审查的决策支持；记录特征、版本、阈值与理由码，定期开展公平性、稳定性和漂移测试。",
        "Automated decision-making",
    ),
    RiskRule(
        "R05",
        "客户重大权益",
        9,
        "影响授信、定价、准入等重大权益",
        lambda c: bool(c.get("material_rights_impact")),
        lambda c: "模型结果会影响授信、定价、准入或其他客户重大权益。",
        "错误、偏见或不可解释输出可能放大为实际拒绝、差别定价或准入限制。",
        "若涉及个人信息自动化决策，重点关注《个人信息保护法》第24条；存在对个人权益重大影响的个人信息处理时关注第55—56条。",
        "设置人工实质复核和一票否决；禁止模型单独作出最终决定，向客户提供清晰理由、人工申诉与更正路径。",
        "Material impact on customer rights",
    ),
    RiskRule(
        "R06",
        "第三方大模型/API",
        5,
        "使用第三方大模型或API",
        lambda c: bool(c.get("third_party_model")),
        lambda c: "项目依赖第三方大模型、API或外部模型服务。",
        "供应商的处理角色、转委托、训练使用、留存位置和退出删除能力可能不透明。",
        "《个人信息保护法》第21条适用于委托处理；第23条适用于向其他个人信息处理者提供。二者须按实际决定权与数据流分类，不能机械并用。",
        "完成模型供应商尽调和角色判定；签署数据处理协议，明确用途/期限、训练禁用、转委托限制、审计权、事件协助及返还删除证明。",
        "Third-party model or API",
    ),
    RiskRule(
        "R07",
        "境外数据处理",
        9,
        "发生境外存储、访问、传输或处理",
        lambda c: bool(c.get("cross_border_processing")),
        lambda c: "项目确认存在境外存储、远程访问、传输或其他境外处理链路。",
        "需要先确认是否构成个人信息或重要数据出境，并基于主体、数据类型、数量和豁免情形选择合规路径。",
        "个人信息出境关注《个人信息保护法》第38—39条；并非所有跨境均当然必须安全评估，须结合现行出境制度判断。",
        "绘制端到端数据流和接收方清单，完成跨境数据合规评估；按适用路径落实安全评估、认证、标准合同或其他条件及专项告知/同意。",
        "Cross-border data processing",
    ),
    RiskRule(
        "R08",
        "重要数据出境",
        5,
        "境外处理且勾选涉及重要数据",
        lambda c: bool(c.get("cross_border_processing") and c.get("important_data")),
        important_data_fact,
        "重要数据目录认定和主体身份会改变出境管理要求，不能仅按普通个人信息路径处理。",
        "《数据安全法》第31条仅针对境内运营中收集、产生的重要数据出境：关基运营者适用网络安全法，其他处理者适用国家有关办法。",
        "由数据安全和业务条线确认重要数据目录映射、关基身份和出境路径，形成分类分级、风险评估及主管部门沟通记录。",
        "Important data export",
    ),
    RiskRule(
        "R09",
        "人工复核与一票否决",
        8,
        "重大权益场景缺少人工实质复核或一票否决",
        lambda c: bool(
            c.get("material_rights_impact")
            and (not c.get("human_review") or not c.get("veto_control"))
        ),
        lambda c: (
            "重大权益场景中缺少"
            + "、".join(
                missing_labels(c, {"human_review": "人工实质复核", "veto_control": "一票否决控制"})
            )
            + "。"
        ),
        "形式化人工点击无法纠正模型错误；需要让有权限、有信息的人员能够独立判断并推翻模型建议。",
        "《个人信息保护法》第24条明确重大影响自动化决定的说明权及拒绝仅由自动化决策作出决定；人工复核/一票否决是支持落实的治理措施，并非条文字面统一要求。",
        "设置独立复核岗、明确触发阈值和升级路径；保留模型建议、人工理由及否决记录，并抽检复核质量。",
        "Missing human review or veto",
    ),
    RiskRule(
        "R10",
        "提示词、输出与审批日志",
        4,
        "AI/决策场景审计日志不完整",
        lambda c: bool(
            (c.get("generative_ai") or c.get("automated_decision") or c.get("material_rights_impact"))
            and any(
                not c.get(key)
                for key in ("prompt_logging", "output_logging", "approval_logging")
            )
        ),
        logging_fact,
        "无法重放关键链路、解释决策或定位事件；但全量保留可识别输入同样可能带来过度留存风险。",
        "《个人信息保护法》第51、55—56条；《生成式人工智能服务管理暂行办法》第11条在适用范围内禁止非法留存可识别身份的输入和使用记录。",
        "按必要性留存提示词、输出、审批和模型版本；先脱敏/去标识化，设置访问权限、保存期限与删除任务，避免以审计为由无限期全量留存。",
        "Incomplete prompt, output or approval logs",
    ),
    RiskRule(
        "R11",
        "个人信息保护影响评估（PIPIA）",
        8,
        "命中法定评估场景但未完成PIPIA",
        lambda c: bool(pipia_required(c) and not c.get("pipia_completed")),
        lambda c: "基于当前勾选的数据链路，处理活动命中或高度可能命中敏感信息、个人信息自动化决策、委托/提供、出境或重大影响等至少一种情形，但PIPIA未完成；仍需按真实数据流核实。",
        "缺少事前、可审计的必要性、权益影响、风险和措施有效性论证。",
        "《个人信息保护法》第55条规定应事前开展PIPIA的情形；第56条规定评估内容，报告和处理记录至少保存三年。",
        "上线前完成PIPIA并纳入审批门禁，覆盖合法正当必要性、权益影响、安全风险和控制有效性；报告及记录依法至少保存三年。",
        "Missing PIPIA",
    ),
    RiskRule(
        "R12",
        "大模型幻觉与事实核验",
        5,
        "高影响/客户侧生成内容缺少事实核验或来源引用",
        lambda c: bool(
            c.get("generative_ai")
            and (is_customer_facing(c) or c.get("material_rights_impact"))
            and (not c.get("fact_verification") or not c.get("rag_citations"))
        ),
        hallucination_fact,
        "无依据或过时内容可能被误用为业务事实、客户承诺或决策理由。",
        "《生成式人工智能服务管理暂行办法》第4条仅在该办法适用范围内要求依服务特点提升透明度、准确性和可靠性；内部场景可作为治理参考。",
        "采用受控RAG并强制展示来源、版本和生效日期；设置幻觉拦截、置信度阈值、关键事实二次核验、红队测试及高风险答案转人工。",
        "Missing hallucination controls",
    ),
    RiskRule(
        "R13",
        "算法偏见与公平性",
        4,
        "自动化或可能适用的公众生成式服务未做公平性测试",
        lambda c: bool(
            (c.get("automated_decision") or genai_rules_may_apply(c))
            and not c.get("fairness_test")
        ),
        lambda c: "项目存在自动化决策或尚未排除面向境内公众的生成式服务，但未完成算法偏见/公平性测试。",
        "群体间错误率、通过率或定价差异可能未被发现，导致不合理差别待遇或歧视性结果。",
        "自动化决策关注《个人信息保护法》第24条；《生成式人工智能服务管理暂行办法》第4条仅在其适用范围内要求采取措施防止相关歧视。",
        "定义受保护群体和公平性指标，比较通过率、误拒率、误受率及校准差异；开展上线前、版本变更后和持续监测，并记录纠偏。",
        "Missing fairness testing",
    ),
    RiskRule(
        "R14",
        "客户解释、申诉与更正",
        5,
        "客户/重大影响场景的解释或权利渠道不完整",
        lambda c: bool(
            (
                c.get("automated_decision")
                or c.get("material_rights_impact")
                or (is_customer_facing(c) and has_personal_data(c))
            )
            and (not c.get("customer_explanation") or not c.get("appeal_correction_channel"))
        ),
        rights_channel_fact,
        "客户可能无法理解不利结果、纠正错误数据或获得有意义的人工复核。",
        "在处理个人信息或作出重大影响自动化决定的前提下，关注《个人信息保护法》第24条（说明权）；第46条（更正、补充）和第50条（便捷的权利申请受理机制）可作为渠道设计的补充依据。",
        "提供理由码和通俗说明，建立人工申诉、数据更正与复核SLA；把申诉结果用于模型与流程纠偏。",
        "Missing explanation, appeal or correction channel",
    ),
    RiskRule(
        "R15",
        "数据安全工程控制",
        5,
        "处理个人信息但白名单/脱敏/隔离控制不完整",
        lambda c: bool(
            has_personal_data(c)
            and any(
                not c.get(key)
                for key in ("field_whitelist", "masking", "environment_isolation")
            )
        ),
        data_controls_fact,
        "个人信息可能被超范围调用、明文暴露或在开发测试与生产之间扩散。",
        "《个人信息保护法》第6条及第51条（内部制度、分类管理、加密/去标识化、权限控制、培训和事件预案）；第57条仅在发生或可能发生泄露、篡改、丢失时要求立即补救并依法通知，且个人通知存在法定例外。",
        "在AI网关执行字段白名单与DLP；默认脱敏/令牌化，开发测试使用合成数据，模型区与核心业务区网络/账号/密钥隔离；建立事件检测、立即补救、通知判断和全程留痕机制。",
        "Missing data security controls",
    ),
    RiskRule(
        "R16",
        "模型供应商治理",
        5,
        "使用第三方模型但关键尽调/合同控制不完整",
        lambda c: bool(
            c.get("third_party_model")
            and any(
                not c.get(key)
                for key in (
                    "vendor_due_diligence",
                    "data_processing_agreement",
                    "training_prohibited",
                    "audit_right",
                    "deletion_certificate",
                )
            )
        ),
        vendor_controls_fact,
        "合同和技术边界不足会削弱对训练使用、转委托、事件响应及退出删除的控制。",
        "根据实际角色关注《个人信息保护法》第21条（委托处理约定、监督、返还/删除、转委托）或第23条（向其他处理者提供的告知与单独同意等）。",
        "补齐供应商安全/合规尽调、数据处理协议、训练禁用、转委托审批、审计权、事件通报、退出迁移和删除证明。",
        "Incomplete vendor governance",
    ),
    RiskRule(
        "R17",
        "公众生成式服务评估/备案",
        4,
        "未排除公众服务及舆论/社会动员属性且状态未完成",
        lambda c: bool(
            c.get("generative_ai")
            and c.get("public_service_scope") != "否"
            and c.get("public_opinion_capability") != "否"
            and not c.get("safety_assessment_filing")
        ),
        filing_fact,
        "若实际同时满足适用范围和属性条件，可能需要按规定开展安全评估并履行算法备案手续。",
        "《生成式人工智能服务管理暂行办法》第2条限定适用范围；第17条仅针对具有舆论属性或者社会动员能力的生成式人工智能服务。",
        "由法务/网信合规确认是否向境内公众提供服务及是否具有相关属性；命中时按规定完成安全评估、算法备案及变更/注销管理。",
        "Unresolved public GenAI assessment or filing",
    ),
]


BASELINE_REMEDIATIONS = [
    "数据最小化：建立字段白名单，在模型网关实施脱敏/去标识化、DLP拦截，并隔离模型区与核心数据区。",
    "决策治理：重大权益场景设置有能力、有权限的人工实质复核与一票否决，禁止模型单独作最终决定。",
    "供应商治理：完成尽调和处理角色判定，以数据处理协议落实训练禁用、转委托限制、审计权、事件协助和删除证明。",
    "跨境治理：绘制境内外数据流，识别个人信息/重要数据及主体属性，按现行规则选择适用的跨境合规路径。",
    "PIPIA：命中法定情形时上线前完成评估，论证必要性、权益影响、风险与措施有效性，并依法保存记录。",
    "大模型可靠性：受控RAG强制来源引用，设置幻觉拦截、关键事实核验、红队测试、版本管理和必要且合规的日志。",
    "公平与权利：开展群体公平性测试，输出可理解的理由，并建立申诉、人工复核、数据更正和反馈闭环。",
    "事件响应：建立个人信息泄露、篡改、丢失的检测、立即补救、通知判断与留痕机制，按法定条件处理部门及个人通知。",
]


def evaluate_case(case: dict) -> dict:
    triggered = []
    for rule in RULES:
        if rule.condition(case):
            triggered.append(
                {
                    "规则ID": rule.rule_id,
                    "风险维度": rule.dimension,
                    "分值": rule.points,
                    "事实判断": rule.fact(case),
                    "风险说明": rule.risk,
                    "法律或监管依据": rule.legal_basis,
                    "整改建议": rule.remediation,
                    "英文标题": rule.english_title,
                }
            )
    raw_score = sum(item["分值"] for item in triggered)
    score = min(100, raw_score)
    level, level_en, color = risk_level(score)
    return {
        "score": score,
        "raw_score": raw_score,
        "level": level,
        "level_en": level_en,
        "color": color,
        "triggered": triggered,
        "trigger_count": len(triggered),
    }


def risk_level(score: int) -> tuple[str, str, str]:
    for upper, label, label_en, color in LEVEL_BANDS:
        if score <= upper:
            return label, label_en, color
    return "极高", "VERY HIGH", "#b91c1c"


def validate_case(case: dict) -> list[str]:
    notes: list[str] = []
    if case.get("identifiable_customer_input") and not (
        case.get("personal_financial_info") or case.get("sensitive_personal_info")
    ):
        notes.append("已勾选“可识别客户信息入模”，但未勾选个人金融/敏感信息；请确认是否遗漏数据分类。")
    if case.get("sensitive_personal_info") and not case.get("pipia_completed"):
        notes.append("敏感个人信息处理已触发PIPIA关注；当前未勾选完成。")
    if case.get("cross_border_processing") and case.get("public_service_scope") == "否":
        notes.append("内部应用也可能发生数据出境；“仅内部使用”不代表跨境规则当然不适用。")
    if case.get("material_rights_impact") and not case.get("automated_decision"):
        notes.append("重大权益影响已勾选但自动化决策未勾选；若模型仅提供建议，该组合可以成立，请确认最终决策边界。")
    if case.get("important_data") and not case.get("cross_border_processing"):
        notes.append("已勾选重要数据但未勾选境外处理；本工具不会触发第31条出境规则，但仍需执行重要数据安全治理。")
    return notes


def normalize_case(case: dict) -> dict:
    normalized = DEFAULT_CASE.copy()
    for key in TEXT_FIELDS:
        if key in case:
            normalized[key] = str(case[key])
    for key in BOOLEAN_FIELDS:
        if key in case:
            normalized[key] = bool(case[key])
    return normalized


def load_sample_cases() -> list[dict]:
    if not SAMPLE_FILE.exists():
        return [DEFAULT_CASE.copy()]
    try:
        data = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"样例文件读取失败：{exc}")
        return [DEFAULT_CASE.copy()]
    if isinstance(data, dict):
        data = data.get("cases", [])
    if not isinstance(data, list) or not data:
        st.error("sample_cases.json 必须是非空数组，或包含非空 cases 数组。")
        return [DEFAULT_CASE.copy()]
    normalized = [normalize_case(item) for item in data if isinstance(item, dict)]
    if not normalized:
        st.error("sample_cases.json 未包含有效对象，已回退到默认案例。")
        return [DEFAULT_CASE.copy()]
    return normalized


def case_from_state() -> dict:
    return {key: st.session_state.get(key, DEFAULT_CASE.get(key)) for key in TEXT_FIELDS + BOOLEAN_FIELDS}


def load_case_into_state(case: dict) -> None:
    normalized = normalize_case(case)
    for key, value in normalized.items():
        st.session_state[key] = value


def html_report(case: dict, result: dict) -> bytes:
    esc = lambda value: html.escape(str(value), quote=True)
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S %Z")
    rows = []
    for item in result["triggered"]:
        rows.append(
            "<tr>"
            f"<td>{esc(item['规则ID'])}</td>"
            f"<td>{esc(item['风险维度'])}</td>"
            f"<td class='points'>+{esc(item['分值'])}</td>"
            f"<td>{esc(item['事实判断'])}</td>"
            f"<td>{esc(item['风险说明'])}</td>"
            f"<td>{esc(item['法律或监管依据'])}</td>"
            f"<td>{esc(item['整改建议'])}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='7'>当前输入未触发加分规则；低风险不等于无风险，仍需维持基础控制。</td></tr>")

    facts = [
        ("项目名称", case.get("name", "")),
        ("业务部门", case.get("department", "")),
        ("项目阶段", case.get("stage", "")),
        ("服务对象", case.get("audience", "")),
        ("生成式AI", bool_text(bool(case.get("generative_ai")))),
        ("面向境内公众", case.get("public_service_scope", "待判断")),
        ("自动化决策", bool_text(bool(case.get("automated_decision")))),
        ("重大权益影响", bool_text(bool(case.get("material_rights_impact")))),
        ("第三方模型/API", bool_text(bool(case.get("third_party_model")))),
        ("境外处理", bool_text(bool(case.get("cross_border_processing")))),
    ]
    facts_html = "".join(
        f"<div class='fact'><span>{esc(label)}</span><strong>{esc(value)}</strong></div>"
        for label, value in facts
    )
    remediation_html = "".join(f"<li>{esc(item)}</li>" for item in BASELINE_REMEDIATIONS)
    source_html = "".join(
        f"<li><a href='{esc(source['url'])}'>{esc(source['name'])}</a> — {esc(source['publisher'])}</li>"
        for source in LEGAL_SOURCES
    )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{esc(case.get('name', 'AI应用'))}｜合规风险模拟评估报告</title>
  <style>
    @page {{ size: A4 landscape; margin: 14mm; }}
    :root {{ --navy:#0b1f33; --teal:#0f766e; --line:#dbe4ea; --ink:#182532; --muted:#5c6b78; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); font:14px/1.65 "Microsoft YaHei", "PingFang SC", Arial, sans-serif; background:#f4f7f9; }}
    .page {{ max-width:1400px; margin:24px auto; background:#fff; padding:28px 32px; box-shadow:0 8px 30px rgba(11,31,51,.08); }}
    .hero {{ background:linear-gradient(120deg,#0b1f33,#123b52); color:#fff; padding:26px 30px; border-radius:12px; }}
    h1 {{ margin:0 0 4px; font-size:28px; }} h2 {{ margin:28px 0 12px; color:var(--navy); font-size:19px; }}
    .subtitle {{ color:#c8d9e5; }} .disclaimer {{ margin:18px 0; padding:14px 16px; background:#fff7ed; border-left:5px solid #c2410c; font-weight:700; }}
    .notice {{ color:var(--muted); }} .score {{ display:flex; gap:20px; align-items:center; margin:20px 0; }}
    .score-badge {{ width:116px; height:116px; border-radius:58px; color:#fff; display:flex; flex-direction:column; justify-content:center; align-items:center; background:{esc(result['color'])}; }}
    .score-badge strong {{ font-size:38px; line-height:1; }} .score-badge span {{ font-size:13px; margin-top:7px; }}
    .facts {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; }} .fact {{ border:1px solid var(--line); padding:10px 12px; border-radius:8px; }}
    .fact span {{ display:block; color:var(--muted); font-size:12px; }} .fact strong {{ display:block; margin-top:3px; }}
    table {{ border-collapse:collapse; width:100%; font-size:11.5px; table-layout:fixed; }} th {{ background:var(--navy); color:#fff; padding:8px; text-align:left; }}
    td {{ border:1px solid var(--line); padding:8px; vertical-align:top; overflow-wrap:anywhere; }} tr:nth-child(even) {{ background:#f8fafb; }} .points {{ font-weight:700; color:#b91c1c; text-align:center; }}
    ol,ul {{ padding-left:22px; }} a {{ color:#0b7285; }} .footer {{ margin-top:28px; padding-top:14px; border-top:1px solid var(--line); color:var(--muted); font-size:12px; }}
    @media print {{ body {{ background:#fff; }} .page {{ margin:0; padding:0; box-shadow:none; max-width:none; }} a {{ color:inherit; text-decoration:none; }} }}
  </style>
</head>
<body><main class="page">
  <section class="hero"><h1>{APP_TITLE}</h1><div class="subtitle">{APP_SUBTITLE}｜完整模拟评估报告</div></section>
  <div class="disclaimer">{esc(DISCLAIMER_SHORT)}</div>
  <p class="notice">{esc(APPLICABILITY_NOTICE)}</p>
  <section class="score">
    <div class="score-badge"><strong>{esc(result['score'])}</strong><span>/ 100</span></div>
    <div><h2 style="margin:0">风险等级：{esc(result['level'])}</h2><p>触发 {esc(result['trigger_count'])} 条规则；逐条固定分值直接相加，无隐藏系数。全部规则权重合计100分。</p><p>报告生成时间：{esc(generated_at)}</p></div>
  </section>
  <h2>一、项目事实快照</h2><div class="facts">{facts_html}</div>
  <p><strong>场景描述：</strong>{esc(case.get('description', ''))}</p>
  <h2>二、触发风险明细</h2>
  <table><thead><tr><th style="width:5%">ID</th><th style="width:9%">风险维度</th><th style="width:5%">分值</th><th style="width:17%">事实判断</th><th style="width:15%">风险说明</th><th style="width:22%">法律或监管依据</th><th style="width:27%">整改建议</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
  <h2>三、基础治理清单（按适用性实施）</h2><ol>{remediation_html}</ol>
  <h2>四、法源与适用性说明</h2><p>法源核验日期：{LEGAL_REVIEW_DATE}。本报告仅引用部分条款要义，不代替阅读法规全文，也不穷尽银行业、网络安全、数据出境、算法治理等其他要求。</p><ul>{source_html}</ul>
  <div class="footer"><strong>{esc(DISCLAIMER_SHORT)}</strong><br>{esc(APPLICABILITY_NOTICE)}</div>
</main></body></html>"""
    return document.encode("utf-8")


def csv_report(result: dict) -> bytes:
    output = io.StringIO(newline="")
    headers = ["规则ID", "风险维度", "分值", "事实判断", "风险说明", "法律或监管依据", "整改建议"]
    writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(result["triggered"])
    return ("\ufeff" + output.getvalue()).encode("utf-8")


def deepseek_payload(result: dict) -> dict:
    """Build a minimized payload: no project name, description, customer data, or raw prompts."""
    risks = [
        {
            "rule_id": item["规则ID"],
            "dimension": item["风险维度"],
            "points": item["分值"],
            "fact_summary": item["事实判断"],
            "risk": item["风险说明"],
            "legal_basis": item["法律或监管依据"],
            "recommended_control": item["整改建议"],
        }
        for item in result["triggered"]
    ]
    return {
        "score": result["score"],
        "level": result["level"],
        "trigger_count": result["trigger_count"],
        "triggered_rules": risks,
        "mandatory_disclaimer": DISCLAIMER_SHORT,
        "applicability_notice": APPLICABILITY_NOTICE,
    }


def result_fingerprint(result: dict) -> str:
    canonical = [
        {
            "id": item["规则ID"],
            "points": item["分值"],
            "fact": item["事实判断"],
        }
        for item in result["triggered"]
    ]
    serialized = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def call_deepseek(api_key: str, model: str, result: dict) -> str:
    if model not in DEEPSEEK_MODELS:
        raise ValueError("不支持的DeepSeek模型名称。")
    if not api_key or not api_key.strip():
        raise ValueError("请先输入DeepSeek API Key。")
    minimized_input = deepseek_payload(result)
    system_prompt = (
        "你是一名银行AI治理工程辅助助手。你只能基于用户提供的透明规则命中结果撰写治理摘要，"
        "不得改变分数、不得推断未提供的事实、不得声称形成法律意见或监管结论。"
        "请用中文输出，结构固定为：一、管理层摘要；二、P0/P1/P2整改清单（含责任角色与验收证据）；"
        "三、上线门禁建议；四、适用性与人工复核提示。引用法律时只能沿用输入中的条款要义，并保留条件性表述。"
    )
    response = requests.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "请根据以下去标识化评估结果生成整改计划：\n"
                    + json.dumps(minimized_input, ensure_ascii=False),
                },
            ],
            "thinking": {"type": "disabled"},
            "max_tokens": 1600,
            "stream": False,
        },
        timeout=(8, 45),
        allow_redirects=False,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"DeepSeek API返回HTTP {response.status_code}，请检查密钥、额度或服务状态。") from exc
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("DeepSeek API响应结构不符合预期。") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("DeepSeek API返回了空内容。")
    return content.strip()


def ascii_project_name(value: str) -> str:
    cleaned = "".join(ch for ch in value if ord(ch) < 128 and ch.isprintable()).strip()
    return cleaned or "Bank AI project (see Chinese report for project name)"


def english_stage(value: str) -> str:
    mapping = {
        "方案设计": "Design",
        "开发测试": "Development and testing",
        "试点": "Pilot",
        "已上线": "Production",
    }
    return mapping.get(value, "See Chinese report")


def english_pdf_summary(case: dict, result: dict) -> bytes:
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Bank AI Compliance Assessment - English Summary",
        author=APP_TITLE,
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=24,
            textColor=colors.HexColor("#0b1f33"),
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallMuted",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#596773"),
        )
    )
    story = [
        Paragraph("Bank AI Compliance Assessment", styles["ReportTitle"]),
        Paragraph("English Executive Summary", styles["Heading2"]),
        Spacer(1, 4 * mm),
    ]
    summary_data = [
        ["Project", ascii_project_name(str(case.get("name", "")))],
        ["Business stage", english_stage(str(case.get("stage", "")))],
        ["Risk score", f"{result['score']} / 100"],
        ["Risk level", result["level_en"]],
        ["Triggered rules", str(result["trigger_count"])],
        ["Legal source review date", LEGAL_REVIEW_DATE],
    ]
    table = Table(summary_data, colWidths=[48 * mm, 118 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f0f4")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#182532")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7d3da")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([table, Spacer(1, 6 * mm), Paragraph("Top triggered risks", styles["Heading2"])])
    top_items = sorted(result["triggered"], key=lambda item: (-item["分值"], item["规则ID"]))[:8]
    if top_items:
        for item in top_items:
            story.append(
                Paragraph(
                    f"<b>{html.escape(item['规则ID'])} (+{item['分值']})</b> — {html.escape(item['英文标题'])}",
                    styles["BodyText"],
                )
            )
            story.append(Spacer(1, 1.3 * mm))
    else:
        story.append(Paragraph("No additive risk rule was triggered by the submitted facts.", styles["BodyText"]))
    story.extend([Spacer(1, 4 * mm), Paragraph("Priority control themes", styles["Heading2"])])
    controls = [
        "Minimize data with field allowlists, masking or tokenization, and environment segregation.",
        "Use meaningful human review and a documented veto for decisions affecting customer rights.",
        "Perform vendor due diligence and contract for no training, audit rights, incident support, and deletion evidence.",
        "Map cross-border data flows and determine the applicable transfer mechanism based on the actual facts.",
        "Complete a PIPIA before processing when a statutory trigger applies; retain the assessment record as required.",
        "Use cited RAG sources, hallucination gates, red-team tests, model versioning, and privacy-aware logs.",
        "Test fairness and provide understandable explanations, appeal, human review, and correction channels.",
    ]
    for control in controls:
        story.append(Paragraph(f"• {html.escape(control)}", styles["BodyText"]))
        story.append(Spacer(1, 1.2 * mm))
    story.extend(
        [
            Spacer(1, 5 * mm),
            Paragraph("Important limitation", styles["Heading2"]),
            Paragraph(
                "For simulation, interview demonstration, and internal initial screening only. This is not legal advice, "
                "a regulatory conclusion, or an approval basis. Applicability depends on the business, legal roles, data "
                "classification, data flows, audience, and service scope. Professional legal, compliance, privacy, and "
                "security review is required.",
                styles["SmallMuted"],
            ),
        ]
    )
    document.build(story)
    return buffer.getvalue()


def safe_filename(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", value.strip())
    return value.strip("-")[:60] or "AI应用"


def configure_page() -> None:
    st.set_page_config(
        page_title=f"{APP_TITLE}｜银行AI合规评估",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
<style>
  :root { --bank-navy:#0b1f33; --bank-blue:#123b52; --bank-teal:#0f766e; --line:#dbe4ea; }
  .stApp { background: linear-gradient(180deg,#f3f7f9 0,#ffffff 420px); color:#182532; }
  .block-container { max-width: 1480px; padding-top: 1.3rem; padding-bottom: 3rem; }
  .bank-hero { background:linear-gradient(118deg,#0b1f33 0%,#123b52 68%,#0f5f68 100%); color:white; border-radius:16px; padding:27px 30px 25px; box-shadow:0 12px 32px rgba(11,31,51,.14); margin-bottom:16px; }
  .bank-hero .eyebrow { color:#8fd8cf; font-size:.78rem; letter-spacing:.14em; font-weight:700; }
  .bank-hero h1 { color:white; margin:.25rem 0 .1rem; font-size:clamp(1.45rem,3.1vw,2.05rem); overflow-wrap:anywhere; }
  .bank-hero p { color:#d5e4eb; margin:.25rem 0 0; font-size:1rem; }
  .legal-warning { background:#fff7ed; border:1px solid #fed7aa; border-left:5px solid #c2410c; border-radius:10px; padding:13px 16px; color:#7c2d12; font-weight:700; margin:10px 0 14px; }
  .applicability { background:#eef6f7; border:1px solid #c7e3e1; border-radius:10px; padding:11px 15px; color:#24454a; font-size:.9rem; margin-bottom:16px; }
  div[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:12px; padding:12px 15px; box-shadow:0 4px 15px rgba(11,31,51,.05); }
  div[data-testid="stMetric"] label { color:#50616e; }
  .score-panel { background:white; border:1px solid var(--line); border-radius:14px; padding:18px; box-shadow:0 8px 22px rgba(11,31,51,.06); }
  .score-number { font-size:3.25rem; line-height:1; font-weight:800; letter-spacing:-.04em; }
  .score-caption { color:#647480; font-size:.83rem; margin-top:5px; }
  .risk-level { display:inline-block; color:white; font-weight:800; border-radius:999px; padding:5px 11px; font-size:.84rem; }
  .section-kicker { color:#0f766e; font-weight:800; letter-spacing:.05em; font-size:.75rem; text-transform:uppercase; }
  .source-card { border:1px solid var(--line); border-radius:10px; padding:11px 13px; background:white; margin:8px 0; }
  .source-card a { color:#0b6670; font-weight:700; text-decoration:none; }
  .small-note { color:#657582; font-size:.82rem; }
  [data-testid="stSidebar"] { background:#edf3f6; border-right:1px solid #d7e1e7; }
  [data-testid="stSidebar"] .block-container { padding-top:1.25rem; }
  .stButton>button, .stDownloadButton>button { border-radius:8px; font-weight:700; }
  .stButton>button[kind="primary"] { background:#0f6670; border-color:#0f6670; }
  div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  @media (max-width: 900px) {
    .block-container { padding-left:.85rem; padding-right:.85rem; }
    .bank-hero { padding:21px 20px; border-radius:12px; }
    div[data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:.75rem; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] { flex:1 1 230px; min-width:0; }
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        f"""
<section class="bank-hero">
  <div class="eyebrow">RESPONSIBLE AI · PRIVACY · MODEL RISK</div>
  <h1>{APP_TITLE}</h1>
  <p>{APP_SUBTITLE}｜透明规则 · 逐条证据 · 可执行整改</p>
</section>
<div class="legal-warning">⚠ {html.escape(DISCLAIMER_SHORT)}</div>
<div class="applicability">{html.escape(APPLICABILITY_NOTICE)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(samples: list[dict]) -> None:
    with st.sidebar:
        st.markdown("### 演示控制台")
        sample_names = [item.get("name", f"样例 {index + 1}") for index, item in enumerate(samples)]
        selected_name = st.selectbox("选择预置案例", sample_names, key="sample_selector")
        selected_case = next(item for item in samples if item.get("name") == selected_name)
        if st.button("载入该案例", type="primary", width="stretch"):
            load_case_into_state(selected_case)
            st.session_state["active_sample"] = selected_name
            st.rerun()
        st.caption(f"当前载入：{st.session_state.get('active_sample', '自定义/默认')} ")
        st.divider()
        st.markdown("**评分区间**")
        st.caption("0—24 低｜25—49 中｜50—74 高｜75—100 极高")
        st.caption("17条规则固定加分，全部权重合计100分，无隐藏系数。")
        st.divider()
        st.markdown("**法源核验**")
        st.caption(f"核验日期：{LEGAL_REVIEW_DATE}")
        for source in LEGAL_SOURCES:
            st.markdown(f"- [{source['name']}]({source['url']})")
        st.divider()
        st.markdown("**DeepSeek API（可选）**")
        st.toggle("启用AI辅助整改摘要", key="deepseek_enabled")
        if st.session_state.get("deepseek_enabled"):
            st.text_input(
                "DeepSeek API Key",
                type="password",
                key="deepseek_api_key",
                help="仅保存在当前Streamlit会话内；项目不会写入文件或报告。也可通过DEEPSEEK_API_KEY环境变量提供。",
            )
            st.selectbox("模型", DEEPSEEK_MODELS, key="deepseek_model")
            st.caption("仅发送规则分数和去标识化风险摘要；不发送项目名称、场景描述或客户原始数据。")
            st.markdown(f"[DeepSeek官方API文档]({DEEPSEEK_DOC_URL})")
        st.divider()
        st.caption(DISCLAIMER_SHORT)


def render_questionnaire() -> bool:
    st.markdown("<div class='section-kicker'>01 / ASSESSMENT INPUT</div>", unsafe_allow_html=True)
    st.subheader("评估事实与控制措施")
    st.caption("勾选内容应来自项目材料、数据流和控制证据；未勾选控制会在适用场景下形成风险加分。")

    with st.form("assessment_form"):
        with st.expander("A. 项目概况", expanded=True):
            left, middle, right = st.columns([1.4, 1, 1])
            with left:
                st.text_input("项目名称", key="name")
            with middle:
                st.text_input("业务部门", key="department")
            with right:
                st.selectbox("项目阶段", ["方案设计", "开发测试", "试点", "已上线"], key="stage")
            st.text_area("场景描述", key="description", height=95)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.selectbox("服务对象", ["员工", "客户", "混合"], key="audience")
            with c2:
                st.toggle("使用生成式AI/大模型", key="generative_ai")
            with c3:
                st.selectbox(
                    "是否向境内公众提供服务",
                    ["否", "是", "待判断"],
                    key="public_service_scope",
                    help="《生成式人工智能服务管理暂行办法》第2条的适用范围判断之一。仅内部使用通常不在该办法适用范围内。",
                )
            with c4:
                st.selectbox(
                    "舆论/社会动员属性",
                    ["否", "是", "待判断"],
                    key="public_opinion_capability",
                    help="仅用于第17条的条件性提示，不代表本工具作出最终属性认定。",
                )

        with st.expander("B. 数据与处理链路", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.toggle("处理个人金融信息", key="personal_financial_info")
                st.toggle("处理敏感个人信息", key="sensitive_personal_info")
                st.toggle("向模型输入可识别客户信息", key="identifiable_customer_input")
            with c2:
                st.toggle("使用第三方大模型/API", key="third_party_model")
                st.toggle("存在境外存储/访问/处理", key="cross_border_processing")
                st.toggle("涉及重要数据（已识别/待确认）", key="important_data")
            with c3:
                st.toggle("主体为关键信息基础设施运营者", key="cii_operator")
                st.info("“个人金融信息”并不当然全部属于个保法第28条敏感信息；金融账户等字段属于法定列举类型，应逐字段识别。")

        with st.expander("C. 决策影响与客户权利", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.toggle("利用个人信息进行自动化决策", key="automated_decision")
                st.toggle("影响授信/定价/准入等重大权益", key="material_rights_impact")
            with c2:
                st.toggle("已设置人工实质复核", key="human_review")
                st.toggle("人工可一票否决模型建议", key="veto_control")
            with c3:
                st.toggle("已提供客户解释机制", key="customer_explanation")
                st.toggle("已提供申诉/更正渠道", key="appeal_correction_channel")

        with st.expander("D. 数据安全、日志与PIPIA", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.toggle("已实施字段白名单", key="field_whitelist")
                st.toggle("已脱敏/去标识化", key="masking")
                st.toggle("已实施环境与权限隔离", key="environment_isolation")
            with c2:
                st.toggle("留存必要且合规的提示词日志", key="prompt_logging")
                st.toggle("留存必要且合规的输出日志", key="output_logging")
                st.toggle("留存审批/人工复核日志", key="approval_logging")
            with c3:
                st.toggle("已完成PIPIA并形成记录", key="pipia_completed")
                st.caption("日志应先最小化和脱敏，并配置权限、期限与删除；不是要求全量、永久保留可识别输入。")

        with st.expander("E. 大模型可靠性与版本治理", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.toggle("已实施事实核验/幻觉拦截", key="fact_verification")
                st.toggle("RAG答案展示来源引用", key="rag_citations")
            with c2:
                st.toggle("已完成算法偏见/公平性测试", key="fairness_test")
                st.toggle("已完成对抗/红队测试", key="red_team_test")
            with c3:
                st.toggle("已实施模型/提示词/知识库版本管理", key="version_management")
                st.toggle("条件适用时已完成安全评估/算法备案", key="safety_assessment_filing")

        with st.expander("F. 第三方模型供应商治理", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.toggle("已完成供应商尽调", key="vendor_due_diligence")
                st.toggle("已签署数据处理协议", key="data_processing_agreement")
            with c2:
                st.toggle("合同/配置明确禁止用于训练", key="training_prohibited")
                st.toggle("银行保留审计权", key="audit_right")
            with c3:
                st.toggle("可取得返还/删除证明", key="deletion_certificate")
                st.caption("需先按事实判断供应商是受托人还是独立处理者，再匹配个保法第21条或第23条要求。")

        submitted = st.form_submit_button("运行透明规则评估", type="primary", width="stretch")
    return submitted


def render_results(case: dict, result: dict) -> None:
    st.markdown("<div class='section-kicker'>02 / EXPLAINABLE RESULT</div>", unsafe_allow_html=True)
    st.subheader("风险评分与逐条依据")
    c1, c2, c3, c4 = st.columns([1.15, 1, 1, 1])
    with c1:
        st.markdown(
            f"""
<div class="score-panel">
  <div class="score-number" style="color:{result['color']}">{result['score']}</div>
  <div class="score-caption">风险分 / 100</div>
  <div style="margin-top:12px"><span class="risk-level" style="background:{result['color']}">{result['level']}风险</span></div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("触发规则", f"{result['trigger_count']} 条")
    with c3:
        st.metric("规则权重总计", "100 分")
    with c4:
        st.metric("法源核验日期", LEGAL_REVIEW_DATE)
    st.caption("计算方式：命中的固定分值直接相加；总分 = min(100, 触发分值合计)。本规则库全部权重合计100分，不使用模型推断或隐藏系数。")

    notes = validate_case(case)
    if notes:
        with st.expander(f"事实一致性提示（{len(notes)}项）", expanded=False):
            for note in notes:
                st.warning(note)

    if result["triggered"]:
        display_rows = [
            {
                "规则ID": item["规则ID"],
                "风险维度": item["风险维度"],
                "分值": f"+{item['分值']}",
                "事实判断": item["事实判断"],
                "风险说明": item["风险说明"],
                "法律或监管依据": item["法律或监管依据"],
            }
            for item in result["triggered"]
        ]
        st.dataframe(
            pd.DataFrame(display_rows),
            width="stretch",
            hide_index=True,
            column_config={
                "规则ID": st.column_config.TextColumn(width="small"),
                "风险维度": st.column_config.TextColumn(width="medium"),
                "分值": st.column_config.TextColumn(width="small"),
                "事实判断": st.column_config.TextColumn(width="large"),
                "风险说明": st.column_config.TextColumn(width="large"),
                "法律或监管依据": st.column_config.TextColumn(width="large"),
            },
        )
        st.markdown("#### 按触发规则展开整改")
        for item in sorted(result["triggered"], key=lambda row: (-row["分值"], row["规则ID"])):
            with st.expander(f"{item['规则ID']}｜{item['风险维度']}｜+{item['分值']}分"):
                st.markdown(f"**事实判断**：{item['事实判断']}")
                st.markdown(f"**风险说明**：{item['风险说明']}")
                st.markdown(f"**法律或监管依据**：{item['法律或监管依据']}")
                st.markdown(f"**建议控制**：{item['整改建议']}")
    else:
        st.success("当前事实未触发加分规则。低风险不等于无风险，请维持基础控制并在数据流或功能变化时重新评估。")

    st.markdown("#### 基础治理清单（按适用性实施）")
    for index, item in enumerate(BASELINE_REMEDIATIONS, start=1):
        st.markdown(f"{index}. {item}")

    st.markdown("#### DeepSeek辅助治理摘要（可选）")
    st.caption("DeepSeek只对本地规则结果做结构化归纳，不参与评分、不替代法务复核；关闭或未配置API也不影响全部评估与导出功能。")
    if not st.session_state.get("deepseek_enabled", False):
        st.info("如需AI生成P0/P1/P2整改计划，可在左侧开启DeepSeek API。当前评分完全由本地透明规则生成。")
    else:
        api_key = st.session_state.get("deepseek_api_key", "") or os.getenv("DEEPSEEK_API_KEY", "")
        model = st.session_state.get("deepseek_model", DEEPSEEK_MODELS[0])
        if not api_key:
            st.warning("已开启DeepSeek增强，但尚未提供API Key。可在左侧输入，或设置DEEPSEEK_API_KEY环境变量。")
        if st.button("调用DeepSeek生成整改计划", disabled=not bool(api_key), width="stretch"):
            with st.spinner(f"正在调用 {model} 生成去标识化整改摘要…"):
                try:
                    st.session_state["deepseek_analysis"] = call_deepseek(api_key, model, result)
                    st.session_state["deepseek_analysis_fingerprint"] = result_fingerprint(result)
                    st.session_state["deepseek_analysis_model"] = model
                except (requests.RequestException, RuntimeError, ValueError) as exc:
                    st.error(f"DeepSeek调用失败：{exc}")
        if st.session_state.get("deepseek_analysis"):
            if st.session_state.get("deepseek_analysis_fingerprint") != result_fingerprint(result):
                st.warning("当前评分已变化，下方DeepSeek摘要对应上一次评分；请重新生成。")
            st.markdown(st.session_state["deepseek_analysis"])
            st.caption(f"生成模型：{st.session_state.get('deepseek_analysis_model', model)}｜请人工核验后使用。")

    st.markdown("<div class='section-kicker'>03 / EXPORT</div>", unsafe_allow_html=True)
    st.subheader("导出可审阅材料")
    html_bytes = html_report(case, result)
    csv_bytes = csv_report(result)
    pdf_bytes = english_pdf_summary(case, result)
    file_stem = safe_filename(str(case.get("name", "AI应用")))
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "下载中文HTML完整报告",
            data=html_bytes,
            file_name=f"{file_stem}_合规风险评估报告.html",
            mime="text/html",
            width="stretch",
        )
        st.caption("UTF-8自包含文件，可在浏览器中打印为PDF。")
    with d2:
        st.download_button(
            "下载CSV风险明细",
            data=csv_bytes,
            file_name=f"{file_stem}_风险明细.csv",
            mime="text/csv",
            width="stretch",
        )
        st.caption("UTF-8 BOM，便于使用Excel直接打开。")
    with d3:
        st.download_button(
            "下载英文PDF摘要",
            data=pdf_bytes,
            file_name=f"{file_stem}_English_Summary.pdf",
            mime="application/pdf",
            width="stretch",
        )
        st.caption("面向面试/跨团队沟通的英文执行摘要。")


def render_rules_and_sources() -> None:
    st.markdown("<div class='section-kicker'>RULE CATALOG</div>", unsafe_allow_html=True)
    st.subheader("公开的评分规则")
    st.write("以下为全部17条固定加分规则。每条规则的触发条件与分值均直接写在源码中，总权重恰为100分。")
    catalog = pd.DataFrame(
        [
            {
                "规则ID": rule.rule_id,
                "风险维度": rule.dimension,
                "固定分值": rule.points,
                "触发条件（摘要）": rule.trigger_text,
                "主要依据/边界": rule.legal_basis,
            }
            for rule in RULES
        ]
    )
    st.dataframe(catalog, width="stretch", hide_index=True)
    st.info(f"规则分值合计：{sum(rule.points for rule in RULES)}。等级：0—24低、25—49中、50—74高、75—100极高。该分值为演示用内部筛查尺度，不是监管评分标准。")

    st.markdown("#### 关键适用性边界")
    st.markdown(
        """
1. 《生成式人工智能服务管理暂行办法》第2条将其范围限定为利用生成式AI技术向境内公众提供服务；企业内部应用未向境内公众提供服务的，不适用该办法。内部场景可把相关条款作为治理参考，但不能据此直接下违法结论。
2. 第三方模型按银行指示处理个人信息时可能属于委托处理，重点看《个人信息保护法》第21条；若供应商自行决定处理目的或方式，则可能涉及第23条。必须基于合同、技术配置和真实数据流做角色判定。
3. 《数据安全法》第31条聚焦重要数据出境，不是所有个人信息或所有数据跨境的一般安全评估条款。
4. 人工复核、一票否决、训练禁用、审计权、删除证明、RAG引用、幻觉拦截、红队测试和版本日志等是支持落实法定目标的治理措施，并非所引条款对所有项目逐字规定的统一义务。
        """
    )
    st.markdown("#### 官方法源")
    for source in LEGAL_SOURCES:
        st.markdown(
            f"""
<div class="source-card"><a href="{source['url']}" target="_blank">{source['name']}</a><br><span class="small-note">{source['publisher']}｜核验日期：{LEGAL_REVIEW_DATE}</span></div>
            """,
            unsafe_allow_html=True,
        )
    st.warning(f"{DISCLAIMER_SHORT} {APPLICABILITY_NOTICE}")


def main() -> None:
    configure_page()
    samples = load_sample_cases()
    if "state_initialized" not in st.session_state:
        initial = samples[0] if samples else DEFAULT_CASE
        load_case_into_state(initial)
        st.session_state["active_sample"] = initial.get("name", "默认")
        st.session_state["state_initialized"] = True
        st.session_state["deepseek_enabled"] = False
        st.session_state["deepseek_api_key"] = ""
        st.session_state["deepseek_model"] = DEEPSEEK_MODELS[0]
    render_sidebar(samples)
    render_header()
    workbench, rulebook = st.tabs(["评估工作台", "透明规则与法源"])
    with workbench:
        render_questionnaire()
        case = case_from_state()
        result = evaluate_case(case)
        st.divider()
        render_results(case, result)
    with rulebook:
        render_rules_and_sources()


if __name__ == "__main__":
    main()
