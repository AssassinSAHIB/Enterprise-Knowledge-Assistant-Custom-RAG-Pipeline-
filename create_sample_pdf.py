"""
Script to generate a realistic enterprise policy document (sample.pdf).
Used for ingestion and retrieval benchmarking.
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def create_sample_pdf(filename: str = "sample.pdf") -> None:
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        name="DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12,
    )
    
    heading2_style = ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=14,
        spaceAfter=6,
    )
    
    body_style = ParagraphStyle(
        name="DocBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8,
    )

    story = []

    story.append(Paragraph("Enterprise Knowledge & Compliance Manual", title_style))
    story.append(Paragraph("Document Version: 2026.2 | Status: Approved | Distribution: Internal Only", body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Corporate Policy Guidelines & Code of Conduct", heading2_style))
    story.append(Paragraph(
        "All employees, contractors, and partners must strictly adhere to the enterprise policy guidelines. "
        "The core policy guidelines mandate that organizational assets must be used exclusively for authorized business operations. "
        "Employees are expected to conduct all communications with transparency, integrity, and ethical compliance. "
        "Any conflict of interest, whether financial, interpersonal, or external consultancy, must be disclosed immediately to the compliance officer. "
        "Violations of these policy guidelines may result in corrective actions up to and including termination of employment.",
        body_style
    ))

    story.append(Paragraph("2. Information Security and Data Protection Policies", heading2_style))
    story.append(Paragraph(
        "Data classification consists of four levels: Public, Internal, Confidential, and Restricted. "
        "Customer personally identifiable information (PII) and intellectual property are categorized as Restricted. "
        "Encryption at rest using AES-256 and encryption in transit using TLS 1.3 are strictly mandatory for all internal and external data pipelines. "
        "Multi-factor authentication (MFA) must be enforced across all corporate accounts, single sign-on (SSO) gateways, and remote VPN connections. "
        "Employees are prohibited from storing corporate credentials or proprietary codebases on personal machines or unapproved cloud repositories.",
        body_style
    ))

    story.append(Paragraph("3. Remote Work & Device Security Protocol", heading2_style))
    story.append(Paragraph(
        "The remote work policy guidelines state that team members working outside company facilities must connect through authorized corporate VPN networks. "
        "Corporate-issued hardware must remain updated with automated security patches and enterprise endpoint detection software. "
        "Public Wi-Fi networks must never be used without an active encrypted corporate tunnel. "
        "Physical documents containing sensitive company information must be shredded according to ISO/IEC 27001 document disposal standards.",
        body_style
    ))

    story.append(Paragraph("4. Incident Reporting and Escalation Procedures", heading2_style))
    story.append(Paragraph(
        "Any observed or suspected security vulnerability, data leakage, or phishing attempt must be reported to security@enterprise.internal within sixty minutes of discovery. "
        "The Security Operations Center (SOC) operates 24/7/365 to triage alerts and mitigate threat propagation. "
        "Emergency containment protocols include immediate network isolation of compromised hosts and revocation of associated API keys and session tokens. "
        "Post-incident reviews must be conducted within 72 hours, producing a formal Root Cause Analysis (RCA) report.",
        body_style
    ))

    story.append(Paragraph("5. AI Model Governance and Automated Tool Usage", heading2_style))
    story.append(Paragraph(
        "Deployment of generative AI systems and large language models must undergo thorough algorithmic auditing. "
        "No internal business records, proprietary algorithm designs, or unmasked client identifiers may be submitted to external AI model APIs without legal and technical risk review. "
        "Local vector search and private retrieval-augmented generation architectures are encouraged to preserve institutional data privacy and security.",
        body_style
    ))

    doc.build(story)
    print(f"[OK] Successfully generated enterprise document: {filename}")


if __name__ == "__main__":
    create_sample_pdf("sample.pdf")
