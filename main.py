"""
================================================================
  COMPLAINT INTEL — Multilingual Complaint Intelligence Platform
================================================================
  A single-file Django + DRF app for a college project demo.

  RUN IT:
      python complaint_platform.py

  OPEN:
      http://127.0.0.1:8000/

  PAGES:
      /                 Overview dashboard
      /submit/          File a new complaint (runs the NLP pipeline live)
      /complaints/      Browse + filter every complaint
      /complaints/<id>/ Full detail + agent workflow for one complaint
      /trends/          Analytics deep dive
      /alerts/          RBI / TRAI / SEBI regulatory alert queue

  NO API KEYS NEEDED.
      Language detection, category classification, sentiment scoring
      and draft-response generation are all done with plain Python
      (keyword heuristics + VADER if installed). Nothing here calls
      out to a paid AI service, so it runs fully offline once the
      pip packages below are installed.

  INSTALL (only if not already installed):
      pip install django djangorestframework
      pip install vaderSentiment      # optional, improves sentiment scoring
================================================================
"""

import os
import sys
import uuid
import json
import random
import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  DJANGO BOOTSTRAP  (must happen before any django import)
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "__main__")

from django.conf import settings

if not settings.configured:
    settings.configure(
        BASE_DIR=BASE_DIR,
        SECRET_KEY="complaint-intel-local-dev-key",
        DEBUG=True,
        ALLOWED_HOSTS=["*"],
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.staticfiles",
            "rest_framework",
        ],
        MIDDLEWARE=[
            "django.middleware.security.SecurityMiddleware",
            "django.middleware.common.CommonMiddleware",
        ],
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": BASE_DIR / "complaints.db",
            }
        },
        ROOT_URLCONF="__main__",
        STATIC_URL="/static/",
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
        REST_FRAMEWORK={"UNAUTHENTICATED_USER": None},
        TIME_ZONE="Asia/Kolkata",
        USE_TZ=True,
    )

import django
django.setup()

from django.db import models
from django.core.management import call_command
from django.core.exceptions import ValidationError
from django.http import HttpResponse, Http404
from django.views import View
from django.utils import timezone
from django.db.models import Count, Avg
from django.urls import path
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import serializers as drf_serializers, status


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — DATABASE MODELS
# ══════════════════════════════════════════════════════════════

class Complaint(models.Model):
    """Core complaint record — every field from ingestion through NLP."""

    CHANNEL_CHOICES = [
        ("whatsapp", "WhatsApp"),
        ("email", "Email"),
        ("phone", "Phone Call"),
        ("twitter", "Twitter / X"),
        ("web_form", "Web Form"),
    ]
    SEVERITY_CHOICES = [(i, str(i)) for i in range(1, 6)]
    STATUS_CHOICES = [
        ("new", "New"),
        ("in_progress", "In Progress"),
        ("resolved", "Resolved"),
        ("escalated", "Escalated"),
    ]

    complaint_id         = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    channel               = models.CharField(max_length=20, choices=CHANNEL_CHOICES, default="web_form")
    sender_id             = models.CharField(max_length=200, default="anonymous")
    original_text         = models.TextField()
    detected_language     = models.CharField(max_length=40, default="English")
    category               = models.CharField(max_length=100, blank=True)
    category_confidence    = models.FloatField(default=0.0)
    severity                = models.IntegerField(choices=SEVERITY_CHOICES, default=1)
    sentiment_score         = models.FloatField(default=0.0)   # -1 very negative -> +1 very positive
    regulatory_breach       = models.TextField(default="{}")  # JSON string — plain TextField so it
                                                                # never depends on SQLite's JSON1 extension
    requires_escalation     = models.BooleanField(default=False)
    draft_response          = models.TextField(blank=True)
    agent_response           = models.TextField(blank=True)
    status                    = models.CharField(max_length=20, choices=STATUS_CHOICES, default="new")
    created_at                = models.DateTimeField(auto_now_add=True)
    resolved_at               = models.DateTimeField(null=True, blank=True)

    class Meta:
        app_label = "complaints"
        ordering = ["-severity", "-created_at"]

    def __str__(self):
        return f"[{self.severity}/5] {self.category} - {self.channel}"


class RegulatoryAlert(models.Model):
    """A complaint that tripped an RBI / TRAI / SEBI trigger phrase."""

    complaint     = models.OneToOneField(Complaint, on_delete=models.CASCADE, related_name="alert")
    regulator      = models.CharField(max_length=10)      # RBI / SEBI / TRAI
    triggers        = models.TextField(default="[]")  # JSON string, same reason as above
    acknowledged     = models.BooleanField(default=False)
    created_at        = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "complaints"
        ordering = ["acknowledged", "-created_at"]


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — NLP ENGINE  (rule-based + VADER, 100% offline)
#              A production build would swap these for
#              MuRIL / IndicBERT / Whisper — noted inline.
# ══════════════════════════════════════════════════════════════

CATEGORIES = {
    "billing dispute":              ["bill", "charge", "overcharged", "invoice", "amount", "deduct",
                                      "payment", "refund", "double charge", "extra charge"],
    "service outage":               ["not working", "down", "outage", "disconnected", "no signal",
                                      "network", "internet", "offline", "no service", "speed"],
    "fraud or unauthorized access": ["fraud", "hack", "unauthorized", "stolen", "scam",
                                      "phishing", "unknown transaction", "suspicious"],
    "kyc / account issue":          ["kyc", "account", "blocked", "frozen", "verification",
                                      "documents", "pan", "aadhar", "closed account"],
    "refund not received":          ["refund", "money back", "return", "not received", "pending refund",
                                      "waiting", "credit back"],
    "rude staff behavior":          ["rude", "unprofessional", "behavior", "staff", "agent",
                                      "disrespectful", "not helpful", "ignored"],
    "loan / credit issue":          ["loan", "emi", "credit", "interest", "repayment",
                                      "foreclosure", "sanction", "lender"],
    "data privacy breach":          ["data", "privacy", "personal info", "leak", "shared",
                                      "consent", "information disclosed"],
}

REGULATORY_PATTERNS = {
    "RBI":  ["unauthorized debit", "upi fraud", "loan mis-selling", "excessive interest",
             "bank not responding", "account frozen", "unknown transaction"],
    "TRAI": ["call drop", "data speed", "sim block", "recharge failed",
             "number portability", "no signal", "network outage"],
    "SEBI": ["broker fraud", "mutual fund", "demat", "trading halt",
             "dividends", "stock", "market manipulation"],
}

LANGUAGE_HINTS = {
    "hindi":       ["mera", "meri", "hai", "nahi", "karo", "kiya", "rupees", "paisa",
                    "account", "bank", "problem", "chahiye", "hua", "kar"],
    "tamil":       ["enna", "illai", "pannu", "vandha", "pochi", "senji", "romba"],
    "telugu":      ["emi", "cheyandi", "ledu", "undi", "chesta", "ayyindi"],
    "hindi_roman": ["aapka", "hamara", "tumhara", "kyun", "kaise", "bahut"],
}

# A few phrasing variants per severity tier so replies don't all read identically.
DRAFT_TEMPLATES = {
    1: [
        "Thank you for writing in. We've logged your {category} complaint and a "
        "member of our team will review it within 2 business days. Reference: {ref}.",
        "We've received your note about {category}. Our support team will take a look "
        "and follow up within 2 business days. Reference: {ref}.",
    ],
    2: [
        "We're sorry for the trouble this {category} issue has caused. It's now with "
        "our team and you should hear back within 24 hours. Reference: {ref}.",
        "Apologies for the inconvenience. We've flagged your {category} complaint for "
        "review and will update you within a day. Reference: {ref}.",
    ],
    3: [
        "We're sorry to hear about this. Your {category} complaint has been marked "
        "priority and handed to a senior teammate — expect a resolution within 12 hours. "
        "Reference: {ref}.",
        "This has been escalated internally as a priority {category} case. A senior "
        "agent will be in touch within 12 hours. Reference: {ref}.",
    ],
    4: [
        "We take this seriously. Your {category} complaint has been escalated to our "
        "Grievance Officer per RBI/TRAI guidelines, and a senior representative will "
        "call you within 4 hours. Reference: {ref}.",
        "This has been routed straight to our Grievance Officer given the nature of "
        "the {category} issue. Expect a call within 4 hours. Reference: {ref}.",
    ],
    5: [
        "We sincerely apologize — this {category} issue has been escalated to the "
        "highest level and a team lead will personally call you within 2 hours. "
        "Reference: {ref}.",
        "This is now our top priority. A senior team lead will personally reach out "
        "about your {category} complaint within 2 hours. Reference: {ref}.",
    ],
}


def detect_language(text: str) -> str:
    """Keyword-overlap language guess. Production: langdetect + Whisper for audio."""
    text_lower = text.lower()
    scores = {lang: sum(1 for kw in kws if kw in text_lower) for lang, kws in LANGUAGE_HINTS.items()}
    best = max(scores, key=scores.get)
    if scores[best] >= 2:
        return best.replace("_roman", " (Roman script)").title()
    return "English"


def classify_category(text: str) -> tuple[str, float]:
    """Keyword-overlap classifier. Production: MuRIL / zero-shot BART."""
    text_lower = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in text_lower) for cat, kws in CATEGORIES.items()}
    best = max(scores, key=scores.get)
    total = sum(scores.values()) or 1
    confidence = round(scores[best] / total, 2) if scores[best] > 0 else 0.1
    category = best if scores[best] > 0 else "general inquiry"
    return category, min(confidence, 0.97)


def score_sentiment(text: str) -> float:
    """VADER if installed, else a small negative-keyword fallback. -1..+1."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        return round(sia.polarity_scores(text)["compound"], 3)
    except ImportError:
        negative_words = ["fraud", "stolen", "rude", "angry", "terrible", "horrible",
                           "worst", "pathetic", "useless", "cheated", "disgusting"]
        count = sum(1 for w in negative_words if w in text.lower())
        return round(max(-1.0, -0.2 * count), 3)


def compute_severity(sentiment: float, category: str, breach: dict) -> int:
    """Severity 1-5 from sentiment + category risk + regulatory breach."""
    HIGH_RISK = {
        "fraud or unauthorized access": 2,
        "data privacy breach": 2,
        "loan / credit issue": 1,
    }
    base = 1
    if sentiment < -0.7:
        base += 2
    elif sentiment < -0.4:
        base += 1
    base += HIGH_RISK.get(category, 0)
    if breach.get("detected"):
        base = max(base, 4)
    return min(base, 5)


def detect_regulatory_breach(text: str) -> dict:
    """Checks text against RBI / TRAI / SEBI trigger phrases."""
    text_lower = text.lower()
    for regulator, patterns in REGULATORY_PATTERNS.items():
        matched = [p for p in patterns if p in text_lower]
        if matched:
            return {"detected": True, "regulator": regulator, "triggers": matched}
    return {"detected": False, "regulator": None, "triggers": []}


def generate_draft_response(category: str, severity: int, ref_id: str) -> str:
    """Offline, template-based draft reply — no external API of any kind."""
    template = random.choice(DRAFT_TEMPLATES.get(severity, DRAFT_TEMPLATES[3]))
    return template.format(category=category, ref=str(ref_id)[:8].upper())


def process_complaint(text: str, channel: str = "web_form", sender_id: str = "anonymous") -> "Complaint":
    """Full pipeline: raw text -> classified + scored Complaint saved to DB."""
    language = detect_language(text)
    category, confidence = classify_category(text)
    sentiment = score_sentiment(text)
    breach = detect_regulatory_breach(text)
    severity = compute_severity(sentiment, category, breach)
    ref_id = uuid.uuid4()
    draft = generate_draft_response(category, severity, str(ref_id))

    complaint = Complaint.objects.create(
        complaint_id=ref_id,
        channel=channel,
        sender_id=sender_id or "anonymous",
        original_text=text,
        detected_language=language,
        category=category,
        category_confidence=confidence,
        severity=severity,
        sentiment_score=sentiment,
        regulatory_breach=json.dumps(breach),
        requires_escalation=(severity >= 4 or breach.get("detected", False)),
        draft_response=draft,
        status="escalated" if severity >= 4 else "new",
    )

    if breach.get("detected"):
        RegulatoryAlert.objects.create(
            complaint=complaint,
            regulator=breach["regulator"],
            triggers=json.dumps(breach["triggers"]),
        )
    return complaint


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — REST API  (JSON only, consumed by the pages' JS)
# ══════════════════════════════════════════════════════════════

class ComplaintSerializer(drf_serializers.ModelSerializer):
    # regulatory_breach is stored as a raw JSON string in the DB (see model comment);
    # decode it back into an object here so the frontend JS keeps working unchanged.
    regulatory_breach = drf_serializers.SerializerMethodField()

    class Meta:
        model = Complaint
        fields = "__all__"

    def get_regulatory_breach(self, obj):
        try:
            return json.loads(obj.regulatory_breach)
        except (TypeError, ValueError):
            return {"detected": False, "regulator": None, "triggers": []}


class AlertSerializer(drf_serializers.ModelSerializer):
    complaint_id       = drf_serializers.CharField(source="complaint.complaint_id", read_only=True)
    complaint_category = drf_serializers.CharField(source="complaint.category", read_only=True)
    complaint_severity  = drf_serializers.IntegerField(source="complaint.severity", read_only=True)
    complaint_channel   = drf_serializers.CharField(source="complaint.channel", read_only=True)
    complaint_text      = drf_serializers.SerializerMethodField()
    triggers              = drf_serializers.SerializerMethodField()

    class Meta:
        model = RegulatoryAlert
        fields = ["id", "complaint_id", "regulator", "triggers", "acknowledged", "created_at",
                  "complaint_category", "complaint_severity", "complaint_channel", "complaint_text"]

    def get_complaint_text(self, obj):
        t = obj.complaint.original_text
        return t[:160] + ("…" if len(t) > 160 else "")

    def get_triggers(self, obj):
        try:
            return json.loads(obj.triggers)
        except (TypeError, ValueError):
            return []


class ComplaintListCreateAPI(APIView):
    """GET list (+filters) / POST new complaint (runs the NLP pipeline)."""

    def get(self, request):
        qs = Complaint.objects.all()

        channel  = request.query_params.get("channel")
        severity = request.query_params.get("severity")
        category = request.query_params.get("category")
        cstatus  = request.query_params.get("status")
        search   = request.query_params.get("q")
        order    = request.query_params.get("order")
        limit    = int(request.query_params.get("limit", 50))

        if channel:  qs = qs.filter(channel=channel)
        if severity: qs = qs.filter(severity=severity)
        if category: qs = qs.filter(category__icontains=category)
        if cstatus:  qs = qs.filter(status=cstatus)
        if search:   qs = qs.filter(original_text__icontains=search)
        if order == "recent":
            qs = qs.order_by("-created_at")

        total = qs.count()
        serializer = ComplaintSerializer(qs[:limit], many=True)
        return Response({"total": total, "results": serializer.data})

    def post(self, request):
        text      = (request.data.get("text") or "").strip()
        channel   = request.data.get("channel", "web_form")
        sender_id = request.data.get("sender_id", "anonymous")

        if not text:
            return Response({"error": "text field is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            complaint = process_complaint(text, channel, sender_id)
        except Exception as exc:
            # Surface the real reason as JSON instead of letting Django's HTML
            # error page reach the frontend (which is what "Unexpected token '<'" means).
            import traceback
            traceback.print_exc()
            return Response({"error": f"{type(exc).__name__}: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(ComplaintSerializer(complaint).data, status=status.HTTP_201_CREATED)


class ComplaintDetailAPI(APIView):
    """GET single complaint / PATCH status or agent_response."""

    def _get(self, complaint_id):
        try:
            return Complaint.objects.get(complaint_id=complaint_id)
        except (Complaint.DoesNotExist, ValueError, ValidationError):
            return None

    def get(self, request, complaint_id):
        c = self._get(complaint_id)
        if not c:
            return Response({"error": "not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(ComplaintSerializer(c).data)

    def patch(self, request, complaint_id):
        c = self._get(complaint_id)
        if not c:
            return Response({"error": "not found"}, status=status.HTTP_404_NOT_FOUND)

        for field in ("status", "agent_response"):
            if field in request.data:
                setattr(c, field, request.data[field])
        if request.data.get("status") == "resolved" and not c.resolved_at:
            c.resolved_at = timezone.now()

        try:
            c.save()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return Response({"error": f"{type(exc).__name__}: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(ComplaintSerializer(c).data)


class TrendsAPI(APIView):
    """Aggregate analytics for the Overview + Trends pages."""

    def get(self, request):
        last_30d = timezone.now() - datetime.timedelta(days=30)
        qs = Complaint.objects.filter(created_at__gte=last_30d)

        by_category = list(qs.values("category").annotate(count=Count("complaint_id")).order_by("-count")[:8])
        by_channel  = list(qs.values("channel").annotate(count=Count("complaint_id")))
        by_severity = list(qs.values("severity").annotate(count=Count("complaint_id")).order_by("severity"))
        by_language = list(qs.values("detected_language").annotate(count=Count("complaint_id")).order_by("-count")[:6])
        by_status   = list(qs.values("status").annotate(count=Count("complaint_id")))

        daily = []
        for i in range(6, -1, -1):
            day = timezone.now().date() - datetime.timedelta(days=i)
            cnt = Complaint.objects.filter(created_at__date=day).count()
            daily.append({"date": str(day), "count": cnt})

        return Response({
            "summary": {
                "total_complaints":     Complaint.objects.count(),
                "open_complaints":      Complaint.objects.filter(status="new").count(),
                "escalated_complaints": Complaint.objects.filter(status="escalated").count(),
                "regulatory_alerts":    RegulatoryAlert.objects.filter(acknowledged=False).count(),
                "avg_severity":         round(Complaint.objects.aggregate(a=Avg("severity"))["a"] or 0, 1),
            },
            "by_category": by_category,
            "by_channel":  by_channel,
            "by_severity": by_severity,
            "by_language": by_language,
            "by_status":   by_status,
            "daily_volume": daily,
        })


class AlertListAPI(APIView):
    def get(self, request):
        qs = RegulatoryAlert.objects.select_related("complaint").all()
        only = request.query_params.get("status")
        if only == "open":
            qs = qs.filter(acknowledged=False)
        elif only == "acknowledged":
            qs = qs.filter(acknowledged=True)
        return Response({"total": qs.count(), "results": AlertSerializer(qs[:100], many=True).data})


class AlertDetailAPI(APIView):
    def patch(self, request, alert_id):
        try:
            a = RegulatoryAlert.objects.get(id=alert_id)
        except RegulatoryAlert.DoesNotExist:
            return Response({"error": "not found"}, status=status.HTTP_404_NOT_FOUND)
        if "acknowledged" in request.data:
            a.acknowledged = bool(request.data["acknowledged"])
            a.save()
        return Response(AlertSerializer(a).data)


class SeedDataAPI(APIView):
    """POST -> loads ~20 realistic sample complaints so the demo looks alive."""

    def post(self, request):
        samples = [
            ("My UPI payment of Rs 5000 was deducted but not credited. This is fraud!", "whatsapp", "9876543210"),
            ("Internet not working for 3 days. Very poor service. I need refund!", "email", "customer@gmail.com"),
            ("Mera account se 2000 rupees deduct ho gaye bina kisi reason ke. Please help.", "whatsapp", "9812345678"),
            ("Your agent was extremely rude and hung up on me. Totally unprofessional behavior.", "phone", "9911223344"),
            ("KYC documents submitted 3 weeks ago but account still blocked. Need urgent help.", "web_form", "user1"),
            ("Mutual fund SIP amount was deducted twice this month. Please investigate immediately.", "email", "investor@yahoo.com"),
            ("Call drops every 5 minutes. Network speed is 0.5 mbps. This is unacceptable for 5G plan.", "twitter", "@angrycustomer"),
            ("Loan EMI was charged at 24% instead of agreed 14%. This is mis-selling!", "email", "borrower@gmail.com"),
            ("My personal data was shared with a third party without my consent. Privacy breach!", "web_form", "user2"),
            ("Refund of Rs 12,000 still pending after 45 days. No response from customer care.", "whatsapp", "9123456789"),
            ("SIM card blocked without notice. Number portability request rejected 3 times.", "phone", "9988776655"),
            ("Unknown transactions of Rs 800 and Rs 1200 on my debit card. I did not do these.", "web_form", "user3"),
            ("Recharge of Rs 599 done but balance not updated. Happened 3rd time this month.", "twitter", "@frustrated_user"),
            ("Account frozen after I complained about unauthorized debit. This is unfair!", "email", "user4@gmail.com"),
            ("Net banking password was changed without my request. I suspect my account is hacked.", "web_form", "user5"),
            ("Bill shows Rs 2400 but I was on a Rs 999 plan. Please correct immediately.", "whatsapp", "9700123456"),
            ("Enna panreenga? Service romba mosam. 3 days la network illai. Refund kudunga.", "twitter", "@tamil_user"),
            ("Demat account showing negative balance. I never traded in futures. Please check.", "email", "trader@hotmail.com"),
            ("I have been waiting 2 months for my insurance claim. No updates whatsoever.", "phone", "9654321098"),
            ("Staff at your branch demanded documents not mentioned in KYC list. Very harassing.", "web_form", "user6"),
        ]
        created = 0
        try:
            for text, channel, sender in samples:
                c = process_complaint(text, channel, sender)
                fake_time = timezone.now() - datetime.timedelta(
                    days=random.randint(0, 6), hours=random.randint(0, 23))
                Complaint.objects.filter(complaint_id=c.complaint_id).update(created_at=fake_time)
                created += 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return Response({"error": f"{type(exc).__name__}: {exc}", "created_before_failure": created},
                             status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({
            "message": f"{created} sample complaints seeded successfully.",
            "total_in_db": Complaint.objects.count(),
        })


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — SHARED PAGE SHELL  (nav + design system)
# ══════════════════════════════════════════════════════════════

NAV_ITEMS = [
    ("home",       "/",            "Overview"),
    ("submit",     "/submit/",     "File a Complaint"),
    ("complaints", "/complaints/", "All Complaints"),
    ("trends",     "/trends/",     "Trends"),
    ("alerts",     "/alerts/",     "Regulatory Alerts"),
]

BASE_CSS = """
:root{
  --bg:#f5f2ec; --paper:#fbf9f4; --ink:#15161a; --ink-soft:#4a4a45;
  --muted:#8a887f; --line:#e4ddcd;
  --accent:#e8491d; --accent-dark:#c23a14; --amber:#c8891a; --teal:#1f7a6c;
  --danger:#c73a3a; --blue:#2f5fa8;
  --nav-bg:#14151a; --nav-text:#f2efe6;
  --radius:10px; --shadow:0 1px 3px rgba(20,17,10,.06);
  --mono:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace;
}
*{box-sizing:border-box;margin:0;padding:0;}
html{-webkit-text-size-adjust:100%;}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  background:var(--bg); color:var(--ink); font-size:15px; line-height:1.5;
}
a{color:inherit; text-decoration:none;}
.eyebrow{font-family:var(--mono); font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:var(--accent); font-weight:700;}
h1,h2,h3{font-weight:800; letter-spacing:-.01em;}

/* ── nav ── */
nav.topnav{
  background:var(--nav-bg); color:var(--nav-text); position:sticky; top:0; z-index:100;
  border-bottom:3px solid var(--accent);
}
.nav-inner{
  max-width:1280px; margin:0 auto; padding:0 24px; height:60px;
  display:flex; align-items:center; justify-content:space-between; gap:20px;
}
.brand{display:flex; align-items:center; gap:9px; font-weight:800; font-size:16px; letter-spacing:-.02em; white-space:nowrap;}
.brand .dot{width:9px; height:9px; background:var(--accent); border-radius:2px; display:inline-block;}
.brand small{font-family:var(--mono); font-weight:400; font-size:10px; color:#9b9a91; letter-spacing:.08em;}
.nav-links{display:flex; gap:2px; flex-wrap:wrap;}
.nav-link{
  padding:8px 13px; border-radius:7px; font-size:13px; font-weight:600; color:#c9c7bc;
  transition:background .15s, color .15s;
}
.nav-link:hover{background:rgba(255,255,255,.06); color:#fff;}
.nav-link.active{background:var(--accent); color:#fff;}
.nav-toggle{display:none; background:none; border:1px solid #33343c; color:#fff; border-radius:7px; padding:6px 10px; font-size:16px; cursor:pointer;}

/* ── layout ── */
main{max-width:1280px; margin:0 auto; padding:28px 24px 60px;}
.page-head{margin-bottom:22px;}
.page-head h1{font-size:26px; margin-top:6px;}
.page-head p{color:var(--ink-soft); margin-top:6px; max-width:640px;}

/* ── buttons ── */
.btn{
  display:inline-flex; align-items:center; gap:6px; padding:9px 16px; border-radius:8px;
  border:1.5px solid var(--ink); background:var(--paper); color:var(--ink); cursor:pointer;
  font-size:13px; font-weight:700; font-family:inherit; transition:all .15s;
}
.btn:hover{background:var(--ink); color:#fff;}
.btn-accent{background:var(--accent); border-color:var(--accent); color:#fff;}
.btn-accent:hover{background:var(--accent-dark); border-color:var(--accent-dark);}
.btn-ghost{border-color:var(--line); background:transparent;}
.btn-ghost:hover{background:var(--ink); border-color:var(--ink);}
.btn-sm{padding:6px 11px; font-size:12px; border-radius:6px;}
.btn:disabled{opacity:.5; cursor:not-allowed;}
.btn:disabled:hover{background:var(--paper); color:var(--ink);}

/* ── cards ── */
.card{background:var(--paper); border:1px solid var(--line); border-radius:var(--radius); box-shadow:var(--shadow);}
.pad{padding:18px 20px;}

/* ── summary grid ── */
.stat-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:22px;}
.stat-card{background:var(--paper); border:1px solid var(--line); border-left:4px solid var(--ink); border-radius:var(--radius); padding:15px 17px; box-shadow:var(--shadow);}
.stat-card .lbl{font-family:var(--mono); font-size:10.5px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted);}
.stat-card .val{font-size:27px; font-weight:800; margin-top:5px; letter-spacing:-.02em;}
.stat-card .sub{font-size:11.5px; color:var(--muted); margin-top:3px;}
.stat-accent{border-left-color:var(--accent);}
.stat-amber{border-left-color:var(--amber);}
.stat-teal{border-left-color:var(--teal);}
.stat-danger{border-left-color:var(--danger);}

/* ── chart grid ── */
.chart-grid{display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:22px;}
.chart-card h3{font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); margin-bottom:12px; font-weight:700;}
.chart-card canvas{max-height:190px;}

/* ── forms ── */
label{font-size:12px; font-weight:700; color:var(--ink-soft); display:block; margin-bottom:5px;}
input,select,textarea{
  width:100%; border:1.5px solid var(--line); border-radius:7px; padding:9px 11px;
  font-size:13.5px; background:#fff; color:var(--ink); font-family:inherit; outline:none;
}
input:focus,select:focus,textarea:focus{border-color:var(--accent);}
textarea{resize:vertical; min-height:110px;}
.field{margin-bottom:14px;}
.field-row{display:grid; grid-template-columns:1fr 1fr; gap:12px;}

/* ── badges ── */
.badge{display:inline-block; padding:3px 9px; border-radius:20px; font-size:11px; font-weight:700; white-space:nowrap;}
.sev-1{background:#e7f4ea; color:#227a3e;} .sev-2{background:#eef4d6; color:#6b7a1f;}
.sev-3{background:#fdf1d6; color:#a06b0a;} .sev-4{background:#fce3d6; color:#c14a17;}
.sev-5{background:#fbdada; color:#b32222;}
.status-new{background:#e6eefb; color:var(--blue);} .status-in_progress{background:#fdf1d6; color:#a06b0a;}
.status-resolved{background:#e7f4ea; color:#227a3e;} .status-escalated{background:#fbdada; color:#b32222;}
.reg-RBI{background:#e6eefb; color:var(--blue);} .reg-TRAI{background:#fdf1d6; color:#a06b0a;} .reg-SEBI{background:#e2f3f0; color:var(--teal);}

/* ── table ── */
.table-wrap{overflow-x:auto;}
table{width:100%; border-collapse:collapse; font-size:13px;}
th{text-align:left; font-family:var(--mono); font-size:10.5px; text-transform:uppercase; letter-spacing:.06em;
   color:var(--muted); padding:10px 14px; border-bottom:1.5px solid var(--line); white-space:nowrap;}
td{padding:11px 14px; border-bottom:1px solid var(--line); vertical-align:middle;}
tr.row-link{cursor:pointer;}
tr.row-link:hover{background:#f2eee4;}
.mono{font-family:var(--mono); font-size:12px;}
.muted{color:var(--muted);}
.empty-state{padding:40px 20px; text-align:center; color:var(--muted);}

/* ── filter bar ── */
.filter-bar{display:flex; flex-wrap:wrap; gap:10px; align-items:end; margin-bottom:16px;}
.filter-bar .field{margin-bottom:0; min-width:130px;}
.filter-bar input[type=text]{min-width:200px;}

/* ── toast / result box ── */
.notice{padding:12px 15px; border-radius:8px; font-size:13px; line-height:1.6; margin-top:14px; display:none;}
.notice.show{display:block;}
.notice-ok{background:#e7f4ea; border:1px solid #b9e3c4; color:#1e6b37;}
.notice-err{background:#fbdada; border:1px solid #f3b3b3; color:#a02323;}

/* ── misc ── */
.hr{border:none; border-top:1px solid var(--line); margin:18px 0;}
.grid-2{display:grid; grid-template-columns:1.3fr 1fr; gap:18px;}
.chip{display:inline-block; font-family:var(--mono); font-size:11px; background:#efe9da; border:1px solid var(--line);
      padding:3px 9px; border-radius:6px; margin:0 6px 6px 0;}
.section-title{font-size:13px; font-weight:800; text-transform:uppercase; letter-spacing:.05em; margin-bottom:10px; color:var(--ink);}
.kv{display:grid; grid-template-columns:130px 1fr; gap:8px 12px; font-size:13px;}
.kv .k{color:var(--muted); font-weight:600;}
.spinner{display:inline-block; width:13px; height:13px; border:2px solid #d8d3c4; border-top-color:var(--accent);
         border-radius:50%; animation:spin .7s linear infinite; vertical-align:-2px;}
@keyframes spin{to{transform:rotate(360deg);}}
.tabs{display:flex; gap:6px; margin-bottom:16px;}
.tab{padding:7px 14px; border-radius:7px; font-size:12.5px; font-weight:700; border:1.5px solid var(--line); background:var(--paper); cursor:pointer;}
.tab.active{background:var(--ink); color:#fff; border-color:var(--ink);}

/* ── hero (home only) ── */
.hero{background:var(--nav-bg); color:#f2efe6; border-radius:14px; padding:38px 34px; margin-bottom:26px; position:relative; overflow:hidden;}
.hero::after{content:""; position:absolute; right:-60px; top:-60px; width:220px; height:220px; background:var(--accent); opacity:.15; border-radius:50%;}
.hero .eyebrow{color:var(--amber);}
.hero h1{font-size:34px; color:#fff; margin:10px 0 8px; max-width:640px;}
.hero p{color:#c9c7bc; max-width:560px; margin-bottom:18px; font-size:14.5px;}
.hero .btn-accent{margin-right:10px;}
.hero .btn-ghost{border-color:#3a3b42; color:#f2efe6;}
.hero .btn-ghost:hover{background:#23242c;}

/* ── responsive ── */
@media(max-width:900px){
  .chart-grid{grid-template-columns:1fr 1fr;}
  .grid-2{grid-template-columns:1fr;}
}
@media(max-width:720px){
  .nav-links{position:absolute; top:60px; left:0; right:0; background:var(--nav-bg); flex-direction:column;
             padding:8px 16px 16px; display:none; border-bottom:3px solid var(--accent);}
  .nav-links.open{display:flex;}
  .nav-toggle{display:inline-block;}
  .chart-grid{grid-template-columns:1fr;}
  .field-row{grid-template-columns:1fr;}
  .hero h1{font-size:26px;}
  .stat-grid{grid-template-columns:repeat(2,1fr);}
}
"""

COMMON_JS = """
const API = '';
function esc(s){ const d=document.createElement('div'); d.textContent = (s===null||s===undefined) ? '' : s; return d.innerHTML; }
function fmtDate(iso){
  if(!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString('en-IN',{day:'2-digit',month:'short'}) + ' · ' + d.toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'});
}
function titleCase(s){ return (s||'').replace(/_/g,' ').replace(/\\w\\S*/g, t => t.charAt(0).toUpperCase()+t.slice(1)); }
function sevBadge(n){ return `<span class="badge sev-${n}">Sev ${n}/5</span>`; }
function statusBadge(s){ return `<span class="badge status-${s}">${titleCase(s)}</span>`; }
function regBadge(r){ return `<span class="badge reg-${r}">${r}</span>`; }
function channelLabel(c){ const m={whatsapp:'WhatsApp',email:'Email',phone:'Phone Call',twitter:'Twitter / X',web_form:'Web Form'}; return m[c]||c; }
function toggleNav(){ document.getElementById('navLinks').classList.toggle('open'); }
"""


def render_page(active: str, title: str, body: str, script: str) -> str:
    nav_links = "".join(
        f'<a href="{href}" class="nav-link{" active" if key == active else ""}">{label}</a>'
        for key, href, label in NAV_ITEMS
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} · Complaint Intel</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>{BASE_CSS}</style>
</head>
<body>
<nav class="topnav">
  <div class="nav-inner">
    <a href="/" class="brand"><span class="dot"></span>Complaint&nbsp;Intel <small>v1&nbsp;·&nbsp;offline&nbsp;NLP</small></a>
    <button class="nav-toggle" onclick="toggleNav()">&#9776;</button>
    <div class="nav-links" id="navLinks">{nav_links}</div>
  </div>
</nav>
<main>
{body}
</main>
<script>{COMMON_JS}
{script}
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — PAGES
# ══════════════════════════════════════════════════════════════

class HomeView(View):
    def get(self, request):
        body = """
<div class="hero">
  <div class="eyebrow">Banking · Telecom · NBFC · Insurance</div>
  <h1>Complaints arrive in a dozen languages. This reads all of them in one place.</h1>
  <p>Every WhatsApp message, email and web form gets auto-classified by category, severity and
     regulatory risk the moment it lands — no manual triage, no spreadsheet.</p>
  <a href="/submit/" class="btn btn-accent">File a Complaint →</a>
  <button class="btn btn-ghost" onclick="seedData()" id="seedBtn">Load Demo Data</button>
</div>

<div class="stat-grid" id="statGrid"></div>

<div class="chart-grid">
  <div class="card pad chart-card"><h3>By Category</h3><canvas id="chCategory"></canvas></div>
  <div class="card pad chart-card"><h3>By Severity</h3><canvas id="chSeverity"></canvas></div>
  <div class="card pad chart-card"><h3>By Channel</h3><canvas id="chChannel"></canvas></div>
</div>

<div class="card">
  <div class="pad" style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line);">
    <span class="section-title" style="margin:0;">Recent Complaints</span>
    <a href="/complaints/" class="btn btn-sm btn-ghost">View all →</a>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th>Ref</th><th>Category</th><th>Severity</th><th>Channel</th><th>Language</th><th>Status</th><th>Received</th></tr></thead>
    <tbody id="recentBody"><tr><td colspan="7" class="empty-state"><span class="spinner"></span> Loading…</td></tr></tbody>
  </table></div>
</div>
"""
        script = """
async function seedData(){
  if(!confirm('Load 20 sample complaints into the database?')) return;
  const btn = document.getElementById('seedBtn'); btn.disabled = true; btn.textContent = 'Loading…';
  try{
    const res = await fetch(API + '/api/seed/', {method:'POST'});
    const data = await res.json();
    alert(data.message);
    loadHome();
  } finally { btn.disabled = false; btn.textContent = 'Load Demo Data'; }
}

function statCard(cls,label,val,sub){
  return `<div class="stat-card ${cls}"><div class="lbl">${label}</div><div class="val">${val}</div><div class="sub">${sub||''}</div></div>`;
}

let charts = {};
function drawBar(id, labels, data, color){
  if(charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), {
    type: 'bar',
    data: { labels, datasets: [{ data, backgroundColor: color, borderRadius: 4 }] },
    options: { plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true, ticks:{precision:0}}} }
  });
}
function drawDoughnut(id, labels, data, colors){
  if(charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), {
    type: 'doughnut',
    data: { labels, datasets: [{ data, backgroundColor: colors }] },
    options: { plugins:{legend:{position:'bottom', labels:{boxWidth:10, font:{size:10}}}} }
  });
}

async function loadHome(){
  const [trendsRes, recentRes] = await Promise.all([
    fetch(API + '/api/trends/').then(r=>r.json()),
    fetch(API + '/api/complaints/?order=recent&limit=8').then(r=>r.json()),
  ]);

  const s = trendsRes.summary;
  document.getElementById('statGrid').innerHTML = [
    statCard('', 'Total Complaints', s.total_complaints, 'all time'),
    statCard('stat-accent', 'Open', s.open_complaints, 'awaiting first response'),
    statCard('stat-danger', 'Escalated', s.escalated_complaints, 'severity 4-5'),
    statCard('stat-amber', 'Regulatory Alerts', s.regulatory_alerts, 'unacknowledged'),
    statCard('stat-teal', 'Avg Severity', s.avg_severity + '/5', 'last 30 days'),
  ].join('');

  drawBar('chCategory', trendsRes.by_category.map(c=>titleCase(c.category)), trendsRes.by_category.map(c=>c.count), '#e8491d');
  drawBar('chSeverity', trendsRes.by_severity.map(c=>'Sev '+c.severity), trendsRes.by_severity.map(c=>c.count), '#c8891a');
  drawDoughnut('chChannel', trendsRes.by_channel.map(c=>channelLabel(c.channel)), trendsRes.by_channel.map(c=>c.count),
    ['#e8491d','#c8891a','#1f7a6c','#2f5fa8','#8a887f']);

  const rows = recentRes.results;
  document.getElementById('recentBody').innerHTML = rows.length ? rows.map(c => `
    <tr class="row-link" onclick="location.href='/complaints/${c.complaint_id}/'">
      <td class="mono">${c.complaint_id.slice(0,8)}</td>
      <td>${titleCase(c.category)}</td>
      <td>${sevBadge(c.severity)}</td>
      <td>${channelLabel(c.channel)}</td>
      <td>${esc(c.detected_language)}</td>
      <td>${statusBadge(c.status)}</td>
      <td class="muted">${fmtDate(c.created_at)}</td>
    </tr>`).join('') : '<tr><td colspan="7" class="empty-state">No complaints yet — click "Load Demo Data" above, or file one yourself.</td></tr>';
}
loadHome();
"""
        return HttpResponse(render_page("home", "Overview", body, script))


class SubmitView(View):
    def get(self, request):
        body = """
<div class="page-head">
  <div class="eyebrow">Step 1 of 1</div>
  <h1>File a Complaint</h1>
  <p>Paste any complaint text — English, Hindi, Tamil, Telugu, or a language mixed with English.
     The pipeline classifies it live: category, severity, sentiment, regulatory risk and a draft reply.</p>
</div>

<div class="grid-2">
  <div class="card pad">
    <div class="field">
      <label>Complaint Text</label>
      <textarea id="formText" placeholder="e.g. Mera account se 2000 rupees deduct ho gaye bina kisi reason ke..."></textarea>
    </div>
    <div class="field-row">
      <div class="field">
        <label>Channel</label>
        <select id="formChannel">
          <option value="web_form">Web Form</option>
          <option value="whatsapp">WhatsApp</option>
          <option value="email">Email</option>
          <option value="phone">Phone Call</option>
          <option value="twitter">Twitter / X</option>
        </select>
      </div>
      <div class="field">
        <label>Sender ID (phone / email — optional)</label>
        <input id="formSender" type="text" placeholder="anonymous">
      </div>
    </div>
    <div style="display:flex; gap:10px;">
      <button class="btn btn-accent" id="submitBtn" onclick="submitComplaint()">Analyze &amp; Submit</button>
      <button class="btn btn-ghost" onclick="fillExample()">Fill Example</button>
    </div>
    <div class="notice" id="notice"></div>
  </div>

  <div>
    <div class="card pad" style="margin-bottom:14px;">
      <div class="section-title">Languages Recognized</div>
      <span class="chip">English</span><span class="chip">Hindi</span><span class="chip">Tamil</span>
      <span class="chip">Telugu</span><span class="chip">Hindi (Roman script)</span>
      <p class="muted" style="margin-top:8px; font-size:12.5px;">Heuristic keyword detection today — swap in
        <code>langdetect</code> / Whisper transcripts for production accuracy.</p>
    </div>
    <div class="card pad" id="resultCard" style="display:none;">
      <div class="section-title">Pipeline Result</div>
      <div class="kv" id="resultKv"></div>
      <hr class="hr">
      <div class="section-title">Draft Response</div>
      <p id="resultDraft" style="font-size:13px; line-height:1.6; color:var(--ink-soft);"></p>
      <a id="resultLink" class="btn btn-sm btn-ghost" style="margin-top:10px;">Open full record →</a>
    </div>
  </div>
</div>
"""
        script = """
function fillExample(){
  const examples = [
    "Mera account se 5000 rupees bina kisi reason ke deduct ho gaye. UPI fraud ho gaya hai mujhe!",
    "My internet speed is 0.1 mbps for a 5G plan. Network is completely down for 2 days.",
    "Enna panneenga! Bill amount romba jayasthi send panni irukeenga. Refund kudunga!",
    "Your agent was extremely rude and refused to help. Very unprofessional behavior.",
    "Loan EMI deducted twice this month. Also the interest rate is 22% not the 14% agreed.",
    "Unknown transaction of Rs 3200 on my account. I suspect my account is hacked.",
    "SIM card blocked without any notice. Number portability rejected 4 times now.",
  ];
  document.getElementById('formText').value = examples[Math.floor(Math.random()*examples.length)];
  document.getElementById('formChannel').value = ['whatsapp','email','phone','twitter','web_form'][Math.floor(Math.random()*5)];
  document.getElementById('formSender').value = '98' + Math.floor(Math.random()*90000000 + 10000000);
}

function showNotice(kind, msg){
  const el = document.getElementById('notice');
  el.className = 'notice show ' + (kind === 'ok' ? 'notice-ok' : 'notice-err');
  el.textContent = msg;
}

async function submitComplaint(){
  const text = document.getElementById('formText').value.trim();
  const channel = document.getElementById('formChannel').value;
  const sender = document.getElementById('formSender').value.trim() || 'anonymous';
  if(!text){ showNotice('err', 'Please enter the complaint text first.'); return; }

  const btn = document.getElementById('submitBtn');
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Analyzing…';

  try{
    const res = await fetch(API + '/api/complaints/', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text, channel, sender_id: sender})
    });
    const data = await res.json();
    if(!res.ok){ showNotice('err', 'Error: ' + JSON.stringify(data)); return; }

    showNotice('ok', 'Complaint processed and saved.');
    const breach = data.regulatory_breach && data.regulatory_breach.detected;
    document.getElementById('resultCard').style.display = 'block';
    document.getElementById('resultKv').innerHTML = `
      <div class="k">Category</div><div>${titleCase(data.category)}</div>
      <div class="k">Language</div><div>${esc(data.detected_language)}</div>
      <div class="k">Severity</div><div>${sevBadge(data.severity)}</div>
      <div class="k">Sentiment</div><div>${data.sentiment_score}</div>
      <div class="k">Regulatory Risk</div><div>${breach ? regBadge(data.regulatory_breach.regulator) + ' flagged' : '<span class="muted">none detected</span>'}</div>
      <div class="k">Status</div><div>${statusBadge(data.status)}</div>
    `;
    document.getElementById('resultDraft').textContent = data.draft_response;
    document.getElementById('resultLink').href = '/complaints/' + data.complaint_id + '/';
    document.getElementById('formText').value = '';
    document.getElementById('formSender').value = '';
  } catch(e){
    showNotice('err', 'Network error: ' + e.message);
  } finally {
    btn.disabled = false; btn.textContent = 'Analyze & Submit';
  }
}
"""
        return HttpResponse(render_page("submit", "File a Complaint", body, script))


class ComplaintsListView(View):
    def get(self, request):
        body = """
<div class="page-head">
  <div class="eyebrow">Live Queue</div>
  <h1>All Complaints</h1>
  <p>Filter by channel, category, severity or status — or search the raw text.</p>
</div>

<div class="filter-bar card pad">
  <div class="field"><label>Search</label><input type="text" id="fSearch" placeholder="keyword…"></div>
  <div class="field"><label>Channel</label>
    <select id="fChannel"><option value="">All</option>
      <option value="whatsapp">WhatsApp</option><option value="email">Email</option>
      <option value="phone">Phone Call</option><option value="twitter">Twitter / X</option>
      <option value="web_form">Web Form</option></select></div>
  <div class="field"><label>Severity</label>
    <select id="fSeverity"><option value="">All</option>
      <option value="1">1</option><option value="2">2</option><option value="3">3</option>
      <option value="4">4</option><option value="5">5</option></select></div>
  <div class="field"><label>Status</label>
    <select id="fStatus"><option value="">All</option>
      <option value="new">New</option><option value="in_progress">In Progress</option>
      <option value="resolved">Resolved</option><option value="escalated">Escalated</option></select></div>
  <button class="btn btn-accent btn-sm" onclick="applyFilters()">Apply</button>
  <button class="btn btn-ghost btn-sm" onclick="resetFilters()">Reset</button>
</div>

<div class="card">
  <div class="pad" style="display:flex; justify-content:space-between; border-bottom:1px solid var(--line);">
    <span class="section-title" style="margin:0;" id="resultCount">Loading…</span>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th>Ref</th><th>Category</th><th>Severity</th><th>Channel</th><th>Language</th><th>Status</th><th>Received</th></tr></thead>
    <tbody id="listBody"><tr><td colspan="7" class="empty-state"><span class="spinner"></span> Loading…</td></tr></tbody>
  </table></div>
</div>
"""
        script = """
let currentLimit = 50;
function applyFilters(){ loadList(); }
function resetFilters(){
  document.getElementById('fSearch').value = '';
  document.getElementById('fChannel').value = '';
  document.getElementById('fSeverity').value = '';
  document.getElementById('fStatus').value = '';
  loadList();
}
async function loadList(){
  const params = new URLSearchParams();
  const q = document.getElementById('fSearch').value.trim();
  const ch = document.getElementById('fChannel').value;
  const sv = document.getElementById('fSeverity').value;
  const st = document.getElementById('fStatus').value;
  if(q) params.set('q', q);
  if(ch) params.set('channel', ch);
  if(sv) params.set('severity', sv);
  if(st) params.set('status', st);
  params.set('order', 'recent');
  params.set('limit', currentLimit);

  const res = await fetch(API + '/api/complaints/?' + params.toString());
  const data = await res.json();
  document.getElementById('resultCount').textContent = data.total + ' complaint' + (data.total===1?'':'s');
  document.getElementById('listBody').innerHTML = data.results.length ? data.results.map(c => `
    <tr class="row-link" onclick="location.href='/complaints/${c.complaint_id}/'">
      <td class="mono">${c.complaint_id.slice(0,8)}</td>
      <td>${titleCase(c.category)}</td>
      <td>${sevBadge(c.severity)}</td>
      <td>${channelLabel(c.channel)}</td>
      <td>${esc(c.detected_language)}</td>
      <td>${statusBadge(c.status)}</td>
      <td class="muted">${fmtDate(c.created_at)}</td>
    </tr>`).join('') : '<tr><td colspan="7" class="empty-state">No complaints match these filters.</td></tr>';
}
loadList();
"""
        return HttpResponse(render_page("complaints", "All Complaints", body, script))


class ComplaintDetailPageView(View):
    def get(self, request, complaint_id):
        body = f"""
<a href="/complaints/" class="btn btn-sm btn-ghost" style="margin-bottom:16px;">&larr; Back to all complaints</a>
<div id="detailRoot"><span class="spinner"></span> Loading…</div>
"""
        script = f"""
const COMPLAINT_ID = "{complaint_id}";

async function loadDetail(){{
  const res = await fetch(API + '/api/complaints/' + COMPLAINT_ID + '/');
  if(!res.ok){{ document.getElementById('detailRoot').innerHTML = '<div class="card pad empty-state">Complaint not found.</div>'; return; }}
  const c = await res.json();
  const breach = c.regulatory_breach && c.regulatory_breach.detected;

  document.getElementById('detailRoot').innerHTML = `
    <div class="page-head">
      <div class="eyebrow">Ref ${{c.complaint_id.slice(0,8).toUpperCase()}}</div>
      <h1>${{titleCase(c.category)}}</h1>
      <p>${{sevBadge(c.severity)}} &nbsp; ${{statusBadge(c.status)}} &nbsp; ${{channelLabel(c.channel)}} &nbsp; ${{esc(c.detected_language)}}</p>
    </div>
    <div class="grid-2">
      <div>
        <div class="card pad" style="margin-bottom:14px;">
          <div class="section-title">Original Complaint</div>
          <p style="font-size:14px; line-height:1.7;">${{esc(c.original_text)}}</p>
        </div>
        ${{breach ? `
        <div class="card pad" style="margin-bottom:14px; border-left:4px solid var(--danger);">
          <div class="section-title">Regulatory Alert &mdash; ${{regBadge(c.regulatory_breach.regulator)}}</div>
          <p class="muted" style="font-size:12.5px; margin-top:6px;">Triggered by: ${{c.regulatory_breach.triggers.map(esc).join(', ')}}</p>
        </div>` : ''}}
        <div class="card pad">
          <div class="section-title">Draft Response</div>
          <p style="font-size:13.5px; line-height:1.7; color:var(--ink-soft);">${{esc(c.draft_response)}}</p>
        </div>
      </div>
      <div>
        <div class="card pad" style="margin-bottom:14px;">
          <div class="section-title">Details</div>
          <div class="kv">
            <div class="k">Category</div><div>${{titleCase(c.category)}}</div>
            <div class="k">Confidence</div><div>${{Math.round(c.category_confidence*100)}}%</div>
            <div class="k">Sentiment</div><div>${{c.sentiment_score}}</div>
            <div class="k">Sender</div><div class="mono">${{esc(c.sender_id)}}</div>
            <div class="k">Received</div><div>${{fmtDate(c.created_at)}}</div>
            <div class="k">Resolved</div><div>${{c.resolved_at ? fmtDate(c.resolved_at) : '—'}}</div>
          </div>
        </div>
        <div class="card pad">
          <div class="section-title">Agent Workflow</div>
          <div class="field"><label>Status</label>
            <select id="drawerStatus">
              <option value="new" ${{c.status==='new'?'selected':''}}>New</option>
              <option value="in_progress" ${{c.status==='in_progress'?'selected':''}}>In Progress</option>
              <option value="resolved" ${{c.status==='resolved'?'selected':''}}>Resolved</option>
              <option value="escalated" ${{c.status==='escalated'?'selected':''}}>Escalated</option>
            </select>
          </div>
          <div class="field"><label>Agent Response</label>
            <textarea id="drawerResponse" placeholder="Write the final reply sent to the customer…">${{esc(c.agent_response)}}</textarea>
          </div>
          <button class="btn btn-accent" onclick="saveComplaint()">Save Changes</button>
          <div class="notice" id="notice"></div>
        </div>
      </div>
    </div>
  `;
}}

function showNotice(kind, msg){{
  const el = document.getElementById('notice');
  el.className = 'notice show ' + (kind === 'ok' ? 'notice-ok' : 'notice-err');
  el.textContent = msg;
}}

async function saveComplaint(){{
  const statusVal = document.getElementById('drawerStatus').value;
  const agentResponse = document.getElementById('drawerResponse').value;
  const res = await fetch(API + '/api/complaints/' + COMPLAINT_ID + '/', {{
    method:'PATCH', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{status: statusVal, agent_response: agentResponse}})
  }});
  if(res.ok){{ showNotice('ok', 'Saved.'); loadDetail(); }} else {{ showNotice('err', 'Could not save changes.'); }}
}}
loadDetail();
"""
        return HttpResponse(render_page("complaints", "Complaint Detail", body, script))


class TrendsView(View):
    def get(self, request):
        body = """
<div class="page-head">
  <div class="eyebrow">Last 30 Days</div>
  <h1>Trends &amp; Analytics</h1>
  <p>Volume, category mix and regulatory exposure at a glance — the numbers an ops lead would pull every morning.</p>
</div>

<div class="chart-grid" style="grid-template-columns:1fr 1fr;">
  <div class="card pad chart-card"><h3>Daily Volume (7 days)</h3><canvas id="chDaily"></canvas></div>
  <div class="card pad chart-card"><h3>By Status</h3><canvas id="chStatus"></canvas></div>
</div>
<div class="chart-grid">
  <div class="card pad chart-card"><h3>By Category</h3><canvas id="chCat"></canvas></div>
  <div class="card pad chart-card"><h3>By Severity</h3><canvas id="chSev"></canvas></div>
  <div class="card pad chart-card"><h3>By Language</h3><canvas id="chLang"></canvas></div>
</div>
"""
        script = """
let charts = {};
function bar(id, labels, data, color, horizontal){
  if(charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), {
    type: 'bar',
    data: { labels, datasets: [{ data, backgroundColor: color, borderRadius: 4 }] },
    options: { indexAxis: horizontal ? 'y' : 'x', plugins:{legend:{display:false}}, scales:{x:{beginAtZero:true, ticks:{precision:0}}} }
  });
}
function line(id, labels, data){
  if(charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), {
    type: 'line',
    data: { labels, datasets: [{ data, borderColor:'#e8491d', backgroundColor:'rgba(232,73,29,.12)', fill:true, tension:.35, pointRadius:3 }] },
    options: { plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true, ticks:{precision:0}}} }
  });
}
function doughnut(id, labels, data, colors){
  if(charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), {
    type: 'doughnut',
    data: { labels, datasets: [{ data, backgroundColor: colors }] },
    options: { plugins:{legend:{position:'bottom', labels:{boxWidth:10, font:{size:10}}}} }
  });
}
async function loadTrends(){
  const t = await fetch(API + '/api/trends/').then(r=>r.json());
  line('chDaily', t.daily_volume.map(d=>d.date.slice(5)), t.daily_volume.map(d=>d.count));
  doughnut('chStatus', t.by_status.map(s=>titleCase(s.status)), t.by_status.map(s=>s.count), ['#2f5fa8','#c8891a','#1f7a6c','#c73a3a']);
  bar('chCat', t.by_category.map(c=>titleCase(c.category)), t.by_category.map(c=>c.count), '#e8491d', true);
  bar('chSev', t.by_severity.map(c=>'Sev '+c.severity), t.by_severity.map(c=>c.count), '#c8891a', false);
  bar('chLang', t.by_language.map(c=>c.detected_language), t.by_language.map(c=>c.count), '#1f7a6c', true);
}
loadTrends();
"""
        return HttpResponse(render_page("trends", "Trends & Analytics", body, script))


class AlertsView(View):
    def get(self, request):
        body = """
<div class="page-head">
  <div class="eyebrow">RBI · TRAI · SEBI</div>
  <h1>Regulatory Alerts</h1>
  <p>Complaints whose text matched a regulator trigger phrase — these carry acknowledgement SLAs.</p>
</div>

<div class="tabs">
  <button class="tab active" id="tabAll" onclick="setTab('')">All</button>
  <button class="tab" id="tabOpen" onclick="setTab('open')">Open</button>
  <button class="tab" id="tabAck" onclick="setTab('acknowledged')">Acknowledged</button>
</div>

<div class="card">
  <div class="table-wrap"><table>
    <thead><tr><th>Regulator</th><th>Complaint</th><th>Category</th><th>Severity</th><th>Triggered By</th><th>Raised</th><th></th></tr></thead>
    <tbody id="alertBody"><tr><td colspan="7" class="empty-state"><span class="spinner"></span> Loading…</td></tr></tbody>
  </table></div>
</div>
"""
        script = """
let currentTab = '';
function setTab(t){
  currentTab = t;
  document.getElementById('tabAll').classList.toggle('active', t==='');
  document.getElementById('tabOpen').classList.toggle('active', t==='open');
  document.getElementById('tabAck').classList.toggle('active', t==='acknowledged');
  loadAlerts();
}
async function toggleAck(id, next){
  await fetch(API + '/api/alerts/' + id + '/', {
    method:'PATCH', headers:{'Content-Type':'application/json'}, body: JSON.stringify({acknowledged: next})
  });
  loadAlerts();
}
async function loadAlerts(){
  const params = currentTab ? ('?status=' + currentTab) : '';
  const data = await fetch(API + '/api/alerts/' + params).then(r=>r.json());
  document.getElementById('alertBody').innerHTML = data.results.length ? data.results.map(a => `
    <tr>
      <td>${regBadge(a.regulator)}</td>
      <td><a href="/complaints/${a.complaint_id}/" style="text-decoration:underline;">${esc(a.complaint_text)}</a></td>
      <td>${titleCase(a.complaint_category)}</td>
      <td>${sevBadge(a.complaint_severity)}</td>
      <td class="muted" style="font-size:12px;">${a.triggers.map(esc).join(', ')}</td>
      <td class="muted">${fmtDate(a.created_at)}</td>
      <td>${a.acknowledged
          ? `<button class="btn btn-sm btn-ghost" onclick="toggleAck(${a.id}, false)">Reopen</button>`
          : `<button class="btn btn-sm btn-accent" onclick="toggleAck(${a.id}, true)">Acknowledge</button>`}</td>
    </tr>`).join('') : '<tr><td colspan="7" class="empty-state">No alerts in this view.</td></tr>';
}
loadAlerts();
"""
        return HttpResponse(render_page("alerts", "Regulatory Alerts", body, script))


# ══════════════════════════════════════════════════════════════
#  SECTION 6 — URL ROUTING
# ══════════════════════════════════════════════════════════════

urlpatterns = [
    # pages
    path("",                                    HomeView.as_view(),              name="home"),
    path("submit/",                              SubmitView.as_view(),            name="submit"),
    path("complaints/",                          ComplaintsListView.as_view(),    name="complaints-list"),
    path("complaints/<uuid:complaint_id>/",       ComplaintDetailPageView.as_view(), name="complaint-detail-page"),
    path("trends/",                               TrendsView.as_view(),            name="trends-page"),
    path("alerts/",                                AlertsView.as_view(),            name="alerts-page"),
    # api
    path("api/complaints/",                        ComplaintListCreateAPI.as_view(), name="api-complaints-list"),
    path("api/complaints/<str:complaint_id>/",      ComplaintDetailAPI.as_view(),    name="api-complaint-detail"),
    path("api/trends/",                              TrendsAPI.as_view(),              name="api-trends"),
    path("api/alerts/",                                AlertListAPI.as_view(),           name="api-alerts-list"),
    path("api/alerts/<int:alert_id>/",                 AlertDetailAPI.as_view(),         name="api-alert-detail"),
    path("api/seed/",                                    SeedDataAPI.as_view(),            name="api-seed"),
]


# ══════════════════════════════════════════════════════════════
#  SECTION 7 — DB SETUP + SERVER STARTUP
# ══════════════════════════════════════════════════════════════

def create_tables():
    from django.db import connection
    with connection.schema_editor() as schema_editor:
        for model in (Complaint, RegulatoryAlert):
            try:
                schema_editor.create_model(model)
                print(f"  ✅ Created table: {model._meta.db_table}")
            except Exception:
                pass  # table already exists


def run_migrations():
    from io import StringIO
    call_command("migrate", "--run-syncdb", verbosity=0, stdout=StringIO())


def print_banner():
    print("""
+------------------------------------------------------------+
|   COMPLAINT INTEL - Multilingual Complaint Platform         |
|   Django + DRF + rule-based NLP  --  100% offline            |
+------------------------------------------------------------+
|  Overview     ->  http://127.0.0.1:8000/                    |
|  File one     ->  http://127.0.0.1:8000/submit/             |
|  All records  ->  http://127.0.0.1:8000/complaints/         |
|  Trends       ->  http://127.0.0.1:8000/trends/             |
|  Alerts       ->  http://127.0.0.1:8000/alerts/             |
|  REST API     ->  http://127.0.0.1:8000/api/complaints/     |
+------------------------------------------------------------+
|  No API keys required. Click "Load Demo Data" on the         |
|  Overview page to populate the dashboard instantly.          |
+------------------------------------------------------------+
    """)


if __name__ == "__main__":
    print("\nStarting Complaint Intel...")
    print("Setting up database...")
    run_migrations()
    create_tables()
    print_banner()

    sys.argv = ["manage.py", "runserver", "127.0.0.1:8000", "--noreload"]
    call_command("runserver", "127.0.0.1:8000", use_reloader=False)