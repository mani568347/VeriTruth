# pyrefly: ignore [missing-import]
from django.db import models
# pyrefly: ignore [missing-import]
from django.contrib.auth.models import User


# ================= USER PROFILE (users) =================
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    phone_number = models.CharField(max_length=20, blank=True)
    bio = models.TextField(blank=True)
    profile_picture = models.ImageField(upload_to='profiles/', blank=True, null=True)
    role = models.CharField(max_length=20, default='USER')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'users'

    def __str__(self):
        return f"{self.user.username} ({self.role})"


# ================= LOGIN LOGS (login_logs) =================
class LoginLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    username = models.CharField(max_length=150)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    status = models.CharField(max_length=20, default='SUCCESS')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'login_logs'

    def __str__(self):
        return f"{self.username} - {self.status} at {self.created_at}"


# ================= PREDICTION HISTORY (prediction_history) =================
class PredictionHistory(models.Model):
    username = models.CharField(max_length=100, blank=True)
    input_type = models.CharField(max_length=20)
    image = models.ImageField(upload_to='prediction_images/', blank=True, null=True)
    ai_explanation = models.TextField(blank=True)
    article_title = models.CharField(max_length=500, blank=True)
    author = models.CharField(max_length=200, blank=True)
    publish_date = models.CharField(max_length=100, blank=True)
    source_bias_explanation = models.TextField(blank=True)
    similarity_explanation = models.TextField(blank=True)
    lime_explanation = models.TextField(blank=True)
    image_explanation = models.TextField(blank=True)
    groq_explanation = models.TextField(blank=True)
    source_name = models.CharField(max_length=200, blank=True)
    uploaded_file = models.CharField(max_length=300, blank=True)
    ocr_text = models.TextField(blank=True)
    url = models.TextField(blank=True)
    article = models.TextField(blank=True)
    prediction = models.CharField(max_length=20)
    confidence = models.FloatField(default=0.0)
    trust_score = models.FloatField()
    bias_score = models.FloatField()
    source_score = models.FloatField()
    similarity_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'prediction_history'

    def __str__(self):
        return f"{self.username} - {self.prediction}"


# ================= VISITOR LOGS (visitor_logs) =================
class VisitorLog(models.Model):
    ip_address = models.GenericIPAddressField()
    page_visited = models.CharField(max_length=255)
    user_agent = models.TextField(blank=True)
    referrer = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'visitor_logs'

    def __str__(self):
        return f"{self.ip_address} visited {self.page_visited}"


# ================= REPORTS (reports) =================
class Report(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    article_title = models.CharField(max_length=255, blank=True)
    article_url = models.TextField(blank=True)
    reason = models.TextField()
    status = models.CharField(max_length=20, default='PENDING')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'reports'

    def __str__(self):
        return f"Report #{self.id} - {self.status}"


# ================= EXISTING APP MODELS =================
class Source(models.Model):
    domain = models.CharField(max_length=100, unique=True)
    score = models.IntegerField()  # 0–100

    def __str__(self):
        return self.domain


class NewsLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    user_input = models.TextField()
    text = models.TextField()
    prediction = models.CharField(max_length=20)
    confidence = models.FloatField()
    trust_score = models.FloatField()
    explanation = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username if self.user else 'Anonymous'} - {self.prediction}"


class PredictionFeedback(models.Model):
    class FeedbackType(models.TextChoices):
        CORRECT = 'CORRECT', 'Correct'
        INCORRECT = 'INCORRECT', 'Incorrect'

    class FeedbackReason(models.TextChoices):
        WRONG_PREDICTION = 'WRONG_PREDICTION', 'Wrong prediction'
        CONTEXT_MISUNDERSTOOD = 'CONTEXT_MISUNDERSTOOD', 'Context misunderstood'
        SOURCE_INFORMATION = 'SOURCE_INFORMATION', 'Source information incorrect'
        INSUFFICIENT_INFORMATION = 'INSUFFICIENT_INFORMATION', 'Insufficient information'
        OTHER = 'OTHER', 'Other'

    class ReviewStatus(models.TextChoices):
        PENDING = 'PENDING', 'Pending'
        REVIEWED = 'REVIEWED', 'Reviewed'
        VERIFIED = 'VERIFIED', 'Verified'
        REJECTED = 'REJECTED', 'Rejected'

    class VerifiedLabel(models.TextChoices):
        REAL = 'Real News', 'Real News'
        FAKE = 'Fake News', 'Fake News'

    prediction = models.ForeignKey(
        NewsLog,
        on_delete=models.CASCADE,
        related_name='feedback_entries'
    )
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='prediction_feedback')
    feedback_type = models.CharField(max_length=20, choices=FeedbackType.choices)
    reason = models.CharField(max_length=40, choices=FeedbackReason.choices, blank=True)
    comment = models.TextField(blank=True)
    predicted_label = models.CharField(max_length=20)
    prediction_confidence = models.FloatField(default=0.0)
    review_status = models.CharField(
        max_length=20,
        choices=ReviewStatus.choices,
        default=ReviewStatus.PENDING
    )
    admin_note = models.TextField(blank=True)
    verified_label = models.CharField(
        max_length=20,
        choices=VerifiedLabel.choices,
        blank=True
    )
    reviewed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='reviewed_prediction_feedback'
    )
    reviewed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'prediction_feedback'
        constraints = [
            models.UniqueConstraint(
                fields=['prediction', 'user'],
                name='unique_feedback_per_user_prediction'
            )
        ]
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.feedback_type} - {self.predicted_label}"


class AnalysisHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=500)
    verdict = models.CharField(max_length=50)
    trust_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
