from django.contrib.auth.models import User
from django.test import TestCase
from pathlib import Path
from unittest.mock import patch

from ml.bert_service import BertModelUnavailable


class PredictAccessTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        checkpoint = (
            Path(__file__).resolve().parents[1] / 'ml' / 'bert_fake_news' / 'best_checkpoint'
        )
        # The weights are intentionally not committed to Git, so the live
        # inference test only runs on a machine that has the full checkpoint.
        cls.bert_artifact_exists = all(
            (checkpoint / name).is_file()
            for name in ('config.json', 'model.safetensors', 'tokenizer.json')
        )

    def setUp(self):
        self.user = User.objects.create_user(
            username='test-user',
            password='test-password-123',
        )

    def test_guest_is_redirected_to_login_before_analysis(self):
        response = self.client.post('/predict/', {'news': 'A sufficiently long article that should not be analyzed for a guest user.'})

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response['Location'], '/login/?next=/predict/')

    def test_authenticated_user_can_submit_text_for_analysis(self):
        if not self.bert_artifact_exists:
            self.skipTest('Train ml/bert_training.py before running the live prediction test.')
        self.client.force_login(self.user)
        response = self.client.post(
            '/predict/',
            {'news': 'This article reports that a local energy company announced a new solar storage project after a successful pilot trial.'},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Verification Report')

    def test_empty_submission_is_rejected_before_model_inference(self):
        self.client.force_login(self.user)
        response = self.client.post('/predict/', {'news': ''})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'No input content provided')

    @patch('detector.views.bert_predict', side_effect=BertModelUnavailable('model unavailable'))
    def test_missing_bert_model_returns_actionable_error(self, _bert_predict):
        self.client.force_login(self.user)
        response = self.client.post(
            '/predict/',
            {'news': 'This is a sufficiently long article text for validation.'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'ML model is currently unavailable')
