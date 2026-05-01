from django.test import SimpleTestCase
from django.urls import reverse


class GemmaJudgeHomePageTests(SimpleTestCase):
    def test_home_page_renders_template(self):
        response = self.client.get(reverse("gemmajudge-home"))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "GemmaJudge/GemmaJudgeHomePage.html")
        self.assertContains(response, "Gemma Oracle")
        self.assertContains(response, 'data-use-api="true"')
        self.assertContains(response, "Quick Response")
        self.assertContains(response, "Thinking")
        self.assertContains(response, 'id="library-view-toggle"')
        self.assertContains(response, 'data-library-view="cards"')
        self.assertContains(response, 'data-library-view="rules"')
        self.assertContains(response, 'id="context-summary-grid"')
        self.assertContains(response, "markdownRenderer.js")
