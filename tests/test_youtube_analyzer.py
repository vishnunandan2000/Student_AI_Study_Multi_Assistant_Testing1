"""
Unit and Integration Tests for YouTube Video Intelligence & Analytics Engine.
Validates:
1. YouTube URL parsing across formats (watch, share, shorts, embed, raw ID).
2. URL validation and error rejection.
3. ISO 8601 duration parsing.
4. Multi-class sentiment analysis (Positive, Neutral, Negative) and percentage calculation.
5. Recommendation structure and metric integrity.
6. Markdown and JSON export creation.
"""

import os
import sys
import json
import unittest
import tempfile
from pathlib import Path

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.modules.youtube_analyzer import (
    extract_youtube_video_id,
    validate_youtube_url,
    parse_iso_duration,
    analyze_sentiment,
    generate_video_recommendations,
    export_analysis_markdown,
    export_analysis_json
)


class TestYouTubeAnalyzer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_url_extraction(self):
        """Verify 11-character video ID is extracted accurately from all standard formats."""
        test_urls = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("http://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("abc123XYZ-_", "abc123XYZ-_")
        ]
        for url, expected in test_urls:
            extracted = extract_youtube_video_id(url)
            self.assertEqual(extracted, expected, f"Failed for URL: {url}")

    def test_invalid_url_rejection(self):
        """Verify invalid URLs or random strings are safely rejected."""
        invalids = [
            "",
            "   ",
            "https://google.com",
            "https://youtube.com/feed/subscriptions",
            "not_a_valid_id",
            "123"
        ]
        for inv in invalids:
            self.assertIsNone(extract_youtube_video_id(inv))
            valid, _, vid = validate_youtube_url(inv)
            self.assertFalse(valid)
            self.assertIsNone(vid)

    def test_iso_duration_parsing(self):
        """Verify ISO 8601 duration formatting."""
        self.assertEqual(parse_iso_duration("PT1H23M45S"), "01:23:45")
        self.assertEqual(parse_iso_duration("PT14M32S"), "14:32")
        self.assertEqual(parse_iso_duration("PT45S"), "00:45")
        self.assertEqual(parse_iso_duration("PT1H"), "01:00:00")

    def test_sentiment_analysis(self):
        """Verify multi-class sentiment categorization and scoring."""
        sample_comments = [
            "This is an amazing, clear and helpful explanation! Thank you!",
            "Great video, loved the examples.",
            "Can you explain the formula at 4:20?",
            "The audio is terrible and very hard to hear.",
            "Terrible explanation, totally confusing and worst tutorial."
        ]
        res = analyze_sentiment(sample_comments, video_title="Sample Video")

        self.assertEqual(res["total_analyzed"], 5)
        self.assertEqual(res["positive_count"], 2)
        self.assertEqual(res["neutral_count"], 1)
        self.assertEqual(res["negative_count"], 2)
        self.assertEqual(res["positive_percent"], 40.0)
        self.assertEqual(res["neutral_percent"], 20.0)
        self.assertEqual(res["negative_percent"], 40.0)
        self.assertTrue(len(res["top_positive_themes"]) > 0)
        self.assertTrue(len(res["constructive_criticism"]) > 0)

    def test_empty_comments_sentiment(self):
        """Verify safe handling of empty comment lists."""
        res = analyze_sentiment([])
        self.assertEqual(res["total_analyzed"], 0)
        self.assertEqual(res["positive_count"], 0)
        self.assertIn("No Public Comments", res["sentiment_label"])

    def test_recommendations_structure(self):
        """Verify structured recommendations schema and scores."""
        metadata = {
            "title": "Machine Learning Fundamentals",
            "channel": "AI Academy",
            "view_count": 10500,
            "like_count": 820
        }
        transcript = {
            "speaking_rate_wpm": 142.5,
            "pace_category": "Optimal Educational Pace",
            "summary_snippet": "In this lecture we cover neural network loss gradients and backpropagation."
        }
        sentiment = {
            "positive_percent": 80.0,
            "negative_percent": 5.0
        }

        recs = generate_video_recommendations(metadata, transcript, sentiment)

        self.assertIn("overall_score", recs)
        self.assertGreaterEqual(recs["overall_score"], 0)
        self.assertLessEqual(recs["overall_score"], 100)
        self.assertIn("retention_recommendations", recs)
        self.assertIn("title_and_thumbnail_improvements", recs)
        self.assertIn("content_gap_analysis", recs)
        self.assertIn("seo_and_tags", recs)
        self.assertIn("executive_summary", recs)

    def test_export_markdown_and_json(self):
        """Verify report generation in Markdown and JSON formats."""
        analysis_data = {
            "video_id": "test_video123",
            "url": "https://www.youtube.com/watch?v=test_video123",
            "analyzed_at": "2026-10-22T12:00:00Z",
            "metadata": {
                "title": "Test Education Video",
                "channel": "Tech Academy",
                "view_count": 5000,
                "like_count": 300,
                "duration": "10:15",
                "source": "Mock API",
                "authorized_analytics_note": "Note"
            },
            "transcript": {
                "speaking_rate_wpm": 135.0,
                "pace_category": "Optimal",
                "word_count": 1350
            },
            "sentiment": {
                "sentiment_label": "🌟 Highly Positive",
                "positive_percent": 85.0,
                "neutral_percent": 10.0,
                "negative_percent": 5.0,
                "positive_count": 17,
                "neutral_count": 2,
                "negative_count": 1,
                "top_positive_themes": ["Very clear demo"],
                "constructive_criticism": ["None"]
            },
            "recommendations": {
                "overall_score": 88,
                "retention_recommendations": ["Add hook"],
                "title_and_thumbnail_improvements": ["Alternative title"],
                "content_gap_analysis": ["Add cheat sheet"],
                "seo_and_tags": ["ai", "python"],
                "executive_summary": "Strong presentation."
            }
        }

        md_file = str(self.tmp_path / "report.md")
        json_file = str(self.tmp_path / "report.json")

        out_md = export_analysis_markdown(analysis_data, md_file)
        out_json = export_analysis_json(analysis_data, json_file)

        self.assertTrue(os.path.exists(out_md))
        self.assertTrue(os.path.exists(out_json))
        self.assertGreater(os.path.getsize(out_md), 100)

        with open(out_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            self.assertEqual(loaded["video_id"], "test_video123")

    def test_parse_comment_limit(self):
        """Verify comment retrieval window inputs are parsed accurately."""
        from backend.modules.youtube_analyzer import parse_comment_limit
        self.assertEqual(parse_comment_limit("10"), 10)
        self.assertEqual(parse_comment_limit("10 Comments"), 10)
        self.assertEqual(parse_comment_limit("25 Comments"), 25)
        self.assertEqual(parse_comment_limit(50), 50)
        self.assertEqual(parse_comment_limit("100 Comments"), 100)
        self.assertIsNone(parse_comment_limit("All Available Comments"))
        self.assertIsNone(parse_comment_limit("all"))
        self.assertIsNone(parse_comment_limit(None))

    def test_sentiment_analysis_preserves_all_comments_and_replies(self):
        """
        Verify that when 30 comments (e.g., 4 top-level + 26 replies) are analyzed,
        all 30 comments are preserved in classified_comments without artificial capping.
        """
        sample_30 = []
        # 4 top-level comments
        for i in range(4):
            sample_30.append({
                "id": f"top_{i}",
                "text": f"Great video tutorial demonstrating concept {i}!",
                "author": f"User_{i}",
                "likes": 15 + i,
                "is_reply": False,
                "published_at": "2026-10-20T10:00:00Z"
            })
        # 26 reply comments
        for j in range(26):
            sample_30.append({
                "id": f"rep_{j}",
                "text": f"Awesome reply explaining step {j} in detail.",
                "author": f"Replier_{j}",
                "likes": j,
                "is_reply": True,
                "published_at": "2026-10-21T11:00:00Z"
            })

        res = analyze_sentiment(sample_30, video_title="Complete Machine Learning")
        self.assertEqual(res["total_analyzed"], 30, f"Expected 30 analyzed comments, got {res['total_analyzed']}")
        self.assertEqual(len(res["classified_comments"]), 30, "All 30 comments must be preserved in classified_comments")
        self.assertEqual(res["positive_count"], 30)
        self.assertEqual(res["positive_percent"], 100.0)

        # Verify reply flags and authors are intact
        top_items = [c for c in res["classified_comments"] if not c["is_reply"]]
        reply_items = [c for c in res["classified_comments"] if c["is_reply"]]
        self.assertEqual(len(top_items), 4)
        self.assertEqual(len(reply_items), 26)

    def test_export_analysis_csv_full_comments(self):
        """Verify CSV export includes all classified comments and metadata columns."""
        from backend.modules.youtube_analyzer import export_analysis_csv
        import csv
        sample_comments = [
            {"id": "1", "author": "Alice", "text": "Superb demo!", "category": "Positive", "score": 0.8, "likes": 10, "is_reply": False, "published_at": "2026-10-20"},
            {"id": "2", "author": "Bob", "text": "Thanks Alice!", "category": "Positive", "score": 0.8, "likes": 2, "is_reply": True, "published_at": "2026-10-21"}
        ]
        data = {
            "video_id": "vid123",
            "metadata": {"title": "Test Title"},
            "sentiment": {"classified_comments": sample_comments}
        }
        csv_file = str(self.tmp_path / "test_out.csv")
        out_path = export_analysis_csv(data, csv_file)
        self.assertTrue(os.path.exists(out_path))

        with open(out_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            self.assertEqual(len(reader), 3) # Header + 2 rows
            self.assertIn("Author", reader[0])
            self.assertIn("Is Reply", reader[0])
            self.assertEqual(reader[1][2], "Alice")
            self.assertEqual(reader[1][3], "No")
            self.assertEqual(reader[2][2], "Bob")
            self.assertEqual(reader[2][3], "Yes")


if __name__ == "__main__":
    unittest.main()
