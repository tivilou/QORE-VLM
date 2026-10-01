import unittest
from pathlib import Path
import json
import hashlib
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from applications.rag.dpr_span_relevance_quantum import (
    SpanFusionError,
    build_feature_matrix,
    classical_born_scores,
    classical_linear_scores,
    quantum_statevector_scores,
    ranked_indices,
    redundancy_aware_indices,
)
from scripts.collab.five_ideas import run_dpr_span_relevance_quantum_screen_100 as screen


def _rows(count=50):
    return [
        {
            "candidate_id": "p{}".format(index),
            "retrieved_rank": index + 1,
            "relevance_logit": float(50 - index),
            "span_logit": float(index % 7),
            "span_margin": float(index % 5) / 2.0,
            "start_entropy": float(index % 3) / 3.0,
            "end_entropy": float(index % 4) / 4.0,
            "text": "fragment {} with answer-bearing details".format(index),
            "title": "title {}".format(index),
        }
        for index in range(count)
    ]


class DprSpanRelevanceQuantumTests(unittest.TestCase):
    def test_feature_projection_uses_only_reader_outputs_and_passage_text(self):
        case = {
            "case_number": 1,
            "top_50": [
                {
                    "id": "p{}".format(index),
                    "retrieved_rank": index + 1,
                    "title": "title {}".format(index),
                    "text": "text {}".format(index),
                    "evidence": {"positive_consensus": True},
                    "gold_answer": "not an online field",
                }
                for index in range(50)
            ],
        }
        trace_rows = [
            {
                "case_number": 1,
                "candidate_index": index,
                "candidate_id": "p{}".format(index),
                "retrieved_rank": index + 1,
                "relevance_logit": 1.0,
                "span_logit": 2.0,
                "span_margin": 0.5,
                "start_entropy": 0.2,
                "end_entropy": 0.3,
                "direct_consensus": True,
            }
            for index in range(50)
        ]
        projected = screen._project_online_candidates(case, trace_rows)
        self.assertEqual(set(projected[0]), {
            "candidate_id", "retrieved_rank", "title", "text", "relevance_logit",
            "span_logit", "span_margin", "start_entropy", "end_entropy",
        })
        self.assertNotIn("evidence", projected[0])
        self.assertNotIn("gold_answer", projected[0])
        self.assertNotIn("direct_consensus", projected[0])

    def test_feature_normalization_is_finite_and_handles_constant_columns(self):
        rows = _rows()
        features = build_feature_matrix(rows)
        self.assertEqual(features.shape, (50, 4))
        self.assertTrue(np.all(np.isfinite(features)))
        constant_rows = [dict(row, relevance_logit=1.0, span_logit=1.0, span_margin=1.0,
                              start_entropy=1.0, end_entropy=1.0) for row in rows]
        constant = build_feature_matrix(constant_rows)
        self.assertTrue(np.allclose(constant, 0.0))

    def test_statevector_matches_independent_classical_born_control(self):
        rng = np.random.default_rng(17)
        features = rng.uniform(-1.0, 1.0, size=(50, 4))
        quantum = quantum_statevector_scores(features)
        classical = classical_born_scores(features)
        self.assertEqual(quantum.shape, (50,))
        self.assertTrue(np.all(np.isfinite(quantum)))
        self.assertLessEqual(float(np.max(np.abs(quantum - classical))), 1e-10)
        self.assertTrue(np.array_equal(quantum, quantum_statevector_scores(features)))

    def test_invalid_features_fail_closed(self):
        with self.assertRaises(SpanFusionError):
            classical_linear_scores(np.zeros((49, 4)))
        invalid = np.zeros((50, 4))
        invalid[0, 0] = np.nan
        with self.assertRaises(SpanFusionError):
            quantum_statevector_scores(invalid)

    def test_top5_is_unique_and_stably_tie_broken(self):
        rows = _rows()
        scores = np.ones(50)
        selected = ranked_indices(scores, rows)
        self.assertEqual(selected, [0, 1, 2, 3, 4])
        self.assertEqual(len({rows[index]["candidate_id"] for index in selected}), 5)

    def test_redundancy_penalty_avoids_near_duplicate(self):
        rows = _rows()
        repeated = "same passage text with a common evidence fragment"
        rows[0]["text"] = repeated
        rows[1]["text"] = repeated
        scores = np.linspace(1.0, 0.0, 50)
        without = ranked_indices(scores, rows)
        with_penalty = redundancy_aware_indices(scores, rows, penalty=0.3)
        self.assertIn(0, without)
        self.assertIn(1, without)
        self.assertIn(0, with_penalty)
        self.assertNotIn(1, with_penalty)

    def test_screen_scoring_does_not_consult_silver_selector_or_labels(self):
        rows = _rows()
        case = {
            "case_number": 1,
            "selectors": [
                {"selector_id": "qore_common_order", "selected_top_5": [{"id": "p{}".format(i)} for i in range(5)]},
                {"selector_id": "topk_common_order", "selected_top_5": [{"id": "p{}".format(i)} for i in range(5)]},
                {"selector_id": "silver_oracle_common_order", "selected_top_5": [{"id": "p{}".format(i)} for i in range(45, 50)]},
            ],
        }
        prediction = screen._case_predictions(case, rows)
        self.assertEqual(prediction["selections"]["answer_scorer_topk"]["indices"], [0, 1, 2, 3, 4])
        self.assertEqual(prediction["selections"]["quantum_span_interaction"]["indices"],
                         prediction["selections"]["classical_matched_born"]["indices"])
        self.assertLessEqual(prediction["quantum_classical_parity_max_abs"], 1e-10)

    def test_sample_trace_v2_contains_complete_values_and_posthoc_boundary(self):
        cases = []
        predictions = []
        case_results = []
        score_samples = []
        for case_number in range(1, 101):
            rows = _rows()
            top50 = []
            for index, row in enumerate(rows):
                top50.append({
                    "id": row["candidate_id"],
                    "retrieved_rank": row["retrieved_rank"],
                    "title": row["title"],
                    "text": row["text"],
                    "evidence": {
                        "positive_consensus": index % 3 == 0,
                        "direct_consensus": index % 5 == 0,
                    },
                })
            case = {
                "case_number": case_number,
                "top_50": top50,
                "selectors": [
                    {"selector_id": "qore_common_order", "selected_top_5": [{"id": "p{}".format(i)} for i in range(5)]},
                    {"selector_id": "topk_common_order", "selected_top_5": [{"id": "p{}".format(i)} for i in range(5)]},
                    {"selector_id": "silver_oracle_common_order", "selected_top_5": [{"id": "p{}".format(i)} for i in range(45, 50)]},
                ],
            }
            prediction = screen._case_predictions(case, rows)
            metrics = screen._posthoc_case_metrics(case, prediction)
            cases.append(case)
            predictions.append(prediction)
            case_results.append(metrics)
            candidate_rows = prediction["score_trace_rows"]
            score_samples.append({
                "sample_id": "case-{:03d}".format(case_number),
                "candidate_ids": [row["candidate_id"] for row in candidate_rows],
                "retrieved_ranks": [row["retrieved_rank"] for row in candidate_rows],
                "numeric_matrix": [row["numeric_values"] for row in candidate_rows],
                "selection_ranks": [row["selection_rank"] for row in candidate_rows],
            })
        score_trace = {
            "schema_version": "rag.dpr_span_relevance_quantum_screen_100.score_trace.v1",
            "input_sha256": screen.DEFAULT_INPUT_SHA256,
            "reader_trace_sha256": screen.TRACE_SHA256,
            "online_labels_used": False,
            "columns": list(screen.SCORE_TRACE_COLUMNS),
            "samples": score_samples,
        }
        with tempfile.TemporaryDirectory(prefix="dpr-span-trace-test-") as temp_dir:
            score_trace_path = Path(temp_dir) / "score_trace.json"
            score_trace_path.write_text(json.dumps(score_trace, separators=(",", ":")) + "\n", encoding="utf-8")
            artifact = {
                "path": "score_trace.json",
                "sha256": screen._sha256(score_trace_path),
                "bytes": score_trace_path.stat().st_size,
                "visibility": "portal",
            }
            trace = screen._build_sample_trace(
                "synthetic",
                Path(__file__).resolve(),
                123,
                artifact,
                cases,
                predictions,
                case_results,
                "b" * 64,
            )
            sample_trace_path = Path(temp_dir) / "sample_trace.json"
            sample_trace_path.write_text(json.dumps(trace, separators=(",", ":")) + "\n", encoding="utf-8")
            screen.validate_sample_trace_artifacts(
                sample_trace_path,
                score_trace_path,
                screen._sha256(sample_trace_path),
                sample_trace_path.stat().st_size,
                artifact["sha256"],
                artifact["bytes"],
                predictions,
                case_results,
            )
            score_trace_path.write_text(score_trace_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaises(screen.ScreenError):
                screen.validate_sample_trace_artifacts(
                    sample_trace_path,
                    score_trace_path,
                    screen._sha256(sample_trace_path),
                    sample_trace_path.stat().st_size,
                    artifact["sha256"],
                    artifact["bytes"],
                    predictions,
                    case_results,
                )
        self.assertEqual(trace["schema_version"], "sample-trace.v2")
        self.assertEqual(trace["coverage"]["retrieval"], "not_applicable")
        self.assertEqual(len(trace["samples"]), 100)
        self.assertEqual(trace["samples"][0]["representations"][0]["value_capture"]["stored_element_count"], 400)
        self.assertNotIn("gold_answers", trace["samples"][0]["stages"][4]["inputs"])
        fixture_path = os.environ.get("SAMPLE_TRACE_PREFLIGHT_OUTPUT")
        if fixture_path:
            Path(fixture_path).write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def test_exchange_directory_and_streamed_upload_verify_receipt(self):
        observed = {"directory": None, "upload": None}

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self.assert_authorized()
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                observed["directory"] = request["path"]
                self._send_json(201, {"path": request["path"]})

            def do_PUT(self):
                self.assert_authorized()
                body = self.rfile.read(int(self.headers["Content-Length"]))
                path = self.path.removeprefix("/upload/")
                observed["upload"] = (path, body)
                self._send_json(201, {
                    "path": path,
                    "size_bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                })

            def assert_authorized(self):
                if self.headers.get("Authorization") != "Bearer fixture-token":
                    self._send_json(401, {"error": "unauthorized"})
                    raise AssertionError("missing exchange bearer header")

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = "http://127.0.0.1:{}".format(server.server_address[1])
            relative_dir = "five_ideas/synthetic_screen/20261001T000000Z"
            with tempfile.TemporaryDirectory(prefix="dpr-exchange-test-") as temp_dir:
                source = Path(temp_dir) / "payload.json"
                source.write_bytes(b'{"synthetic":true}\n')
                screen._create_exchange_directory(base_url, "fixture-token", relative_dir)
                receipt = screen._upload_file(base_url, "fixture-token", source, relative_dir + "/payload.json")
            self.assertEqual(observed["directory"], relative_dir)
            self.assertEqual(observed["upload"], (relative_dir + "/payload.json", b'{"synthetic":true}\n'))
            self.assertEqual(receipt["size_bytes"], len(b'{"synthetic":true}\n'))
            self.assertEqual(receipt["sha256"], hashlib.sha256(b'{"synthetic":true}\n').hexdigest())
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


if __name__ == "__main__":
    unittest.main()
