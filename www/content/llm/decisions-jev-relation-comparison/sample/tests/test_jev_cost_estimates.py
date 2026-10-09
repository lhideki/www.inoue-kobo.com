"""Offline proxy-cost regression tests; no tiktoken installation or API required."""
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import estimate_jev_cost as estimator
import evaluate_relations as evaluation


class RecordingEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, text, *, disallowed_special):
        self.calls.append((text, disallowed_special))
        return list(text)  # Deliberate synthetic encoder, not a token-count claim.


class JevCostEstimateTests(unittest.TestCase):
    def test_fieldwise_repeats_labels_and_descriptions_without_wrapper_keys(self):
        fields = ('本文', '質問', {'候補A': '同じ説明', '候補B': '同じ説明'})
        encoder = RecordingEncoder()
        count = estimator.estimate(fields, encoder, 'fieldwise')
        expected = ['本文', '質問', '候補A', '同じ説明', '候補B', '同じ説明']
        self.assertEqual(encoder.calls, [(text, ()) for text in expected])
        self.assertEqual(count, sum(map(len, expected)))

    def test_compact_json_has_declared_order_and_structural_keys(self):
        fields = ('本文', '質問', {'B': '説明B', 'A': '説明A'})
        expected = '{"state":"本文","questions":{"relation":{"type":"choice","criteria":{"B":"説明B","A":"説明A"},"instructions":"質問"}}}'
        encoder = RecordingEncoder()
        self.assertEqual(estimator.cost_json(*fields), expected)
        self.assertEqual(estimator.estimate(fields, encoder, 'compact_content_json'), len(expected))
        self.assertEqual(encoder.calls, [(expected, ())])
        with self.assertRaises(ValueError):
            estimator.estimate(fields, encoder, 'unknown-mode')

    def test_published_scenarios_sum_and_price_exactly(self):
        data = json.loads((ROOT / 'jev-cost-estimates.json').read_text())
        combinations = set()
        for scenario in data['scenarios']:
            combinations.add((scenario['tokenizer'], scenario['serialization']))
            counts = scenario['proxy_input_tokens']
            self.assertEqual(counts['both'], counts['ja'] + counts['en'])
            for language, count in counts.items():
                self.assertEqual(Decimal(scenario['proxy_cost_usd'][language]),
                                 Decimal(count) * Decimal('0.042') / Decimal(1000000))
            provenance = scenario['provenance']
            self.assertEqual(provenance['tiktoken_version'], '0.14.0')
            self.assertEqual(provenance['encoding_asset_sha256'], estimator.ASSETS[scenario['tokenizer']])
        self.assertEqual(combinations, {(t, m) for t in ('o200k_base', 'cl100k_base')
                                       for m in ('fieldwise', 'compact_content_json')})
        self.assertEqual(data['requests_per_language'], 1418)
        self.assertEqual(data['candidates_every_request'], 36)
        self.assertIn('No guaranteed billing lower/upper bounds', data['limitations'])

    def test_published_input_audit_counts_and_disclosures(self):
        data = json.loads((ROOT / 'input-comparison-audit.json').read_text())
        self.assertEqual(set(data['reconstructed_fields_equal'].values()), {2836})
        self.assertEqual(data['actual_decisions_request_hash_matches'], {'ja': 1418, 'en': 1418})
        self.assertFalse(data['historical_jev_http_bodies_available'])
        self.assertFalse(data['historical_jev_usage_available'])
        self.assertTrue(data['candidate_set_derived_from_test_gold_relations'])
        self.assertFalse(data['gold_and_prediction_fields_in_requests'])
        self.assertEqual(data['dataset_sha256'], estimator.DATA_HASHES['dataset.jsonl'])
        self.assertEqual(data['choices_sha256'], estimator.DATA_HASHES['relation_choices.json'])

    def test_historical_source_reconstruction_matches_public_decisions_fields(self):
        source = ROOT.parents[1] / 'jev-graph-relation-evaluation/sample/run_jev.py'
        if not source.exists():
            self.skipTest('Optional historical source is not present in this standalone copy')
        build = estimator.jev_builder(source)
        row = {'subject': {'ja': {'label': '架空A'}, 'en': {'label': 'Synthetic A'}},
               'object': {'ja': {'label': '架空B'}, 'en': {'label': 'Synthetic B'}},
               'articles': {'ja': {'text': '架空の本文'}, 'en': {'text': 'Synthetic text'}},
               'expected_relation': {'id': 'not-input'}, 'id': 'not-input'}
        relations = [{'labels': {'ja': '関係A', 'en': 'Relation A'},
                      'descriptions': {'ja': '説明A', 'en': 'Description A'}}]
        for language in ('ja', 'en'):
            state, instructions, criteria = build(row, relations, language)
            request = evaluation.payload(row, relations, language)
            self.assertEqual(state, request['input'])
            self.assertEqual(instructions, request['questions'][0]['instructions'])
            self.assertEqual(list(criteria.items()), [(c['value'], c['description'])
                                                     for c in request['questions'][0]['choices']])
            self.assertNotIn('not-input', estimator.cost_json(state, instructions, criteria))

    def test_changed_source_and_asset_fail_before_execution_or_dependency_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / 'changed-input'
            bad.write_text('raise RuntimeError("must never execute")')
            with self.assertRaisesRegex(ValueError, 'source hash changed'):
                estimator.jev_builder(bad)
            with patch.object(estimator.importlib.util, 'find_spec', side_effect=AssertionError('Dependency inspected')):
                with self.assertRaisesRegex(ValueError, 'asset hash mismatch'):
                    estimator.local_encoder('o200k_base', bad)

    def test_network_audit_hook_rejects_connections_and_sends(self):
        for event in ('socket.connect', 'socket.sendto', 'socket.getaddrinfo', 'http.client.send'):
            with self.subTest(event=event), self.assertRaises(RuntimeError):
                estimator.deny_network(event, ())
        estimator.deny_network('open', ())

    def test_public_cli_needs_no_obsolete_decisions_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'dataset.jsonl').write_text('{"synthetic":true}\n')
            (root / 'relation_choices.json').write_text('{"relations":[{"synthetic":true}]}')
            expected = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                        for name in ('dataset.jsonl', 'relation_choices.json')}
            output = root / 'new.json'
            argv = ['estimate_jev_cost.py', '--jev-source', 'provided-original.py', '--data', str(root),
                    '--cl100k-asset', 'local-cl100k', '--o200k-asset', 'local-o200k', '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(sys, 'stdout', io.StringIO()), \
                 patch.dict(estimator.DATA_HASHES, expected), \
                 patch.object(estimator, 'jev_builder', return_value=lambda row, choices, lang: ('S', 'I', {'L': 'D'})), \
                 patch.object(estimator, 'local_encoder', side_effect=lambda *args: (RecordingEncoder(), {})), \
                 patch.object(evaluation.http.client, 'HTTPSConnection', side_effect=AssertionError('Network')):
                estimator.main()
                with self.assertRaisesRegex(ValueError, 'Refuse overwrite'):
                    estimator.main()
            result = json.loads(output.read_text())
            self.assertEqual(len(result['scenarios']), 4)
            self.assertEqual(result['scenarios'][0]['proxy_input_tokens'], {'ja': 4, 'en': 4, 'both': 8})


if __name__ == '__main__':
    unittest.main()
