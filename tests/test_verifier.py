import asyncio
import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hw2


def response(**changes):
    verdict = dict(identity_resolved=True, person_id=42,
                   checked_fields=['name', 'city', 'jobs', 'education', 'skills'],
                   discrepancies=[], valid=True)
    verdict.update(changes)
    profile = {'id': 42, 'experience': [], 'education': [], 'skills': []}
    return {'structured_response': verdict, 'messages': [
        SimpleNamespace(name='get_linkedin_profile',
                        content=[{'type': 'text', 'text': json.dumps(profile)}])
    ]}


class VerdictTests(unittest.TestCase):
    def test_complete_verified_cv(self):
        self.assertEqual(hw2._verified_score(response()), .95)

    def test_one_discrepancy_overrides_valid(self):
        self.assertEqual(hw2._verified_score(response(
            discrepancies=['CV graduation 2017; profile graduation 2016'])), .05)

    def test_incomplete_audit_cannot_pass(self):
        self.assertLessEqual(hw2._verified_score(response(checked_fields=['name'])), .5)

    def test_unresolved_identity_cannot_pass(self):
        self.assertLessEqual(hw2._verified_score(response(identity_resolved=False)), .5)

    def test_wrong_profile_cannot_pass(self):
        self.assertLessEqual(hw2._verified_score(response(person_id=99)), .5)

    def test_unretrieved_profile_cannot_pass(self):
        result = response()
        result['messages'] = []
        self.assertLessEqual(hw2._verified_score(result), .5)

    def test_failed_tool_cannot_supply_evidence(self):
        result = response()
        result['messages'][0].status = 'error'
        self.assertLessEqual(hw2._verified_score(result), .5)

    def test_real_message_history(self):
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
        result = response()
        result['messages'] = [HumanMessage(content='CV data'),
                              AIMessage(content=''),
                              ToolMessage(content=json.dumps({'id': 42, 'experience': []}),
                                          name='get_linkedin_profile', tool_call_id='call_1')]
        self.assertEqual(hw2._verified_score(result), .95)

    def test_missing_structure_rejected(self):
        with self.assertRaises(ValueError):
            hw2._verified_score({'messages': []})


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_langchain_tool_and_structured_output_cycle(self):
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool

        class ToolModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        @tool
        def get_linkedin_profile(person_id: int) -> str:
            """Retrieve the test profile."""
            return json.dumps({'id': person_id, 'experience': []})

        verdict = response()['structured_response']
        verdict['identity_evidence'] = 'Test identity evidence'
        model = ToolModel(responses=[
            AIMessage(content='', tool_calls=[{'name': 'get_linkedin_profile',
                      'args': {'person_id': 42}, 'id': 'lookup'}]),
            AIMessage(content='', tool_calls=[{'name': 'Verification',
                      'args': verdict, 'id': 'verdict'}]),
        ])
        with patch('langchain_deepseek.ChatDeepSeek', return_value=model):
            agent = hw2.build_agent([get_linkedin_profile])
        self.assertEqual(await hw2.score_cvs(agent, {'sample.pdf': 'test CV'}), {'sample.pdf': .95})

    async def test_parallelism_and_one_output_per_cv(self):
        class Agent:
            active = peak = 0

            async def ainvoke(self, data, config):
                self.active += 1
                self.peak = max(self.peak, self.active)
                await asyncio.sleep(.01)
                self.active -= 1
                return response()

        agent = Agent()
        cvs = {f'document_{i}.pdf': 'sample' for i in range(9)}
        scores = await hw2.score_cvs(agent, cvs)
        self.assertEqual(scores.keys(), cvs.keys())
        self.assertEqual(agent.peak, 3)
        self.assertTrue(all(math.isfinite(s) and 0 <= s <= 1 for s in scores.values()))

    async def test_failure_isolated_and_retry_succeeds(self):
        class Agent:
            calls = {}

            async def ainvoke(self, data, config):
                cv = json.loads(data['messages'][0]['content'].split('\n', 1)[1])['cv_text']
                self.calls[cv] = self.calls.get(cv, 0) + 1
                if cv == 'broken' or (cv == 'transient' and self.calls[cv] == 1):
                    raise RuntimeError('simulated failure')
                return response()

        agent = Agent()
        scores = await hw2.score_cvs(agent, {'a.pdf': 'broken', 'b.pdf': 'good', 'c.pdf': 'transient'})
        self.assertEqual(scores, {'a.pdf': .5, 'b.pdf': .95, 'c.pdf': .95})
        self.assertEqual(agent.calls['transient'], 2)

    async def test_timeout_returns_score(self):
        class Agent:
            async def ainvoke(self, data, config):
                await asyncio.Event().wait()

        original = asyncio.wait_for

        async def short_timeout(awaitable, timeout):
            return await original(awaitable, .01)

        with patch.object(hw2.asyncio, 'wait_for', short_timeout):
            self.assertEqual(await hw2.score_cvs(Agent(), {'a.pdf': 'sample'}), {'a.pdf': .5})

    async def test_empty_batch(self):
        self.assertEqual(await hw2.score_cvs(None, {}), {})


if __name__ == '__main__':
    unittest.main()
