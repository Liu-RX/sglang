"""
Streaming-session throughput regression guard (scheduler-CPU bound).

Concurrent streaming sessions keep a large KV-resident history and extend it a
few tokens per turn. The model is truncated to a few layers and overlap
scheduling is disabled, shrinking the GPU forward so the per-step, O(resident
context) scheduler work (batch assembly, prefix match, KV bookkeeping) lands on
the critical path. This catches end-to-end regressions of those paths -- e.g.
#27965 (in-place fill_ids reconstruction): H200 ~3675 vs ~3367 tok/s reverted.

The floor is provisional: CI runs on H100 (1-gpu-large), so retune it from the
first CI run's printed throughput.
"""

import concurrent.futures
import json
import random
import time
import unittest
from dataclasses import dataclass
from typing import Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=200, stage="extra-a", runner_config="1-gpu-large")

NUM_HIDDEN_LAYERS = 3
NUM_CONCURRENT = 16
CONTEXT_LEN = 30000
NUM_TURNS = 100
INPUT_LEN = 10
MIN_GEN_LEN = 1
MAX_GEN_LEN = 16

# Synthetic ids, seeded per session for a deterministic, identical-input A/B.
TOKEN_ID_START = 1000
TOKEN_ID_COUNT = 1024

# Provisional; CI is H100, retune from the first run (H200: ~3675 vs ~3367).
THROUGHPUT_FLOOR_TOK_S = 3500.0


@dataclass
class SessionState:
    session_id: str
    rid: Optional[str]


def _synthetic_input_ids(length: int, seed: int) -> list[int]:
    return [TOKEN_ID_START + ((seed + i) % TOKEN_ID_COUNT) for i in range(length)]


def _stream_generate(
    base_url: str, input_ids: list[int], session: SessionState, output_len: int
) -> int:
    resp = requests.post(
        base_url + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
            "stream": True,
            "session_params": {"id": session.session_id, "rid": session.rid},
        },
        stream=True,
    )
    completion_tokens = 0
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        meta = json.loads(data)["meta_info"]
        completion_tokens = int(meta["completion_tokens"])
        session.rid = meta["id"]
    return completion_tokens


class TestStreamingSessionThroughput(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-0.6B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--json-model-override-args",
                f'{{"num_hidden_layers": {NUM_HIDDEN_LAYERS}}}',
                "--enable-streaming-session",
                "--enable-mixed-chunk",
                "--chunked-prefill-size",
                "8192",
                "--schedule-policy",
                "fcfs",
                "--max-running-requests",
                "100",
                "--disable-overlap-schedule",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _open_and_prime(self, session_index: int) -> SessionState:
        session_id = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 0, "streaming": True},
        ).json()
        session = SessionState(session_id=session_id, rid=None)
        prime_ids = _synthetic_input_ids(CONTEXT_LEN, seed=session_index)
        _stream_generate(self.base_url, prime_ids, session, output_len=1)
        return session

    def _run_turns(self, session: SessionState, session_index: int) -> int:
        rng = random.Random(session_index)
        output_tokens = 0
        for turn_index in range(NUM_TURNS):
            output_len = rng.randint(MIN_GEN_LEN, MAX_GEN_LEN)
            input_ids = _synthetic_input_ids(
                INPUT_LEN, seed=session_index * NUM_TURNS + turn_index
            )
            output_tokens += _stream_generate(
                self.base_url, input_ids, session, output_len
            )
        return output_tokens

    def test_streaming_session_throughput(self):
        """Prime bs16 sessions to ctx30k, run 100 short turns each, and assert a
        sustained output-throughput floor."""
        requests.post(self.base_url + "/flush_cache")

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CONCURRENT) as pool:
            sessions = list(pool.map(self._open_and_prime, range(NUM_CONCURRENT)))

            start = time.perf_counter()
            output_tokens = sum(
                pool.map(
                    lambda args: self._run_turns(*args),
                    [(session, idx) for idx, session in enumerate(sessions)],
                )
            )
            duration = time.perf_counter() - start

        for session in sessions:
            requests.post(
                self.base_url + "/close_session",
                json={"session_id": session.session_id},
            )

        throughput = output_tokens / duration
        print(
            f"\n[streaming-session throughput] sessions={NUM_CONCURRENT} "
            f"context={CONTEXT_LEN} turns={NUM_TURNS} layers={NUM_HIDDEN_LAYERS}\n"
            f"  output_tokens={output_tokens} duration={duration:.3f}s "
            f"throughput={throughput:.1f} tok/s (floor={THROUGHPUT_FLOOR_TOK_S})"
        )

        self.assertGreaterEqual(
            throughput,
            THROUGHPUT_FLOOR_TOK_S,
            f"output throughput {throughput:.1f} tok/s fell below "
            f"{THROUGHPUT_FLOOR_TOK_S}; per-step scheduler overhead may have regressed",
        )


if __name__ == "__main__":
    unittest.main()
