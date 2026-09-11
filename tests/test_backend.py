import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from aimo3_inference.backends import OpenAICompatibleBackend


class _Handler(BaseHTTPRequestHandler):
    request_payload = None
    authorization = None

    def do_GET(self):
        body = json.dumps({"data": [{"id": "openai/gpt-oss-20b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        size = int(self.headers["Content-Length"])
        type(self).request_payload = json.loads(self.rfile.read(size))
        type(self).authorization = self.headers.get("Authorization")
        body = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "execute_python",
                                        "arguments": '{"code":"print(6*7)"}',
                                    },
                                }
                            ],
                        },
                        "logprobs": {
                            "content": [
                                {
                                    "token": "x",
                                    "logprob": -0.2,
                                    "top_logprobs": [
                                        {"token": "x", "logprob": -0.2},
                                        {"token": "y", "logprob": -1.8},
                                    ],
                                }
                            ]
                        },
                    }
                ],
                "usage": {"completion_tokens": 12},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class OpenAICompatibleBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address
        cls.backend = OpenAICompatibleBackend(f"http://{host}:{port}/v1", api_key="local")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def test_lists_models(self) -> None:
        self.assertEqual(self.backend.list_models(), ("openai/gpt-oss-20b",))

    def test_parses_tool_call_and_logprobs(self) -> None:
        completion = self.backend.complete(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "6*7?"}],
            tools=[{"type": "function", "function": {"name": "execute_python"}}],
            temperature=1.0,
            top_p=1.0,
            max_tokens=100,
            seed=42,
            timeout=2,
            top_logprobs=2,
            reasoning_effort="high",
        )
        self.assertEqual(completion.tool_calls[0].name, "execute_python")
        self.assertEqual(completion.token_logprobs[0]["x"], -0.2)
        self.assertEqual(completion.usage["completion_tokens"], 12)
        self.assertEqual(_Handler.authorization, "Bearer local")
        self.assertFalse(_Handler.request_payload["parallel_tool_calls"])
        self.assertEqual(_Handler.request_payload["reasoning_effort"], "high")


if __name__ == "__main__":
    unittest.main()
