"""Exercise the real ToolEnv dispatcher and MCP SDK without network access."""

import asyncio
import json
import os
import unittest
from unittest.mock import patch

import httpx

from agent_r1.env import AgentEnv
from agent_r1.env.base import Action
from agent_r1.tool import BaseTool, ToolResponse
from agent_r1.tool.parallel import MCP_URL, USER_AGENT


def action(name, **arguments):
    return Action(text=f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>")


@BaseTool.register("test_local")
class LocalTool(BaseTool):
    name = "test_local"

    async def execute(self, args, **kwargs):
        return ToolResponse(text="local result"), 1.0, {}


class ParallelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        self.calls = []
        self.mode = "success"
        self.started = asyncio.Event()
        real_client = httpx.AsyncClient

        async def handle(request):
            self.requests.append(request)
            self.assertEqual(str(request.url), MCP_URL)
            self.assertEqual(request.headers["user-agent"], USER_AGENT)
            if request.method != "POST":
                return httpx.Response(405)
            payload = json.loads(request.content)
            method = payload["method"]
            if "id" not in payload:
                return httpx.Response(202)
            if method == "initialize":
                result = {
                    "protocolVersion": payload["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "test-search", "version": "1"},
                }
            elif method == "tools/list":
                result = {
                    "tools": [
                        {"name": name, "inputSchema": {"type": "object"}}
                        for name in (() if self.mode == "missing_tool" else ("web_search", "web_fetch"))
                    ]
                }
            elif method == "tools/call":
                self.calls.append(payload["params"])
                self.started.set()
                if self.mode == "slow":
                    await asyncio.Event().wait()
                if self.mode == "http_error":
                    return httpx.Response(401, json={"error": "unauthorized"})
                if self.mode == "quota":
                    result = {"content": [{"type": "text", "text": "Anonymous quota exceeded"}], "isError": True}
                else:
                    result = {
                        "content": [
                            {"type": "text", "text": "Source: https://example.org/page?a=1&b=2\nUseful excerpt"}
                        ],
                        "isError": False,
                    }
            else:
                self.fail(f"Unexpected MCP request: {method}")
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

        def client(**kwargs):
            return real_client(transport=httpx.MockTransport(handle), **kwargs)

        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.client_patch = patch("httpx.AsyncClient", side_effect=client)
        self.env_patch.start()
        self.client_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.addCleanup(self.client_patch.stop)

    def env(self, **kwargs):
        env = AgentEnv.from_config("tool", **kwargs)
        env.reset(raw_prompt=[{"role": "user", "content": "Research a question"}])
        return env

    async def test_default_dispatch_consumption_and_identity(self):
        env = self.env()
        self.assertEqual(list(env.tools), ["parallel_search", "parallel_fetch"])
        self.assertEqual([s["function"]["name"] for s in env.tool_schemas], list(env.tools))
        obs, reward, done, _ = await env.step(action("parallel_search", query="Agent R1 tools"))
        self.assertIn("Useful excerpt", obs.messages[-1]["content"])
        self.assertIn("https://example.org/page?a=1&b=2", obs.messages[-1]["content"])
        self.assertIsNone(reward)
        self.assertFalse(done)
        await env.step(action("parallel_fetch", url="https://example.org/page?a=1&b=2", objective="Find the answer"))
        await env.step(action("parallel_search", query="Follow up research"))
        self.assertEqual(len({c["arguments"]["session_id"] for c in self.calls}), 1)
        self.assertEqual(self.calls[0]["arguments"]["search_queries"], ["Agent R1 tools"])
        self.assertEqual(self.calls[1]["arguments"]["urls"], ["https://example.org/page?a=1&b=2"])
        self.assertTrue(all("authorization" not in r.headers for r in self.requests))
        self.assertTrue(any(json.loads(r.content).get("method") == "tools/list" for r in self.requests))
        obs, _, done, _ = await env.step(Action(text="The sourced answer"))
        self.assertTrue(done)
        self.assertEqual(obs.messages[-1]["content"], "The sourced answer")

    async def test_interleaved_conversations_shared_tool_and_reset(self):
        first, second = self.env(), self.env()
        # Even reusing a tool instance must not turn its lifetime into a session.
        second.tools = first.tools
        await asyncio.gather(
            first.step(action("parallel_search", query="First question")),
            second.step(action("parallel_search", query="Second question")),
        )
        ids = {c["arguments"]["objective"]: c["arguments"]["session_id"] for c in self.calls}
        self.assertNotEqual(ids["First question"], ids["Second question"])
        await first.step(action("parallel_fetch", url="https://example.org/first"))
        self.assertEqual(self.calls[-1]["arguments"]["session_id"], ids["First question"])
        await second.step(action("parallel_fetch", url="https://example.org/second"))
        self.assertEqual(self.calls[-1]["arguments"]["session_id"], ids["Second question"])
        first.reset()
        await first.step(action("parallel_search", query="New conversation"))
        self.assertNotIn(self.calls[-1]["arguments"]["session_id"], ids.values())

    async def test_explicit_choices_disabled_access_and_unknown_tool(self):
        env = self.env(tools=["test_local"])
        obs, reward, _, _ = await env.step(action("test_local"))
        self.assertIn("local result", obs.messages[-1]["content"])
        self.assertEqual(reward, 1.0)
        obs, _, _, _ = await env.step(action("parallel_search", query="Not permitted"))
        self.assertIn("not found", obs.messages[-1]["content"])
        empty = self.env(tools=[])
        self.assertEqual(empty.tool_schemas, [])
        await empty.step(action("parallel_search", query="Not permitted"))
        disabled = self.env(tools_kwargs={"parallel": {"enabled": False}})
        obs, _, _, _ = await disabled.step(action("parallel_search", query="Disabled"))
        self.assertIn("disabled", obs.messages[-1]["content"])
        self.assertEqual(self.requests, [])

    async def test_credentials_and_explicit_anonymous_override(self):
        os.environ["PARALLEL_API_KEY"] = "environment-key"
        await self.env().step(action("parallel_search", query="Configured key"))
        self.assertTrue(all(r.headers["authorization"] == "Bearer environment-key" for r in self.requests))
        self.requests.clear()
        await self.env(tools_kwargs={"parallel": {"api_key": "saved-key"}}).step(
            action("parallel_search", query="Saved key")
        )
        self.assertTrue(all(r.headers["authorization"] == "Bearer saved-key" for r in self.requests))
        self.requests.clear()
        await self.env(tools_kwargs={"parallel": {"api_key": ""}}).step(action("parallel_search", query="Anonymous"))
        self.assertTrue(all("authorization" not in r.headers for r in self.requests))

    async def test_errors_preserve_provider_and_conversation(self):
        env = self.env()
        self.mode = "quota"
        obs, _, _, _ = await env.step(action("parallel_search", query="Quota"))
        self.assertIn("Error: Parallel Search MCP: Anonymous quota exceeded", obs.messages[-1]["content"])
        session_id = self.calls[-1]["arguments"]["session_id"]
        self.mode = "success"
        await env.step(action("parallel_search", query="Retry"))
        self.assertEqual(self.calls[-1]["arguments"]["session_id"], session_id)
        self.mode = "http_error"
        obs, _, _, _ = await env.step(action("parallel_search", query="Authentication failure"))
        self.assertIn("request failed", obs.messages[-1]["content"])
        self.assertEqual(self.calls[-1]["arguments"]["session_id"], session_id)
        self.assertEqual(list(env.tools), ["parallel_search", "parallel_fetch"])

    async def test_timeout_and_cancellation(self):
        self.mode = "slow"
        env = self.env(tools_kwargs={"parallel": {"timeout": 0.05}})
        obs, _, _, _ = await env.step(action("parallel_search", query="Timeout"))
        self.assertIn("timed out", obs.messages[-1]["content"])
        self.started.clear()
        env = self.env()
        task = asyncio.create_task(env.step(action("parallel_search", query="Cancel")))
        await asyncio.wait_for(self.started.wait(), timeout=5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_invalid_inputs_do_not_make_requests(self):
        env = self.env()
        for name, args in [
            ("parallel_search", {"query": ""}),
            ("parallel_search", {"query": ["wrong type"]}),
            ("parallel_fetch", {"url": "file:///etc/passwd"}),
            ("parallel_fetch", {"url": "https://user:pass@example.org"}),
            ("parallel_fetch", {"url": "https://example.org", "objective": "x" * 201}),
        ]:
            obs, _, _, _ = await env.step(action(name, **args))
            self.assertIn("Error:", obs.messages[-1]["content"])
        for timeout in (0, -1, "nan", "inf"):
            env = self.env(tools_kwargs={"parallel": {"timeout": timeout}})
            obs, _, _, _ = await env.step(action("parallel_search", query="Bad timeout"))
            self.assertIn("positive finite", obs.messages[-1]["content"])
        self.assertEqual(self.requests, [])

    async def test_gpt_oss_format_and_missing_server_tool(self):
        env = self.env(tool_format="gpt-oss")
        obs, _, _, _ = await env.step(
            Action(
                text="<|start|>assistant<|channel|>analysis to=functions.parallel_search "
                '<|constrain|>json<|message|>{"query":"Agent R1 tools"}<|call|>'
            )
        )
        self.assertIn("<|start|>tool<|message|>", obs.messages[-1]["content"])
        self.assertIn("Useful excerpt", obs.messages[-1]["content"])
        self.mode = "missing_tool"
        obs, _, _, _ = await self.env().step(action("parallel_search", query="Missing server tool"))
        self.assertIn("does not expose web_search", obs.messages[-1]["content"])
        self.assertEqual(len(self.calls), 1)

    async def test_optional_dependency_error_keeps_local_tools_usable(self):
        import builtins

        real_import = builtins.__import__

        def without_mcp(name, *args, **kwargs):
            if name == "mcp" or name.startswith("mcp."):
                raise ImportError("MCP is not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_mcp):
            obs, _, _, _ = await self.env().step(action("parallel_search", query="Missing dependency"))
            self.assertIn("pip install -r requirements-search.txt", obs.messages[-1]["content"])
            obs, _, _, _ = await self.env(tools=["test_local"]).step(action("test_local"))
            self.assertIn("local result", obs.messages[-1]["content"])
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
