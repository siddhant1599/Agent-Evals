"""Serving with no GPU, no network and no server: registry rows, the request contract, response
validation, config groups, the server lifecycle, and the backbone check."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from agent_evals.config import ConfigError, SamplingConfig, ServingConfig, load_group
from agent_evals.registry import RegistryError
from agent_evals.serving import (
    BackboneSpec,
    ModelRegistryError,
    ServingError,
    VLLMChatClient,
    VLLMServer,
    build_serve_command,
    get_client,
    load_model_registry,
    select_backbone,
)
from agent_evals.serving import check as check_module
from agent_evals.serving.launch import server_environment
from agent_evals.types import TokenCost

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "src" / "configs" / "base.yaml"
REGISTRY = REPO_ROOT / "src" / "model_registry.csv"
ENDPOINT_ENV = "FACTAGENT_VLLM_URL"
SHA = "a" * 40
HEADER = "name,hf_path,hf_revision,gated,quantization,max_model_len\n"
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")


def _response(text="ok", prompt_tokens=10, completion_tokens=2, finish_reason="stop"):
    return {
        "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


class StubClient(VLLMChatClient):
    """Replace the HTTP transport with a canned response, recording what would have been sent."""

    def __init__(self, serving, sampling, backbone, response=None):
        super().__init__(serving, sampling, backbone)
        self.response = _response() if response is None else response
        self.sent: list[tuple[str, dict]] = []

    def _post(self, url, body):
        self.sent.append((url, body))
        return self.response


class FakeProcess:
    def __init__(self, exit_code=None):
        self.returncode = exit_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


class StubbornProcess(FakeProcess):
    """Ignores terminate, so stop() has to escalate to kill."""

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        if not self.killed:
            raise subprocess.TimeoutExpired("vllm", timeout)
        return self.returncode


class FakeServer(VLLMServer):
    """Never launches anything: spawning returns a fake process, health is scripted."""

    poll_interval_s = 0.0
    stop_grace_s = 0.0

    def __init__(self, backbone, serving, log_path, process=None, healthy_after=0):
        super().__init__(backbone, serving, log_path)
        self.fake = FakeProcess() if process is None else process
        self.healthy_after = healthy_after
        self.checks = 0
        self.command = None

    def _spawn(self, command, log):
        self.command = command
        return self.fake

    def _signal_group(self, process, sig):
        if process.poll() is not None:
            return False
        (process.kill if sig == signal.SIGKILL else process.terminate)()
        return True

    def _health_ok(self, url):
        self.checks += 1
        return self.checks > self.healthy_after


@pytest.fixture
def backbone():
    return BackboneSpec(
        name="fake-backbone",
        hf_path="org/fake",
        hf_revision=SHA,
        gated=False,
        quantization="bitsandbytes",
        max_model_len=2809,
    )


@pytest.fixture
def endpoint(monkeypatch):
    monkeypatch.setenv(ENDPOINT_ENV, "http://127.0.0.1:8000/")
    return "http://127.0.0.1:8000"


def _client(backbone, response=None):
    return StubClient(ServingConfig(), SamplingConfig(), backbone, response)


def _write_registry(tmp_path, rows):
    path = tmp_path / "model_registry.csv"
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return path


class TestModelRegistry:
    def test_shipped_registry_holds_three_pinned_backbones(self):
        registry = load_model_registry(REGISTRY)
        assert tuple(registry) == (
            "qwen2.5-7b-instruct",
            "mistral-7b-instruct-v0.3",
            "llama-3.1-8b-instruct",
        )
        assert all(len(spec.hf_revision) == 40 for spec in registry.values())
        # D-039/D-040: 32,768 for every backbone, verified on the real A10G -- leaves room for
        # multi-round (T1-T4) prompts without per-dataset re-measurement.
        assert registry["qwen2.5-7b-instruct"].max_model_len == 32768
        assert registry["mistral-7b-instruct-v0.3"].max_model_len == 32768
        assert registry["llama-3.1-8b-instruct"].max_model_len == 32768
        assert registry["llama-3.1-8b-instruct"].gated is True
        assert registry["mistral-7b-instruct-v0.3"].gated is False

    def test_unknown_backbone_lists_what_is_available(self):
        with pytest.raises(ModelRegistryError, match="Available: qwen2.5-7b-instruct"):
            select_backbone(load_model_registry(REGISTRY), "gpt-4")

    def test_unmeasured_context_length_refuses_to_serve(self, backbone):
        with pytest.raises(ModelRegistryError, match="not been measured"):
            replace(backbone, max_model_len=None).require_max_model_len()

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            ("m,org/m,notasha,false,bitsandbytes,10\n", "40-character"),
            (f"m,org/m,{SHA},yes,bitsandbytes,10\n", "gated"),
            (f"m,org/m,{SHA},false,bitsandbytes,0\n", "positive integer"),
            (f",org/m,{SHA},false,bitsandbytes,10\n", "name is empty"),
        ],
    )
    def test_invalid_rows_rejected(self, tmp_path, row, message):
        with pytest.raises(ModelRegistryError, match=message):
            load_model_registry(_write_registry(tmp_path, [row]))

    def test_duplicate_name_rejected(self, tmp_path):
        row = f"m,org/m,{SHA},false,bitsandbytes,10\n"
        with pytest.raises(ModelRegistryError, match="duplicate"):
            load_model_registry(_write_registry(tmp_path, [row, row]))

    def test_missing_column_rejected(self, tmp_path):
        path = tmp_path / "model_registry.csv"
        path.write_text("name,hf_path\nm,org/m\n", encoding="utf-8")
        with pytest.raises(ModelRegistryError, match="missing columns"):
            load_model_registry(path)


class TestRequestContract:
    def test_sends_only_model_message_cap_and_seed(self, backbone):
        body = _client(backbone).request_body("hello", seed=0)
        assert set(body) == {"model", "messages", "max_tokens", "seed"}
        assert body["model"] == "org/fake"
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        assert body["max_tokens"] == 256
        assert not set(SAMPLING_KEYS) & set(body), "sampling settings must come from the model"

    def test_seed_is_omitted_when_none(self, backbone):
        assert "seed" not in _client(backbone).request_body("hello")

    def test_system_message_is_prepended_when_given(self, backbone):
        body = _client(backbone).request_body("hello", system="be terse")
        assert body["messages"] == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hello"},
        ]

    def test_system_message_is_omitted_when_none_or_empty(self, backbone):
        for system in (None, ""):
            body = _client(backbone).request_body("hello", system=system)
            assert body["messages"] == [{"role": "user", "content": "hello"}]

    def test_turns_are_sent_in_order_between_the_system_and_the_final_user_message(self, backbone):
        body = _client(backbone).request_body(
            "now", system="be terse", turns=[("q1", "a1"), ("q2", "a2")]
        )
        assert body["messages"] == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "now"},
        ]

    def test_no_turns_leaves_the_request_unchanged(self, backbone):
        client = _client(backbone)
        assert client.request_body("hello", seed=1, turns=()) == client.request_body(
            "hello", seed=1
        )

    def test_complete_sends_its_turns(self, backbone, endpoint):
        client = _client(backbone)
        client.complete("now", seed=3, turns=[("q", "a")])
        _, body = client.sent[0]
        assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]

    def test_requests_address_the_adapter_when_one_is_served(self, backbone):
        serving = ServingConfig(lora_adapters=("fake-backbone=/adapters/a",))
        body = StubClient(serving, SamplingConfig(), backbone).request_body("hello")
        assert body["model"] == "fake-backbone-lora"
        assert _client(backbone).request_body("hello")["model"] == "org/fake"

    def test_complete_posts_to_chat_endpoint_and_returns_cost(self, backbone, endpoint):
        client = _client(backbone)
        completion = client.complete("hello", seed=3)
        url, body = client.sent[0]
        assert url == endpoint + "/v1/chat/completions"
        assert body["seed"] == 3
        assert completion.text == "ok"
        assert completion.token_cost == TokenCost(prompt_tokens=10, completion_tokens=2)
        assert completion.finish_reason == "stop"
        assert completion.latency_s >= 0
        assert completion.hit_length_cap is False

    def test_length_finish_reason_flags_the_cap(self, backbone, endpoint):
        completion = _client(backbone, _response(finish_reason="length")).complete("hello")
        assert completion.hit_length_cap is True

    def test_missing_endpoint_variable_is_named(self, backbone, monkeypatch):
        monkeypatch.delenv(ENDPOINT_ENV, raising=False)
        with pytest.raises(ServingError, match=ENDPOINT_ENV):
            _client(backbone).complete("hello")


class TestResponseValidation:
    @pytest.mark.parametrize(
        "response",
        [
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            _response(completion_tokens=None),
            _response(prompt_tokens="10"),
            _response(completion_tokens=-1),
            _response(finish_reason=""),
            _response(finish_reason=None),
            _response(text=None),
            {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        ],
    )
    def test_incomplete_responses_are_rejected_never_recorded(self, backbone, endpoint, response):
        with pytest.raises(ServingError):
            _client(backbone, response).complete("hello")


class TestClientFactory:
    def test_builds_the_registered_vllm_client(self, backbone):
        assert isinstance(get_client(ServingConfig(), SamplingConfig(), backbone), VLLMChatClient)

    def test_unknown_client_name_raises(self, backbone):
        with pytest.raises(RegistryError, match="nope"):
            get_client(ServingConfig(client="nope"), SamplingConfig(), backbone)


class TestServingConfig:
    def test_base_yaml_groups_load_without_a_run_owner(self):
        serving = load_group(BASE_CONFIG, "serving", ServingConfig)
        assert serving.request_timeout_s == 120
        assert serving.startup_timeout_s == 600
        assert serving.endpoint_env == ENDPOINT_ENV
        assert serving.model_registry == "src/model_registry.csv"
        assert load_group(BASE_CONFIG, "sampling", SamplingConfig).max_tokens == 256

    def test_overrides_apply_to_a_group(self):
        overridden = load_group(
            BASE_CONFIG, "sampling", SamplingConfig, ["sampling.max_tokens=512"]
        )
        assert overridden.max_tokens == 512

    def test_unknown_key_in_a_group_raises(self):
        with pytest.raises(ConfigError, match="unknown config key"):
            load_group(BASE_CONFIG, "serving", ServingConfig, ["serving.temperature=0.2"])

    def test_absent_group_builds_from_defaults(self, tmp_path):
        path = tmp_path / "bare.yaml"
        path.write_text("name: bare\n", encoding="utf-8")
        assert load_group(path, "sampling", SamplingConfig) == SamplingConfig()

    @pytest.mark.parametrize(
        "build",
        [
            lambda: SamplingConfig(max_tokens=0),
            lambda: ServingConfig(request_timeout_s=0),
            lambda: ServingConfig(startup_timeout_s=-1),
            lambda: ServingConfig(endpoint_env=""),
        ],
    )
    def test_invalid_values_rejected(self, build):
        with pytest.raises(ConfigError):
            build()


class TestServeCommand:
    def test_command_carries_only_the_pinned_decisions(self, backbone):
        command = build_serve_command(backbone, "http://127.0.0.1:8000")
        assert command[:5] == [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            "org/fake",
        ]
        pairs = dict(zip(command[5::2], command[6::2], strict=True))
        assert pairs == {
            "--revision": SHA,
            "--quantization": "bitsandbytes",
            "--max-model-len": "2809",
            "--host": "127.0.0.1",
            "--port": "8000",
        }

    def test_an_adapter_adds_only_the_lora_flags(self, backbone):
        base = build_serve_command(backbone, "http://127.0.0.1:8000")
        command = build_serve_command(backbone, "http://127.0.0.1:8000", "~/adapters/a", 16)
        assert command[: len(base)] == base
        home = str(Path("~").expanduser())
        assert command[len(base) :] == [
            "--enable-lora",
            "--max-lora-rank",
            "16",
            "--lora-modules",
            f"fake-backbone-lora={home}/adapters/a",
        ]

    def test_gpu_memory_utilization_is_passed_only_when_set(self, backbone):
        base = build_serve_command(backbone, "http://127.0.0.1:8000")
        assert "--gpu-memory-utilization" not in base
        command = build_serve_command(
            backbone, "http://127.0.0.1:8000", gpu_memory_utilization=0.85
        )
        assert command == base + ["--gpu-memory-utilization", "0.85"]

    def test_token_never_reaches_the_command_line(self, backbone, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "FAKE_EXAMPLE_token_value")
        assert "FAKE_EXAMPLE_token_value" not in " ".join(
            build_serve_command(backbone, "http://127.0.0.1:8000")
        )

    @pytest.mark.parametrize(
        "url", ["http://127.0.0.1", "https://127.0.0.1:8000", "localhost:8000"]
    )
    def test_endpoint_needs_http_host_and_port(self, backbone, url):
        with pytest.raises(ServingError):
            build_serve_command(backbone, url)

    def test_unmeasured_backbone_is_not_served(self, backbone):
        with pytest.raises(ModelRegistryError):
            build_serve_command(replace(backbone, max_model_len=None), "http://127.0.0.1:8000")


class TestServerLifecycle:
    def test_waits_for_health_then_stops_on_exit(self, tmp_path, endpoint, backbone):
        server = FakeServer(backbone, ServingConfig(), tmp_path / "vllm.log", healthy_after=2)
        with server:
            assert server.checks == 3
            assert server.fake.terminated is False
        assert server.fake.terminated is True

    def test_stops_even_when_the_block_raises(self, tmp_path, endpoint, backbone):
        server = FakeServer(backbone, ServingConfig(), tmp_path / "vllm.log")
        with pytest.raises(RuntimeError), server:
            raise RuntimeError("boom")
        assert server.fake.terminated is True

    def test_process_exit_before_ready_reports_the_log(self, tmp_path, endpoint, backbone):
        log_path = tmp_path / "vllm.log"
        server = FakeServer(
            backbone, ServingConfig(), log_path, process=FakeProcess(1), healthy_after=10**9
        )
        with pytest.raises(ServingError, match="exited with code 1") as excinfo, server:
            pass
        assert str(log_path) in str(excinfo.value)

    def test_startup_timeout_raises_and_stops_the_process(self, tmp_path, endpoint, backbone):
        serving = ServingConfig(startup_timeout_s=0.01)
        server = FakeServer(backbone, serving, tmp_path / "vllm.log", healthy_after=10**9)
        with pytest.raises(ServingError, match="not ready within"), server:
            pass
        assert server.fake.terminated is True

    def test_stop_kills_a_process_that_ignores_terminate(self, tmp_path, endpoint, backbone):
        server = FakeServer(
            backbone, ServingConfig(), tmp_path / "vllm.log", process=StubbornProcess()
        )
        with server:
            pass
        assert server.fake.killed is True


def _fake_repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model_registry.csv").write_text(REGISTRY.read_text(), encoding="utf-8")
    (tmp_path / ".env").write_text("# no secrets in tests\n", encoding="utf-8")
    return tmp_path


class TestBackboneCheck:
    @pytest.fixture
    def fakes(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENDPOINT_ENV, "http://127.0.0.1:8000")
        monkeypatch.setattr(check_module.tempfile, "gettempdir", lambda: str(tmp_path))
        servers, clients = [], []

        def fake_server(backbone, serving, log_path):
            servers.append(FakeServer(backbone, serving, log_path))
            return servers[-1]

        def fake_get_client(serving, sampling, backbone):
            clients.append(StubClient(serving, sampling, backbone))
            return clients[-1]

        monkeypatch.setattr(check_module, "VLLMServer", fake_server)
        monkeypatch.setattr(check_module, "get_client", fake_get_client)
        return servers, clients

    def test_serves_the_probe_and_stops_the_server(self, tmp_path, fakes):
        servers, clients = fakes
        completion = check_module.check_backbone(
            BASE_CONFIG, "qwen2.5-7b-instruct", _fake_repo(tmp_path)
        )
        assert completion.text == "ok"
        assert servers[0].command[4] == "Qwen/Qwen2.5-7B-Instruct"
        assert servers[0].fake.terminated is True
        assert clients[0].sent[0][1]["messages"][0]["content"] == ServingConfig().check_prompt

    def test_unmeasured_backbone_fails_before_any_process_starts(self, tmp_path, fakes):
        servers, _ = fakes
        root = _fake_repo(tmp_path)
        with (root / "src" / "model_registry.csv").open("a", encoding="utf-8") as registry:
            registry.write(f"unmeasured-backbone,org/unmeasured,{SHA},false,bitsandbytes,\n")
        with pytest.raises(ModelRegistryError, match="not been measured"):
            check_module.check_backbone(BASE_CONFIG, "unmeasured-backbone", root)
        assert servers[0].command is None

    def test_main_returns_1_and_names_the_problem(self, tmp_path, fakes, capsys):
        code = check_module.main(
            ["--backbone", "gpt-4"], repo_root=_fake_repo(tmp_path), default_config=BASE_CONFIG
        )
        assert code == 1
        assert "unknown backbone 'gpt-4'" in capsys.readouterr().err


class TestServerEnvironment:
    """The server must find tools in the interpreter's bin directory without venv activation."""

    BIN = str(Path(sys.executable).parent)

    def test_interpreter_bin_dir_comes_first_on_path(self):
        env = server_environment({"PATH": os.pathsep.join(["/usr/bin", "/bin"])})
        assert env["PATH"].split(os.pathsep) == [self.BIN, "/usr/bin", "/bin"]

    def test_bin_dir_is_moved_to_the_front_not_duplicated(self):
        env = server_environment({"PATH": os.pathsep.join(["/usr/bin", self.BIN])})
        assert env["PATH"].split(os.pathsep) == [self.BIN, "/usr/bin"]

    def test_absent_path_still_reaches_the_interpreter_tools(self):
        assert server_environment({})["PATH"] == self.BIN

    def test_other_variables_pass_through_and_input_is_untouched(self):
        base = {"PATH": "/bin", "HF_TOKEN": "FAKE_EXAMPLE_token"}
        env = server_environment(base)
        assert env["HF_TOKEN"] == "FAKE_EXAMPLE_token"
        assert base == {"PATH": "/bin", "HF_TOKEN": "FAKE_EXAMPLE_token"}


_PARENT_WITH_CHILD = """
import pathlib, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(120)
"""


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
        return handle.read().split()[2] != "Z"


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups are POSIX only")
def test_stop_also_kills_processes_the_server_started(tmp_path, backbone):
    """A child that survives its parent's SIGTERM, as vLLM's engine did, is still stopped."""
    pid_file = tmp_path / "child.pid"
    server = VLLMServer(backbone, ServingConfig(), tmp_path / "server.log")
    server._log = (tmp_path / "server.log").open("wb")
    server._process = server._spawn(
        [sys.executable, "-c", _PARENT_WITH_CHILD, str(pid_file)], server._log
    )
    deadline = time.monotonic() + 10
    while not pid_file.exists() or not pid_file.read_text():
        assert time.monotonic() < deadline, "child process never started"
        time.sleep(0.05)
    child = int(pid_file.read_text())
    assert _alive(child)
    server.stop()
    deadline = time.monotonic() + 10
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(child)
