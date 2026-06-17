# pi-mono-python

Python port of [pi-mono](https://github.com/badlogic/pi-mono) — the pi agent harness monorepo (coding agent CLI, agent runtime, and multi-provider LLM API).

This repository contains only the Python implementation. The original TypeScript packages live in the upstream monorepo.

## Requirements

- Python 3.11+

## Install

```bash
python3.11 -m pip install -e ".[dev]"
```

## Usage

Interactive coding agent (equivalent to the `pi` CLI):

```bash
python3.11 -m pi_mono.coding_agent
```

Print mode:

```bash
python3.11 -m pi_mono.coding_agent -p "Say exactly: ok"
```

After install, the `pi` and `pi-ai` console scripts are also available.

## Packages

| Module | Description |
|--------|-------------|
| `pi_mono.ai` | Multi-provider LLM API (OpenAI, Anthropic, Google, Mistral, Bedrock, …) |
| `pi_mono.agent` | Agent runtime with tool calling and harness |
| `pi_mono.coding_agent` | Interactive coding agent CLI |
| `pi_mono.core` | Shared core utilities |
| `pi_mono.tui` | Terminal UI components |

## Tests

```bash
cd /path/to/pi-mono-python
python3.11 -m pytest
```

## Upstream

- Original project: [badlogic/pi-mono](https://github.com/badlogic/pi-mono)
- Fork with Python port branch: [Rudreysh/pi-mono](https://github.com/Rudreysh/pi-mono) (`python-port`)

## License

Same as upstream pi-mono — see [LICENSE](LICENSE).
