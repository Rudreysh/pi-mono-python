import os

_proc_env_cache: dict[str, str] | None = None
_cached_vertex_adc_credentials_exists: bool | None = None


def get_proc_env(key: str) -> str | None:
    global _proc_env_cache
    val = os.environ.get(key)
    if val is not None:
        return val

    if _proc_env_cache is None:
        _proc_env_cache = {}
        try:
            if os.path.exists("/proc/self/environ"):
                with open("/proc/self/environ", "rb") as f:
                    data = f.read()
                for entry_bytes in data.split(b"\0"):
                    if b"=" in entry_bytes:
                        k, v = entry_bytes.split(b"=", 1)
                        _proc_env_cache[k.decode("utf-8", errors="ignore")] = v.decode(
                            "utf-8", errors="ignore"
                        )
        except Exception:
            pass
    return _proc_env_cache.get(key)


def has_vertex_adc_credentials() -> bool:
    global _cached_vertex_adc_credentials_exists
    if _cached_vertex_adc_credentials_exists is None:
        gac_path = get_proc_env("GOOGLE_APPLICATION_CREDENTIALS")
        if gac_path:
            _cached_vertex_adc_credentials_exists = os.path.exists(gac_path)
        else:
            home = os.path.expanduser("~")
            adc_path = os.path.join(
                home, ".config", "gcloud", "application_default_credentials.json"
            )
            _cached_vertex_adc_credentials_exists = os.path.exists(adc_path)
    return _cached_vertex_adc_credentials_exists


def get_api_key_env_vars(provider: str) -> list[str] | None:
    if provider == "github-copilot":
        return ["COPILOT_GITHUB_TOKEN"]

    if provider == "anthropic":
        return ["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]

    env_map = {
        "ant-ling": "ANT_LING_API_KEY",
        "openai": "OPENAI_API_KEY",
        "azure-openai-responses": "AZURE_OPENAI_API_KEY",
        "nvidia": "NVIDIA_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "google": "GEMINI_API_KEY",
        "google-vertex": "GOOGLE_CLOUD_API_KEY",
        "groq": "GROQ_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "xai": "XAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "vercel-ai-gateway": "AI_GATEWAY_API_KEY",
        "zai": "ZAI_API_KEY",
        "zai-coding-cn": "ZAI_CODING_CN_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "minimax": "MINIMAX_API_KEY",
        "minimax-cn": "MINIMAX_CN_API_KEY",
        "moonshotai": "MOONSHOT_API_KEY",
        "moonshotai-cn": "MOONSHOT_API_KEY",
        "huggingface": "HF_TOKEN",
        "fireworks": "FIREWORKS_API_KEY",
        "together": "TOGETHER_API_KEY",
        "opencode": "OPENCODE_API_KEY",
        "opencode-go": "OPENCODE_API_KEY",
        "kimi-coding": "KIMI_API_KEY",
        "cloudflare-workers-ai": "CLOUDFLARE_API_KEY",
        "cloudflare-ai-gateway": "CLOUDFLARE_API_KEY",
        "xiaomi": "XIAOMI_API_KEY",
        "xiaomi-token-plan-cn": "XIAOMI_TOKEN_PLAN_CN_API_KEY",
        "xiaomi-token-plan-ams": "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
        "xiaomi-token-plan-sgp": "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
        "cursor": "CURSOR_API_KEY",
    }

    env_var = env_map.get(provider)
    return [env_var] if env_var else None


def find_env_keys(provider: str) -> list[str] | None:
    env_vars = get_api_key_env_vars(provider)
    if not env_vars:
        return None

    found = [var for var in env_vars if get_proc_env(var) is not None]
    return found if len(found) > 0 else None


def get_env_api_key(provider: str) -> str | None:
    env_keys = find_env_keys(provider)
    if env_keys and len(env_keys) > 0:
        return get_proc_env(env_keys[0])

    if provider == "google-vertex":
        has_credentials = has_vertex_adc_credentials()
        has_project = any(
            get_proc_env(var) is not None for var in ["GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"]
        )
        has_location = get_proc_env("GOOGLE_CLOUD_LOCATION") is not None

        if has_credentials and has_project and has_location:
            return "<authenticated>"

    if provider == "amazon-bedrock":
        aws_vars = [
            "AWS_PROFILE",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_BEARER_TOKEN_BEDROCK",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
        ]
        # For access key/secret access key, both must be present
        has_access_keys = (
            get_proc_env("AWS_ACCESS_KEY_ID") is not None
            and get_proc_env("AWS_SECRET_ACCESS_KEY") is not None
        )
        has_other_aws_vars = any(
            get_proc_env(var) is not None
            for var in aws_vars
            if var not in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        )
        if has_access_keys or has_other_aws_vars:
            return "<authenticated>"

    return None
