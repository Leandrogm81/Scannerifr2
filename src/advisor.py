"""Conselheiro IA (OpenRouter + DeepSeek) para análise textual de oportunidades."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_REASONING_EFFORT = "high"


@dataclass
class AdvisorConfig:
    api_key: str | None
    model: str
    reasoning_effort: str


def _read_streamlit_secret(key: str) -> str | None:
    try:
        import streamlit as st

        if key in st.secrets:
            value = st.secrets[key]
            return str(value).strip() if value else None
    except Exception:
        return None
    return None


def load_advisor_config() -> AdvisorConfig:
    """Carrega configuração da IA de secrets/env sem expor dados sensíveis."""

    api_key = (
        _read_streamlit_secret("OPENROUTER_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or None
    )

    model = (
        _read_streamlit_secret("OPENROUTER_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or DEFAULT_MODEL
    )

    # Compatibilidade: usuário pode ter salvo OPENROUTER_REASONING=xhigh.
    reasoning_effort = (
        _read_streamlit_secret("OPENROUTER_REASONING")
        or os.getenv("OPENROUTER_REASONING")
        or DEFAULT_REASONING_EFFORT
    )

    # Normalizar valores aceitos pelo endpoint reasoning.effort.
    mapping = {
        "xhigh": "high",
        "high": "high",
        "medium": "medium",
        "med": "medium",
        "low": "low",
    }
    reasoning_effort = mapping.get(str(reasoning_effort).lower(), "high")

    return AdvisorConfig(
        api_key=api_key, model=model, reasoning_effort=reasoning_effort
    )


def _build_prompt(payload: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "Você é um conselheiro quantitativo para usuário leigo. "
        "Seja direto e objetivo, em português-BR. "
        "Dê: (1) leitura técnica, (2) leitura fundamentalista, "
        "(3) risco principal, (4) veredito final em 1 linha. "
        "Nunca prometa lucro e nunca diga que é recomendação garantida."
    )

    user = "Analise o ativo abaixo e escreva em linguagem simples:\n\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def run_ai_advice(payload: dict[str, Any]) -> tuple[bool, str]:
    """Executa chamada ao OpenRouter e retorna (sucesso, resposta)."""

    cfg = load_advisor_config()
    if not cfg.api_key:
        return (
            False,
            "OPENROUTER_API_KEY não encontrada em st.secrets nem em variável de ambiente.",
        )

    body = {
        "model": cfg.model,
        "messages": _build_prompt(payload),
        "reasoning": {"effort": cfg.reasoning_effort},
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    req = Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(req, timeout=45) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)

        choices = data.get("choices") or []
        if not choices:
            return False, "OpenRouter retornou resposta vazia (sem choices)."

        message = choices[0].get("message", {}).get("content", "")
        if not message:
            return False, "OpenRouter retornou resposta sem conteúdo de mensagem."

        return True, message.strip()

    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        return False, f"Erro HTTP OpenRouter: {exc.code} - {detail}"
    except URLError as exc:
        return False, f"Erro de conexão OpenRouter: {exc.reason}"
    except Exception as exc:
        return False, f"Erro inesperado ao consultar IA: {exc}"
