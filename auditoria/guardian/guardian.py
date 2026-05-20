#!/usr/bin/env python3
"""Cão de Guarda manual do IFR2 Miner & Screener.

Roda checks de saúde do projeto no fim de cada fase de entrega.
Sem cron job.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

PROJECT_DEFAULT = Path("/mnt/c/Dev/IFR2")
EXCLUDE_PARTS = {
    "venv",
    ".venv",
    ".git",
    "__pycache__",
    "node_modules",
    ".next",
}
CRITICAL_CHECKS = {"py_compile", "smoke_test", "integration_test", "pip_check", "pytest"}


@dataclass
class CheckResult:
    name: str
    status: str
    code: int | None
    command: str
    output: str
    note: str = ""


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "phase"


def human_cmd(args: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in args)


def choose_python(project: Path) -> str:
    venv_python = project / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def collect_code_files(project: Path) -> list[Path]:
    files: list[Path] = []

    for relative in ("ifr2_app.py", "smoke_test.py", "integration_test.py"):
        path = project / relative
        if path.exists():
            files.append(path)

    for folder in ("src", "config", "scripts"):
        base = project / folder
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if any(part in EXCLUDE_PARTS for part in path.parts):
                continue
            files.append(path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def short_text(text: str, limit: int = 1800) -> str:
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "\n...<truncado>..."


def run_command(python_bin: str, args: list[str], cwd: Path, name: str, note: str = "") -> CheckResult:
    command = [python_bin, *args]
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    combined = "\n".join(part for part in [proc.stdout.strip(), proc.stderr.strip()] if part).strip()
    status = "OK" if proc.returncode == 0 else "FAIL"
    return CheckResult(
        name=name,
        status=status,
        code=proc.returncode,
        command=human_cmd(command),
        output=short_text(combined),
        note=note,
    )


def skip_check(name: str, command: str, note: str) -> CheckResult:
    return CheckResult(name=name, status="SKIP", code=None, command=command, output="", note=note)


def is_broad_exception(node: ast.AST | None) -> bool:
    if node is None:
        return True
    if isinstance(node, ast.Name):
        return node.id in {"Exception", "BaseException"}
    if isinstance(node, ast.Attribute):
        return node.attr in {"Exception", "BaseException"}
    if isinstance(node, ast.Tuple):
        return any(is_broad_exception(elt) for elt in node.elts)
    return False


def analyze_file(path: Path, runtime: bool) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    info: dict[str, Any] = {
        "path": str(path),
        "relative": str(path),
        "lines": len(lines),
        "too_long": len(lines) > 300,
        "todo_count": len(re.findall(r"\b(?:TODO|FIXME|HACK)\b", text)),
        "cache_hits": len(re.findall(r"cache_data|cache_resource|lru_cache", text)),
        "broad_excepts": 0,
        "bare_excepts": 0,
        "eval_calls": 0,
        "exec_calls": 0,
        "print_calls": 0,
        "long_functions": [],
        "syntax_error": None,
    }

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        info["syntax_error"] = f"{exc.msg} (linha {exc.lineno})"
        return info

    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                if handler.type is None:
                    info["bare_excepts"] += 1
                elif is_broad_exception(handler.type):
                    info["broad_excepts"] += 1
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "eval":
                info["eval_calls"] += 1
            elif isinstance(node.func, ast.Name) and node.func.id == "exec":
                info["exec_calls"] += 1
            elif runtime and isinstance(node.func, ast.Name) and node.func.id == "print":
                info["print_calls"] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end_line = getattr(node, "end_lineno", None) or node.lineno
            length = end_line - node.lineno + 1
            if length > 50:
                info["long_functions"].append({
                    "name": node.name,
                    "lines": length,
                    "start": node.lineno,
                })

    return info


def collect_metrics(code_files: list[Path]) -> dict[str, Any]:
    per_file: list[dict[str, Any]] = []

    for path in code_files:
        runtime = path.name not in {"smoke_test.py", "integration_test.py"}
        file_info = analyze_file(path, runtime=runtime)
        file_info["relative"] = str(path.relative_to(PROJECT_DEFAULT)) if path.is_relative_to(PROJECT_DEFAULT) else str(path)
        per_file.append(file_info)

    files_over_300 = [
        {"file": item["relative"], "lines": item["lines"]}
        for item in per_file
        if item["too_long"]
    ]
    syntax_errors = [
        {"file": item["relative"], "error": item["syntax_error"]}
        for item in per_file
        if item["syntax_error"]
    ]
    long_functions: list[dict[str, Any]] = []
    for item in per_file:
        for func in item["long_functions"]:
            long_functions.append(
                {
                    "file": item["relative"],
                    "function": func["name"],
                    "lines": func["lines"],
                    "start": func["start"],
                }
            )

    line_count = sum(item["lines"] for item in per_file)
    return {
        "file_count": len(per_file),
        "line_count": line_count,
        "files_over_300": sorted(files_over_300, key=lambda x: x["lines"], reverse=True),
        "long_functions": sorted(long_functions, key=lambda x: x["lines"], reverse=True),
        "syntax_errors": syntax_errors,
        "broad_excepts": sum(item["broad_excepts"] for item in per_file),
        "bare_excepts": sum(item["bare_excepts"] for item in per_file),
        "eval_calls": sum(item["eval_calls"] for item in per_file),
        "exec_calls": sum(item["exec_calls"] for item in per_file),
        "print_calls_runtime": sum(item["print_calls"] for item in per_file),
        "todo_count": sum(item["todo_count"] for item in per_file),
        "cache_hits": sum(item["cache_hits"] for item in per_file),
        "longest_files": sorted(
            [{"file": item["relative"], "lines": item["lines"]} for item in per_file],
            key=lambda x: x["lines"],
            reverse=True,
        )[:10],
        "per_file": per_file,
    }


def gather_git(project: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        proc = subprocess.run(["git", *args], cwd=project, capture_output=True, text=True)
        if proc.returncode != 0:
            return ""
        return proc.stdout.strip()

    branch = run_git(["branch", "--show-current"]) or "desconhecido"
    last_commit = run_git(["log", "-1", "--format=%h - %s (%ar)"]) or "sem commits"
    status = subprocess.run(["git", "status", "--porcelain"], cwd=project, capture_output=True, text=True)
    uncommitted = len([line for line in status.stdout.splitlines() if line.strip()]) if status.returncode == 0 else -1
    return {
        "branch": branch,
        "last_commit": last_commit,
        "uncommitted": uncommitted,
    }


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def detect_regressions(previous: dict[str, Any] | None, current_metrics: dict[str, Any]) -> list[str]:
    if not previous:
        return []

    regressions: list[str] = []
    prev_metrics = previous.get("metrics", {})
    for key in ("bare_excepts", "broad_excepts", "eval_calls", "exec_calls"):
        prev_value = int(prev_metrics.get(key, 0) or 0)
        current_value = int(current_metrics.get(key, 0) or 0)
        if current_value > prev_value:
            regressions.append(f"{key}: {prev_value} → {current_value}")

    prev_syntax = len(previous.get("metrics", {}).get("syntax_errors", []) or [])
    current_syntax = len(current_metrics.get("syntax_errors", []) or [])
    if current_syntax > prev_syntax:
        regressions.append(f"syntax_errors: {prev_syntax} → {current_syntax}")

    return regressions


def compute_score(checks: list[CheckResult], metrics: dict[str, Any], formal_tests_present: bool) -> int:
    score = 100

    for check in checks:
        if check.status != "FAIL":
            continue
        if check.name in {"py_compile", "smoke_test", "integration_test", "pip_check", "pytest"}:
            score -= 20
        elif check.name in {"flake8", "black"}:
            score -= 5

    score -= min(metrics["broad_excepts"] * 2, 20)
    score -= min(metrics["bare_excepts"] * 5, 25)
    score -= min(metrics["eval_calls"] * 10, 20)
    score -= min(metrics["exec_calls"] * 10, 20)

    if metrics["cache_hits"] == 0:
        score -= 5
    if metrics["print_calls_runtime"] > 0:
        score -= min(metrics["print_calls_runtime"], 10)
    if metrics["todo_count"] > 0:
        score -= min(metrics["todo_count"], 10)
    if metrics["files_over_300"]:
        score -= min(len(metrics["files_over_300"]), 10)
    if metrics["long_functions"]:
        score -= min(len(metrics["long_functions"]), 10)
    if not formal_tests_present:
        score -= 5

    return max(score, 0)


def determine_status(checks: list[CheckResult], regressions: list[str], metrics: dict[str, Any], formal_tests_present: bool) -> str:
    critical_failed = any(check.status == "FAIL" and check.name in CRITICAL_CHECKS for check in checks)
    if critical_failed or regressions:
        return "CRITICO"

    important_failed = any(check.status == "FAIL" for check in checks if check.name in {"flake8", "black"})
    if important_failed:
        return "ATENCAO"

    if metrics["files_over_300"] or metrics["long_functions"] or metrics["cache_hits"] == 0 or not formal_tests_present:
        return "ATENCAO"

    return "OK"


def render_badge(status: str) -> str:
    css = {
        "OK": "ok",
        "ATENCAO": "warn",
        "CRITICO": "crit",
        "FAIL": "crit",
        "SKIP": "skip",
    }.get(status, "neutral")
    return f"<span class='badge {css}'>{escape(status)}</span>"


def render_html(report: dict[str, Any]) -> str:
    checks: list[dict[str, Any]] = report["checks"]
    metrics: dict[str, Any] = report["metrics"]
    regressions: list[str] = report["regressions"]
    git_info: dict[str, Any] = report["git"]

    checks_rows = []
    for check in checks:
        output = check.get("output") or check.get("note") or ""
        if output:
            output_html = f"<pre>{escape(output)}</pre>"
        else:
            output_html = "<span class='muted'>Sem saída</span>"
        checks_rows.append(
            f"""
            <tr>
              <td>{escape(check['name'])}</td>
              <td>{render_badge(check['status'])}</td>
              <td>{escape(str(check.get('code', '—')))}</td>
              <td>{escape(check.get('command', ''))}</td>
              <td>{output_html}</td>
            </tr>
            """
        )

    def metric_row(label: str, value: Any) -> str:
        return f"<tr><td>{escape(label)}</td><td>{escape(str(value))}</td></tr>"

    files_over_300 = metrics["files_over_300"]
    long_functions = metrics["long_functions"]
    syntax_errors = metrics["syntax_errors"]
    longest_files = metrics["longest_files"]

    regressions_html = "<p class='muted'>Nenhuma regressão detectada.</p>"
    if regressions:
        regressions_html = "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in regressions) + "</ul>"

    files_html = "<p class='muted'>Nenhum arquivo acima de 300 linhas.</p>"
    if files_over_300:
        files_html = "<ul>" + "".join(
            f"<li>{escape(item['file'])} — {item['lines']} linhas</li>" for item in files_over_300
        ) + "</ul>"

    funcs_html = "<p class='muted'>Nenhuma função acima de 50 linhas.</p>"
    if long_functions:
        funcs_html = "<ul>" + "".join(
            f"<li>{escape(item['file'])}:{escape(item['function'])} — {item['lines']} linhas (linha {item['start']})</li>"
            for item in long_functions
        ) + "</ul>"

    syntax_html = "<p class='muted'>Nenhum erro de sintaxe detectado.</p>"
    if syntax_errors:
        syntax_html = "<ul>" + "".join(
            f"<li>{escape(item['file'])} — {escape(item['error'])}</li>" for item in syntax_errors
        ) + "</ul>"

    longest_html = "<ol>" + "".join(
        f"<li>{escape(item['file'])} — {item['lines']} linhas</li>" for item in longest_files
    ) + "</ol>"

    html = f"""<!doctype html>
<html lang='pt-BR'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Cão de Guarda — IFR2</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#0f172a; color:#e2e8f0; margin:0; }}
    .wrap {{ max-width:1100px; margin:0 auto; padding:24px; }}
    .header {{ background:linear-gradient(135deg,#1e293b,#334155); border:1px solid #475569; border-radius:16px; padding:20px; margin-bottom:16px; }}
    .title {{ font-size:24px; font-weight:800; margin:0 0 6px; }}
    .sub {{ color:#94a3b8; font-size:13px; margin:0; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:16px; }}
    .card {{ background:#1e293b; border:1px solid #475569; border-radius:14px; padding:14px; }}
    .label {{ color:#94a3b8; font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
    .value {{ font-size:28px; font-weight:800; margin-top:6px; }}
    .ok {{ color:#22c55e; }}
    .warn {{ color:#f59e0b; }}
    .crit {{ color:#ef4444; }}
    .skip {{ color:#60a5fa; }}
    .neutral {{ color:#e2e8f0; }}
    .badge {{ display:inline-block; padding:3px 10px; border-radius:999px; font-weight:700; font-size:12px; }}
    .badge.ok {{ background:#22c55e22; color:#22c55e; border:1px solid #22c55e55; }}
    .badge.warn {{ background:#f59e0b22; color:#f59e0b; border:1px solid #f59e0b55; }}
    .badge.crit {{ background:#ef444422; color:#ef4444; border:1px solid #ef444455; }}
    .badge.skip {{ background:#60a5fa22; color:#60a5fa; border:1px solid #60a5fa55; }}
    .section {{ background:#111827; border:1px solid #334155; border-radius:16px; padding:16px; margin-bottom:16px; }}
    table {{ width:100%; border-collapse:collapse; }}
    th, td {{ text-align:left; vertical-align:top; border-bottom:1px solid #334155; padding:10px; font-size:13px; }}
    th {{ color:#94a3b8; font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
    pre {{ white-space:pre-wrap; word-break:break-word; background:#0f172a; color:#cbd5e1; padding:10px; border-radius:8px; border:1px solid #334155; margin:0; }}
    .muted {{ color:#94a3b8; }}
    .two-col {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }}
    ul, ol {{ margin:8px 0 0 20px; }}
    .footer {{ color:#64748b; font-size:12px; padding:8px 2px 0; }}
    @media (max-width: 900px) {{ .grid, .two-col {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class='wrap'>
    <div class='header'>
      <h1 class='title'>🐕 Cão de Guarda — IFR2 Miner & Screener</h1>
      <p class='sub'>Fase: {escape(report['phase'])} | Verificação: {escape(report['timestamp'])} | Branch: {escape(git_info['branch'])}</p>
      <p class='sub'>Último commit: {escape(git_info['last_commit'])}</p>
    </div>

    <div class='grid'>
      <div class='card'><div class='label'>Status</div><div class='value'>{render_badge(report['status'])}</div></div>
      <div class='card'><div class='label'>Score</div><div class='value neutral'>{report['score']}</div></div>
      <div class='card'><div class='label'>Arquivos</div><div class='value neutral'>{metrics['file_count']}</div></div>
      <div class='card'><div class='label'>Linhas</div><div class='value neutral'>{metrics['line_count']}</div></div>
    </div>

    <div class='grid'>
      <div class='card'><div class='label'>Broad excepts</div><div class='value {('warn' if metrics['broad_excepts'] else 'ok')}'>{metrics['broad_excepts']}</div></div>
      <div class='card'><div class='label'>Bare excepts</div><div class='value {('warn' if metrics['bare_excepts'] else 'ok')}'>{metrics['bare_excepts']}</div></div>
      <div class='card'><div class='label'>eval / exec</div><div class='value {('warn' if (metrics['eval_calls'] or metrics['exec_calls']) else 'ok')}'>{metrics['eval_calls']} / {metrics['exec_calls']}</div></div>
      <div class='card'><div class='label'>Cache markers</div><div class='value {('ok' if metrics['cache_hits'] else 'warn')}'>{metrics['cache_hits']}</div></div>
    </div>

    <div class='section'>
      <h2>Checks</h2>
      <table>
        <thead><tr><th>Check</th><th>Status</th><th>Código</th><th>Comando</th><th>Saída / nota</th></tr></thead>
        <tbody>
          {''.join(checks_rows)}
        </tbody>
      </table>
    </div>

    <div class='two-col'>
      <div class='section'>
        <h2>Métricas rápidas</h2>
        <table>
          <tbody>
            {metric_row('Funções longas', len(long_functions))}
            {metric_row('Arquivos longos', len(files_over_300))}
            {metric_row('TODO/FIXME/HACK', metrics['todo_count'])}
            {metric_row('Prints no runtime', metrics['print_calls_runtime'])}
            {metric_row('Erros de sintaxe', len(syntax_errors))}
            {metric_row('Testes formais', 'presentes' if report['formal_tests_present'] else 'ausentes')}
            {metric_row('Arquivos com teste / smoke', report['formal_test_count'])}
            {metric_row('Git não commitado', git_info['uncommitted'])}
          </tbody>
        </table>
      </div>

      <div class='section'>
        <h2>Regressões</h2>
        {regressions_html}
      </div>
    </div>

    <div class='two-col'>
      <div class='section'>
        <h2>Arquivos grandes</h2>
        {files_html}
      </div>
      <div class='section'>
        <h2>Funções grandes</h2>
        {funcs_html}
      </div>
    </div>

    <div class='two-col'>
      <div class='section'>
        <h2>Erros de sintaxe</h2>
        {syntax_html}
      </div>
      <div class='section'>
        <h2>Arquivos mais longos</h2>
        {longest_html}
      </div>
    </div>

    <div class='section'>
      <h2>Próximos passos</h2>
      <ul>
        <li>Corrigir qualquer check crítico antes de fechar a fase.</li>
        <li>Eliminar regressões novas antes de avançar para a próxima etapa.</li>
        <li>Tratar dívida técnica apontada como importante o quanto antes.</li>
      </ul>
    </div>

    <p class='footer'>Relatório salvo em: {escape(report['report_file'])} | Último estado: {escape(report['state_file'])}</p>
  </div>
</body>
</html>
"""
    return html


def formal_tests_present(project: Path) -> tuple[bool, int]:
    tests_dir = project / "tests"
    if not tests_dir.exists():
        return False, 0
    files = [p for p in tests_dir.rglob("*.py") if p.is_file()]
    return bool(files), len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cão de Guarda manual do IFR2")
    parser.add_argument("--project", default=str(PROJECT_DEFAULT), help="Caminho do projeto IFR2")
    parser.add_argument("--phase", default="fase-manual", help="Nome da fase de entrega")
    args = parser.parse_args()

    project = Path(args.project).resolve()
    report_dir = project / "auditoria" / "guardian"
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    phase_slug = slugify(args.phase)
    python_bin = choose_python(project)

    code_files = collect_code_files(project)
    checks: list[CheckResult] = []

    if code_files:
        checks.append(
            run_command(
                python_bin,
                ["-m", "py_compile", *[str(path) for path in code_files]],
                project,
                "py_compile",
                "Compilação dos arquivos Python do projeto",
            )
        )
    else:
        checks.append(skip_check("py_compile", f"{python_bin} -m py_compile ...", "Nenhum arquivo Python encontrado para compilar"))

    smoke_file = project / "smoke_test.py"
    if smoke_file.exists():
        checks.append(
            run_command(
                python_bin,
                [str(smoke_file)],
                project,
                "smoke_test",
                "Execução do smoke_test.py",
            )
        )
    else:
        checks.append(skip_check("smoke_test", f"{python_bin} smoke_test.py", "Arquivo não encontrado"))

    integration_file = project / "integration_test.py"
    if integration_file.exists():
        checks.append(
            run_command(
                python_bin,
                [str(integration_file)],
                project,
                "integration_test",
                "Execução do integration_test.py",
            )
        )
    else:
        checks.append(skip_check("integration_test", f"{python_bin} integration_test.py", "Arquivo não encontrado"))

    tests_present, test_count = formal_tests_present(project)
    if tests_present:
        checks.append(
            run_command(
                python_bin,
                ["-m", "pytest", "-q"],
                project,
                "pytest",
                f"Execução de {test_count} arquivo(s) em tests/",
            )
        )
    else:
        checks.append(skip_check("pytest", f"{python_bin} -m pytest -q", "Nenhuma suíte formal em tests/"))

    checks.append(
        run_command(
            python_bin,
            ["-m", "pip", "check"],
            project,
            "pip_check",
            "Verificação de dependências instaladas",
        )
    )

    if importlib.util.find_spec("flake8") is not None:
        checks.append(
            run_command(
                python_bin,
                ["-m", "flake8", *[str(path) for path in code_files]],
                project,
                "flake8",
                "Checagem de estilo com flake8",
            )
        )
    else:
        checks.append(skip_check("flake8", f"{python_bin} -m flake8 ...", "flake8 não instalado"))

    if importlib.util.find_spec("black") is not None:
        checks.append(
            run_command(
                python_bin,
                ["-m", "black", "--check", *[str(path) for path in code_files]],
                project,
                "black",
                "Checagem de formatação com black",
            )
        )
    else:
        checks.append(skip_check("black", f"{python_bin} -m black --check ...", "black não instalado"))

    metrics = collect_metrics(code_files)
    git_info = gather_git(project)
    previous_state_path = report_dir / "last-state.json"
    previous_state = load_json(previous_state_path)
    regressions = detect_regressions(previous_state, metrics)
    score = compute_score(checks, metrics, tests_present)
    status = determine_status(checks, regressions, metrics, tests_present)

    report = {
        "project": str(project),
        "phase": args.phase,
        "timestamp": timestamp,
        "score": score,
        "status": status,
        "checks": [asdict(check) for check in checks],
        "metrics": metrics,
        "regressions": regressions,
        "git": git_info,
        "formal_tests_present": tests_present,
        "formal_test_count": test_count,
        "report_file": str(report_dir / f"status-{run_id}-{phase_slug}.html"),
        "latest_file": str(report_dir / "status-latest.html"),
        "state_file": str(previous_state_path),
    }

    html = render_html(report)
    (report_dir / f"status-{run_id}-{phase_slug}.html").write_text(html, encoding="utf-8")
    (report_dir / "status-latest.html").write_text(html, encoding="utf-8")
    previous_state_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    log_file = report_dir / "log.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] phase={args.phase} status={status} score={score} regressions={len(regressions)}\n")

    print(f"PHASE={args.phase}")
    print(f"STATUS={status}")
    print(f"SCORE={score}")
    print(f"CRITICAL={1 if status == 'CRITICO' else 0}")
    print(f"REPORT_FILE={report['report_file']}")
    print(f"LATEST_FILE={report['latest_file']}")
    print(f"LOG_FILE={log_file}")
    print(f"STATE_FILE={report['state_file']}")

    return 1 if status == "CRITICO" else 0


if __name__ == "__main__":
    raise SystemExit(main())
