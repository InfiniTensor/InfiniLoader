"""Bash completion derived from the same parser used to execute commands."""
from __future__ import annotations

import argparse
import shlex


def bash_completion(parser: argparse.ArgumentParser) -> str:
    transitions, cases, values = [], [], []

    def visit(node: argparse.ArgumentParser, prefix: str) -> None:
        options = []
        positional_choices = []
        children = {}
        for action in node._actions:
            if action.help == argparse.SUPPRESS:
                continue
            options.extend(action.option_strings)
            if action.option_strings and action.choices is not None:
                for option in action.option_strings:
                    choices = " ".join(str(choice) for choice in action.choices)
                    values.append(f"    {shlex.quote(prefix + '|' + option)}) choices={shlex.quote(choices)} ;;")
            if isinstance(action, argparse._SubParsersAction):
                children.update({name: sub for name, sub in action.choices.items() if not name.startswith("_")})
            elif not action.option_strings and action.choices is not None:
                positional_choices.extend(str(choice) for choice in action.choices)
        commands = list(children) or positional_choices
        cases.append(f"    {shlex.quote(prefix)}) opts={shlex.quote(' '.join(options))}; commands={shlex.quote(' '.join(commands))} ;;")
        for name, sub in children.items():
            child = f"{prefix} {name}".strip()
            transitions.append(f"      {shlex.quote(prefix + '|' + name)}) node={shlex.quote(child)} ;;")
            visit(sub, child)

    visit(parser, "")
    script = r'''_iloader_complete() {
  COMPREPLY=()
  local cur="${COMP_WORDS[COMP_CWORD]}" prev="${COMP_WORDS[COMP_CWORD-1]}"
  local node="" word i opts="" commands="" choices="" base="" previous=""
  local run_name=0 run_cache=0 run_skip=0
  for ((i=1; i<COMP_CWORD; i++)); do
    word="${COMP_WORDS[i]}"
    if [[ "$previous" == --base-dir ]]; then base="$word"; fi
    if [[ "$word" == --base-dir=* ]]; then base="${word#*=}"; fi
    previous="$word"
    if [[ "$node" == run ]]; then
      if (( run_skip )); then run_skip=0; continue; fi
      case "$word" in
        --) return 0 ;;
        --base-dir|--socket) run_skip=1 ;;
        --cache-path) run_cache=1; run_skip=1 ;;
        --base-dir=*|--socket=*) ;;
        --cache-path=*) run_cache=1 ;;
        -*) ;;
        *) run_name=1 ;;
      esac
      continue
    fi
    case "$node|$word" in
@TRANSITIONS@
    esac
  done
  if [[ "$node" == run ]] && (( run_name || (run_cache && !run_skip) )); then return 0; fi
  case "$node" in
@CASES@
  esac
  case "$node|$prev" in
@VALUES@
  esac
  if [[ -n "$choices" ]]; then
    mapfile -t COMPREPLY < <(compgen -W "$choices" -- "$cur")
  elif [[ "$prev" =~ ^--(base-dir|view|cache-root|cache-path|log-file|pid-file|socket|rank-view-root|plan|layout|output)$ ]]; then
    compopt -o filenames 2>/dev/null || true
    mapfile -t COMPREPLY < <(compgen -f -- "$cur")
  elif [[ "$prev" == --name ]] || { [[ "$cur" != -* ]] && [[ "$node" =~ ^(status|view|manifest|stop|run)$ ]] && [[ "$prev" != --base-dir ]]; }; then
    local -a query=(list --names-only)
    if [[ -n "$base" ]]; then query+=(--base-dir "$base"); fi
    mapfile -t COMPREPLY < <(compgen -W "$(iloader "${query[@]}" 2>/dev/null)" -- "$cur")
  elif [[ "$cur" == -* ]]; then
    mapfile -t COMPREPLY < <(compgen -W "$opts" -- "$cur")
  elif [[ -n "$commands" ]]; then
    mapfile -t COMPREPLY < <(compgen -W "$commands" -- "$cur")
  else
    compopt -o filenames 2>/dev/null || true
    mapfile -t COMPREPLY < <(compgen -f -- "$cur")
  fi
}
complete -F _iloader_complete iloader
'''
    return script.replace("@TRANSITIONS@", "\n".join(transitions)).replace("@CASES@", "\n".join(cases)).replace("@VALUES@", "\n".join(values))
