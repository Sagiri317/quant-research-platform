# Agent Capability Decision Matrix

This matrix records which agent capabilities QRP should use now, later, or not
by default. It is a governance decision aid, not a runtime config.

| Capability | QRP decision | Rework reduced | Cost/risk | Layer |
|---|---|---|---|---|
| `AGENTS.md` | use now, short and hard | repeated boundary mistakes | token cap if too long | repo root |
| nested AGENTS | later only for real local owner rules | local rule misses | fragmentation | optional docs |
| repo `.agents/` skills | use now | repeated workflow prompts | maintenance | `.agents/skills` |
| skill descriptions / progressive disclosure | use now | token waste | poor trigger wording | `SKILL.md` metadata |
| templates / prompt templates | use now, task-mode specific | inconsistent execution | template sprawl | references/templates |
| skill-internal scripts | use now for deterministic checks | manual misses | portability | scripts |
| hooks | draft-only | missed preflight checks | false blocks, platform variance | optional adapter |
| deterministic guard scripts | use now | stale status/private package drift | incomplete coverage | scripts/tests |
| subagents / reviewer profiles | use only when approved | broad review overload | extra token and merge complexity | template/draft |
| MCP | later/draft | external context gaps | security, setup, maintenance | optional adapter |
| plugins | not now; draft-only | distribution friction later | supply-chain/trust | optional adapter |
| Codex CLI config/profiles | user-global optional only | repeated local settings | global side effects | global manual note |
| approval/sandbox/network policy | document, do not repo-configure by default | unsafe actions | environment-specific | AGENTS/template |
| Codex CLI vs IDE vs cloud vs app | support through neutral rules | tool-specific drift | capability mismatch | docs/skill |
| non-interactive `codex exec` / CI-style usage | later/draft | scheduled checks | environment/auth drift | optional adapter |
| GitHub integration / PR review | later | missed PR findings | cloud setup, permissions | docs/draft |
| memory / summaries / repo briefs | use summaries, not raw logs | forgotten rework lessons | stale memory | briefs/evidence summaries |
| recurring automation / scheduled stale checks | later | stale docs | false positives | optional adapter |
| handoff packaging scripts | use when requested | ad hoc packages | local tooling drift | `.run` + references |
| review bundle / git bundle workflow | use for review handoff | irreproducible review | package size/safety | handoff reference |
| local public-safe metadata index | later | repeated repo discovery | stale index | docs/draft |
| `.run` workflow evidence layer | use as ignored local layer | lost evidence | privacy/package risk | `.run` local tools |
| conversation logs as rework evidence | use summarized extracts only | repeating past mistakes | privacy/incompleteness | external evidence |

## Decision Rule

Prefer deterministic scripts and short skills before hooks, MCP, plugins, or
repo-local agent config. Add heavier mechanisms only after repeated evidence
shows the lighter layer is insufficient.
