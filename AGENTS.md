# CODEX context for BGEM library

## Project summary

Bgem library is a intended as an extension or wrapper of GMSH python API.
It helps to keep tracks of the geometry objects during geometrical operations;
it introduces `ObjectSet` class keeping the tracks of dimtags and provides wrapper
functions to GMSH API calls.

Further it provides better interface to GMSH fields, reader/writer for GMSH meshes
and a HealMesh class that can help fixing degenerated elements in a given mesh.

Submodule called `Stochastic` provides generator for DFN (Discrete Fracture Network)
based on distributions used in geology.

## Project summary (short)

Robust open source tool for creation of parametric geometries and computational meshes via. Python code. 
Primary focus are hydrogeological applications with geometries including both random fractures and deterministic natural or anthropogenic features.
Bgem library is a intended as an extension and/or wrapper of GMSH python API.

## CODEX Ignore Folders
- Not specified

## CODEX Readonly Folders
- Not specified

## Project Structure
- `src/bgem` - source files
- `tests` - pytest based unit tests of individual source blocks
- `tutorials` - integrated tests, currently insufficient

## `src/bgem/` Project Source Structure

- `bspline/` - 
- `geometry/` - 
- `gmsh/` - main functions providing interface for GMSH API
- `polygons/` - 
- `stochastic/` - code for DFN generation


## CODEX Guidelines
- treat keyword 'AGENT:' in comments as a source context dependent message for your further development
- any comment containing `AGENT:` is an active developer instruction
- NEVER remove, rewrite, or move an `AGENT:` comment unless you implement that instruction in the same change
- if you make only a local fix around an `AGENT:` comment, leave the comment untouched
- if an `AGENT:` instruction looks outdated or wrong, ask before removing it
- never stage or commit changes yourself
- Always review your changes before finishing for human review.
- Do not touch anything outside the repository directory, unless directly asked!

## Status tracking
- `STATUS.md` is the handoff log for interrupted or multi-turn work. Update it when a session ends with unfinished relevant work, when the user asks for a status review, or when a fresh checkpoint would help the next session continue without re-discovery.
- Keep newest entry first. Do not rewrite older entries except to fix clearly factual mistakes.
- Start each entry with one line in this form:
  `` `YYYY-MM-DD`: `<commit>` @ `<branch>` by `<author>` ``
- Under each entry keep exactly these sections in order:
  `## Goal`
  short statement of the intended task or checkpoint scope
  `## Changes summary`
  flat bullets describing committed changes first, then important staged/unstaged/untracked changes if they are relevant to continuing the task
  `## Verified`
  flat bullets with commands actually run and the important observed result
  `## Open items`
  flat bullets for remaining risks, missing verification, known breakage, or next recommended step
- Record only repo-relevant facts that help continuation. Skip conversational history, speculation, and incidental noise.
- When the worktree is dirty, explicitly distinguish committed, staged, unstaged, and untracked changes, but mention only files relevant to the tracked task.
- Prefer clickable file links for important files mentioned in `STATUS.md`.
- If verification was partial, say so plainly. Do not imply a full test pass when only compile checks, a single testcase, or CLI smoke checks were run.
- If a run failed, record the failing command and the actionable failure mode instead of hiding it.
- Before finishing a task that changed the practical project state, review whether `STATUS.md` still matches the actual branch/worktree state and update it if needed.

## Coding rules
- Best code, is no code!
- prefer functional style with poor functions;
  ideally do not change objects after construction, all methods do calculations
  only reading the data in the class
- prefer high level code: numpy, pandas, xarray instead loops and native python structures (lists, dicts)
- use logging
- Use logging for debug outputs.
- use pathlib
- use attrs for dataclasses
- use attrs staticmethod/classmethod technique to construct from other data then is stored in the dataclass
- Be defensive, with strong checks, but only for the user input data.
  That means error inputs must raise early. Therefore only check for existing keys in input dicts
  if these will be required down in a long calculation. Otherwise just let KeyError do the job.
- do not use "guess" default values, only obvious defaults
- Do not use other config keys in the case of a KeyError, just throw early.
- Do just basic asserts for consistency for function inputs.
- Can add more asserts if needed during debugging.
- NEVER resolve test errors by try blocks
- NEVER add runtime fallbacks or import shims to compensate for a broken or incomplete environment.
  If a declared dependency or tool is missing, report the environment problem plainly and fix the environment or tests around it, but do not implement code workarounds.
- NEVER write "self explanatory" into comments
- in comments indicate by ?? if you are not certain about intent of particular variable, function, parameter ...

