# Agent-R1 Documentation

This directory contains the source files for the Agent-R1 documentation site, built with [MkDocs](https://www.mkdocs.org/) and the [Material theme](https://squidfunk.github.io/mkdocs-material/).

The site configuration lives at the repository root in `mkdocs.yml`.

## Quick Start

### Install Documentation Dependencies

```bash
pip install -r docs/requirements.txt
```

### Build the Documentation

Run from the repository root:

```bash
mkdocs build --clean
```

### Serve the Documentation Locally

Run from the repository root:

```bash
mkdocs serve
```

The local preview is usually available at `http://127.0.0.1:8000/Agent-R1/`.

## Structure

```text
docs/
├── README.md                  # maintenance notes for the docs directory
├── requirements.txt           # documentation dependencies
├── index.md                   # documentation homepage
├── getting-started/           # minimal setup and sanity-check flow
│   ├── index.md
│   ├── installation-guide.md
│   └── quick-start.md
├── core-concepts/             # key framework concepts
│   ├── index.md
│   ├── step-level-mdp.md
│   └── layered-abstractions.md
└── tutorials/                 # task-oriented tutorials
    ├── index.md
    └── agent-task.md
```

## Writing Documentation

### Adding a New Page

1. Create a new `.md` file under `docs/` or one of its subdirectories.
2. Add the page to the `nav` section in `mkdocs.yml`.
3. Use relative links between documentation pages when possible.

### Updating Existing Content

- Keep the documentation lightweight and focused on the most important flows.
- Prefer real repository scripts and examples over pseudo-code.
- Keep environment setup guidance aligned with `verl` instead of duplicating a separate installation guide here.

## Documentation Features

The current site uses:

- `mkdocs-material` for the theme and navigation
- `pymdown-extensions` for enhanced Markdown rendering
- Mermaid fences for diagrams
- MathJax for math rendering

## Dependencies

Documentation dependencies are listed in `docs/requirements.txt`:

- `mkdocs`
- `mkdocs-material`
- `pymdown-extensions`

## Notes

- `docs/README.md` is for maintainers and is excluded from the generated site.
- The current Agent-R1 documentation is intentionally compact and centered on the framework's core agent workflow.
