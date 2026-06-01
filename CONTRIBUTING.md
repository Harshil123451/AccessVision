# Contributing to AccessVision

Thank you for your interest in contributing to AccessVision! This project aims to deliver a high-performance, accessibility-first camera assistant for visually impaired users. By contributing, you help make the digital and physical world more accessible.

---

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [How to Contribute](#how-to-contribute)
3. [Coding Standards](#coding-standards)
   - [Backend (Python / FastAPI)](#backend-python--fastapi)
   - [Frontend (TypeScript / Next.js)](#frontend-typescript--nextjs)
4. [Testing Policies](#testing-policies)
5. [Pull Request Process](#pull-request-process)

---

## Code of Conduct
We expect all contributors to adhere to a welcoming, respectful, and inclusive standard of behavior. Please treat others with respect and prioritize clear, constructive communication.

---

## How to Contribute

### 1. Find or Create an Issue
Before writing any code, search the existing issues to ensure the work is not already underway. If you want to build a new feature or fix a bug, please create a new issue detailing your proposal first.

### 2. Fork & Setup
1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/AccessVision.git
   ```
3. Set up the backend and frontend dev environments according to the local development guide in the [README.md](README.md).

### 3. Branching Conventions
Create a branch using descriptive prefixes:
- `feat/feature-name` for new features or capabilities
- `fix/bug-name` for bug fixes
- `perf/optimization-name` for latency or memory improvements
- `docs/doc-update` for documentation changes

---

## Coding Standards

### Backend (Python / FastAPI)
We follow Python PEP 8 guidelines and utilize typing for code clarity.
- **Formater**: Use `black` for formatting.
- **Imports**: Organize imports with `isort`.
- **Linting**: Keep code clean of unused imports or unresolved parameters.
- **Asynchronous Design**: All blocking model inferences MUST be wrapped in worker thread pools using `asyncio.to_thread` or delegated safely to prevent blocking the FastAPI event loop.
- **Logging**: Use the request correlation tracer (`core/telemetry.py`) to inject correlation IDs into any logs. Do not print directly; use the centralized `logger`.

### Frontend (TypeScript / Next.js)
We prioritize WCAG compliance and clean component layouts.
- **Formatting**: Use `prettier` for layout styling.
- **Linting**: Check eslint configuration before pushing (`npm run lint`).
- **Accessibility (a11y)**: Every interactive element must have appropriate ARIA attributes (`aria-live`, `aria-label`), keyboard navigation support, and high-contrast color capabilities.
- **State Management**: Manage user states or system logs through Zustand stores (`store/`). Keep components stateless where possible.

---

## Testing Policies

Before submitting a Pull Request:
1. **Manual Verification**: Run both Next.js and FastAPI servers locally and verify your changes on a mobile screen (via LAN or ngrok).
2. **Telemetry Validation**: Inspect logs during execution to ensure correlation IDs are traced and no memory leaks are triggered.
3. **Load Testing**: If modifying the inference pipeline, run Locust load tests located under `load_tests/` to guarantee tail latencies (p95/p99) do not degrade.

---

## Pull Request Process

1. **Keep PRs Focused**: Each Pull Request should address a single feature or bug fix.
2. **Environment Protection**: Make sure no sensitive keys, `.env` files, or local checkpoints (`*.pt`) are contained in the commit list.
3. **Use the Template**: Fill out the provided Pull Request Template.
4. **Code Review**: At least one maintainer must review and approve the PR before it is merged into the `main` branch.
