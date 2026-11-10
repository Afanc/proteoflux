# Contributing to ProteoFlux

ProteoFlux is maintained by a small team at the Biozentrum Proteomics Core Facility. Contributions and reports from researchers using the software are welcome.

## Report a problem

Open a [GitHub issue](https://github.com/Afanc/proteoflux/issues). Include the ProteoFlux and Python versions, the analysis mode, steps to reproduce, relevant configuration, and the traceback or unexpected output. A small input example is useful when it can be shared safely. Remove private research data, credentials, and identifying sample information before posting.

## Request a feature

Open an issue describing the scientific use case, affected analysis mode, expected behavior, and how the current behavior falls short. This helps us assess statistical validity and compatibility with existing results.

## Propose code changes

Fork the repository, make a focused change on a branch, and open a pull request against the project. Explain the behavior change and any numerical or output-format implications. Add or update relevant tests and documentation. Preserve existing configuration keys and result fields unless the change explicitly requires a coordinated change.

For development, use a fresh Python environment and install from the repository root:

```bash
python -m pip install -c tests/constraints.txt -e '.[dev]'
python -m pytest -q --durations=15
```

The install needs network access to fetch dependencies, including the pinned InMoose fork. The ordinary test suite uses public fixtures; optional slow and local full-data tests are described in [the testing guide](docs/agent-context/testing.md).

## Support and maintenance

Use [GitHub Issues](https://github.com/Afanc/proteoflux/issues) for public technical questions. For facility-specific service questions, use your existing contact with the Proteomics Core Facility rather than posting research data publicly. Maintainers review issues and pull requests and decide scope based on scientific validity, maintainability, compatibility, and project priorities. No response time is guaranteed.
