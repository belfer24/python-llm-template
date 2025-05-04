lint: ## Lint code
	uv run --frozen ruff check src  test
	uv run --frozen ruff format --check src  test 
	uv run --frozen pyright

format: ## format code
	uv run --frozen ruff check --fix src test
	uv run --frozen ruff format  src test