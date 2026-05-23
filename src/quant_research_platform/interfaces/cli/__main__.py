"""提供 ``python -m quant_research_platform.interfaces.cli`` 的 module entrypoint。"""

from quant_research_platform.interfaces.cli.commands import app

if __name__ == "__main__":
    app()
