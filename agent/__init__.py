"""MTS Energytool consultant agent (online cycle)."""

__all__ = ["run"]


def __getattr__(name: str):
    if name == "run":
        from agent.pipeline import run

        return run
    raise AttributeError(name)
