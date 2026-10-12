"""User-facing text. English only; ``tr`` formats a message with its arguments."""


def tr(text, *args):
    return text.format(*args) if args else text
